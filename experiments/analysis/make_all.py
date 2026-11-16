"""Bütün analiz çıktıları tek komutla (CLAUDE.md §5.19): runs.csv [+ paket logları + ek tablolar] -> results/tables/,
results/figures/.

Sırayla:
1. İstatistik (stats.py): runs.csv'deki her deney için results/tables/<deney>_stats.csv (tutarlılık kontrolüyle).
2. Grafikler (plots.py), verisi olanlar:
   - E1 (runs.csv): e1_throughput_by_class, e1_latency_by_class (makale değerleri kesik çizgi),
     e1_latency_percentiles_l<yük>; E1 paket logları varsa e1_latency_cdf_l<yük> ve E3: e3_latency_vs_hops +
     results/tables/e3_hop_slopes.csv.
   - E2: e2_scalability; E4: e4_packet_size; E8: e8_lifetime.
   - Henüz üreticisi olmayan kaynaklar (dosya varsa çizilir; sütunlar plots.py docstring'lerinde):
     results/tables/e8_root_energy.csv -> e8_root_energy, results/tables/e5_timeline.csv -> e5_root_failure,
     results/tables/training_curves.csv -> training_curves, results/tables/e9_summary.csv -> e9_pareto,
     --trained-model <json> (FuzzyParams.to_dict) -> e9_term_shapes (önce = §5.14 uzman başlangıcı).
3. Makale karşılaştırması (paper_comparison.py): results/tables/paper_comparison.csv.
Eksik deney/kaynak atlanır ve sonda uyarı listesi basılır; bir adımın hatası diğerlerini durdurmaz, ama istatistik
tutarlılık hatası (ConsistencyError) çıkış kodunu 1 yapar.

Kullanım:
    python experiments/analysis/make_all.py [--runs results/tables/runs.csv] [--packet-logs results/packet_logs]
        [--tables results/tables] [--figures results/figures] [--trained-model results/models/<run_id>.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent


def _load(name: str):
    key = f"analysis_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("stats")
pl = _load("plots")
pc = _load("paper_comparison")


def deadlines_from_config(config_path: Path) -> dict[str, float]:
    """π3 (policy.<topoloji>.max_delay_ms) -> {sınıf: ms}; sınıf-topoloji eşlemesi traffic.classes'tan."""
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return {cls: float(cfg["policy"][spec["topology"]]["max_delay_ms"])
                for cls, spec in cfg["traffic"]["classes"].items()}
    except (OSError, KeyError, TypeError):
        return {}


class Runner:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.outputs: list[Path] = []
        self.failed = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"uyarı: {msg}", file=sys.stderr)

    def step(self, name: str, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
        except st.ConsistencyError:
            self.failed = True
            raise
        except Exception:  # noqa: BLE001 — bir grafiğin hatası diğerlerini durdurmasın; sonda raporlanır
            self.failed = True
            self.warn(f"{name} başarısız:\n{traceback.format_exc()}")
            return None
        for p in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(p, Path):
                self.outputs.append(p)
                print(f"-> {p}")
        return out


def run_all(runs_path: Path, packet_dir: Path, tables: Path, figures: Path, reference: Path, config: Path,
            trained_model: Path | None = None) -> Runner:
    R = Runner()
    runs = st.load_runs(runs_path) if runs_path.exists() else None
    if runs is None:
        R.warn(f"{runs_path} yok: istatistik ve runs.csv grafikleri atlandı")

    # 1. istatistik
    if runs is not None:
        stats_tables = st.compute_stats(runs)
        if not stats_tables:
            R.warn("runs.csv'de istatistiği alınacak deney yok")
        R.step("stats", st.write_stats, stats_tables, tables)

    def exp(name: str) -> pd.DataFrame | None:
        if runs is None:
            return None
        d = runs[runs["experiment"] == name]
        if d.empty:
            R.warn(f"{name} runs.csv'de yok: {name} grafikleri atlandı")
            return None
        return d

    ref = pc.load_reference(reference)
    deadlines = deadlines_from_config(config)

    # 2. grafikler
    e1 = exp("E1")
    if e1 is not None:
        R.step("e1_throughput_by_class", pl.bar_by_class, e1, "norm_throughput", figures / "e1_throughput_by_class.png",
               "Normalize throughput (%)", scale=100.0, paper=_paper_pair(ref, "throughput_pct_by_class"))
        R.step("e1_latency_by_class", pl.bar_by_class, e1, "lat_mean_ms", figures / "e1_latency_by_class.png",
               "Ortalama uçtan uca gecikme (ms)", paper=_paper_pair(ref, "latency_ms_by_class"))
        for load in sorted(e1["load_multiplier"].dropna().unique()):
            R.step(f"e1_latency_percentiles l{load:g}", pl.latency_percentiles, e1,
                   figures / f"e1_latency_percentiles_l{load:g}.png", load=load)
    packets = pc.load_packet_logs(packet_dir, runs, "E1") if runs is not None else None
    if packets is None:
        R.warn(f"E1 paket logu yok ({packet_dir}): e1_latency_cdf ve E3 atlandı")
    else:
        for load in sorted(packets["load_multiplier"].dropna().unique()):
            R.step(f"e1_latency_cdf l{load:g}", pl.latency_cdf, packets[packets["load_multiplier"] == load],
                   figures / f"e1_latency_cdf_l{load:g}.png", deadlines_ms=deadlines)
        res = R.step("e3_latency_vs_hops", pl.latency_vs_hops, packets, figures / "e3_latency_vs_hops.png",
                     paper=_paper_curve(ref, "latency_vs_hops_ms"))
        if res is not None:
            tables.mkdir(parents=True, exist_ok=True)
            res[1].to_csv(tables / "e3_hop_slopes.csv", index=False)
            R.outputs.append(tables / "e3_hop_slopes.csv")
            print(f"-> {tables / 'e3_hop_slopes.csv'}")

    e2 = exp("E2")
    if e2 is not None:
        R.step("e2_scalability", pl.scalability, e2, figures / "e2_scalability.png",
               paper=_paper_curve(ref, "scalability_latency_ms"))
    e4 = exp("E4")
    if e4 is not None:
        col = pc.find_packet_size_column(e4)
        if col is None:
            R.warn("E4'te paket boyutu ızgara sütunu (grid_*packet_bytes) yok: e4_packet_size atlandı")
        else:
            R.step("e4_packet_size", pl.scalability, e4, figures / "e4_packet_size.png", x_col=col,
                   xlabel="Kritik paket boyutu (B)", log_x=True, paper=_paper_curve(ref, "latency_vs_packet_size_ms"))
    e8 = exp("E8")
    if e8 is not None:
        R.step("e8_lifetime", pl.lifetime, e8, figures / "e8_lifetime.png")

    for fname, label, fn in (
        ("e8_root_energy.csv", "e8_root_energy", pl.root_energy),
        ("e5_timeline.csv", "e5_root_failure", pl.root_failure_timeline),
        ("training_curves.csv", "training_curves", pl.training_curves),
        ("e9_summary.csv", "e9_pareto", pl.e9_pareto),
    ):
        src = tables / fname
        if src.exists():
            R.step(label, fn, pd.read_csv(src), figures / f"{label}.png")
        else:
            R.warn(f"{src} yok: {label} atlandı")
    if trained_model is not None and trained_model.exists():
        after = json.loads(trained_model.read_text(encoding="utf-8"))
        before = pl.expert_params([len(c) for c in after["centers"]])
        R.step("e9_term_shapes", pl.term_shapes, before, after, figures / "e9_term_shapes.png")
    else:
        R.warn("eğitilmiş model verilmedi (--trained-model): e9_term_shapes atlandı")

    # 3. makale karşılaştırması
    def _paper_table():
        out = tables / "paper_comparison.csv"
        tables.mkdir(parents=True, exist_ok=True)
        pc.build_comparison(runs, ref, packets).to_csv(out, index=False)
        return out

    R.step("paper_comparison", _paper_table)
    return R


def _paper_pair(ref: dict, key: str) -> dict:
    """Bar grafiği için makale değerleri: {'NF-MTAR': {...}, 'RPL': {...}}."""
    return {k: v for k, v in ref.get(key, {}).items()}


def _paper_curve(ref: dict, key: str) -> dict:
    return {k: {int(x): float(y) for x, y in v.items()} for k, v in ref.get(key, {}).items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(PROJECT_ROOT / "results" / "tables" / "runs.csv"))
    parser.add_argument("--packet-logs", default=str(PROJECT_ROOT / "results" / "packet_logs"))
    parser.add_argument("--tables", default=str(PROJECT_ROOT / "results" / "tables"))
    parser.add_argument("--figures", default=str(PROJECT_ROOT / "results" / "figures"))
    parser.add_argument("--reference", default=str(PROJECT_ROOT / "configs" / "paper_reference.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--trained-model", default=None, help="results/models/<run_id>.json (e9_term_shapes için)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # Windows konsolunda Türkçe karakterler bozulmasın
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        R = run_all(Path(args.runs), Path(args.packet_logs), Path(args.tables), Path(args.figures),
                    Path(args.reference), Path(args.config), Path(args.trained_model) if args.trained_model else None)
    except st.ConsistencyError as e:
        print(f"HATA: istatistik tutarlılık kontrolü: {e}", file=sys.stderr)
        return 1
    print(f"\n{len(R.outputs)} çıktı; {len(R.warnings)} uyarı")
    for w in R.warnings:
        print(f"  - {w.splitlines()[0]}")
    return 1 if R.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

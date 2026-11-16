"""Makale karşılaştırması (CLAUDE.md §5.19 "Makale karşılaştırması"): configs/paper_reference.yaml + runs.csv
[+ E1 paket logları] -> results/tables/paper_comparison.csv.

Sütunlar: metric | x (sınıf / N / atlama / bayt) | load | paper_nfmtar | paper_baseline (+ adı) | ours_nfmtar |
ours_baseline (+ adı) | ours_n_seeds | rel_impr_paper | rel_impr_ours | trend_agreement | note.

- Bizim değerimiz: ilgili deneyin seed ortalaması. Makalenin RPL'si <-> bizim B1-plain (paper_reference.yaml
  method_mapping; §5.12). TSFIS-GWO'nun bizde karşılığı yok: yalnızca makale sütunu doldurulur.
- Göreli iyileşme, pozitif = NF-MTAR daha iyi: büyük-iyi metrikte (NF − taban)/taban, küçük-iyi (gecikme) metrikte
  (taban − NF)/taban. Makale ve bizim için ayrı ayrı; birebir sayısal uyum beklenmez (§2).
- Trend uyumu: ikili satırlarda iki iyileşmenin işareti aynı mı; eğri satırlarında (ölçeklenebilirlik, atlama,
  paket boyutu) ayrıca `<eğri>_slope` özet satırı: makale ve bizim doğrusal eğimin işareti aynı mı.
- E1 satırları runs.csv'deki her yük için ayrı yazılır (makale yükü belirtmiyor; seçim raporda yapılır).
- Makalenin Tablo 4'ündeki toplu throughput / gecikme / QoS'un bizdeki karşılığı tanımsız (makale tanımlamıyor):
  yalnızca PDR (pdr_total) eşlenir, diğerlerinde bizim sütunumuz boş ve `note` açıklar.
- paper_reference.yaml yoksa §5.19'daki değerlerle (DEFAULT_REFERENCE) oluşturulur.

Kullanım:
    python experiments/analysis/paper_comparison.py [--runs results/tables/runs.csv] [--reference configs/paper_reference.yaml]
        [--packet-logs results/packet_logs] [--out results/tables/paper_comparison.csv]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("critical", "regular", "noncritical")
OURS_NF = "NF-MTAR"
OURS_BASE = "B1-plain"

# §5.19 değerleri (paper_reference.yaml yoksa bu yazılır)
DEFAULT_REFERENCE: dict[str, Any] = {
    "method_mapping": {"NF-MTAR": "NF-MTAR", "RPL": "B1-plain", "TSFIS-GWO": None},
    "classes": list(CLASSES),
    "throughput_pct_by_class": {
        "NF-MTAR": {"critical": 95.2, "regular": 87.3, "noncritical": 78.9},
        "RPL": {"critical": 71.4, "regular": 68.2, "noncritical": 62.1},
    },
    "latency_ms_by_class": {
        "NF-MTAR": {"critical": 18.4, "regular": 45.2, "noncritical": 89.7},
        "RPL": {"critical": 67.3, "regular": 94.8, "noncritical": 156.2},
    },
    "table4": {
        "NF-MTAR": {"throughput_pct": 94.3, "latency_ms": 18.4, "pdr_pct": 98.7, "qos_pct": 97.3},
        "TSFIS-GWO": {"throughput_pct": 86.4, "latency_ms": 38.9, "pdr_pct": 97.2, "qos_pct": 93.8},
    },
    "scalability_latency_ms": {"NF-MTAR": {20: 12, 50: 15, 100: 20, 150: 24, 200: 28}},
    "latency_vs_hops_ms": {
        "NF-MTAR": {1: 8, 2: 12, 3: 16, 5: 24, 10: 42},
        "RPL": {1: 28, 2: 48, 3: 72, 5: 120, 10: 200},
    },
    "latency_vs_packet_size_ms": {"NF-MTAR": {64: 9, 128: 13, 256: 17, 512: 23, 1024: 29}},
}


def load_reference(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Makalenin raporladığı değerler (CLAUDE.md §5.19); paper_comparison.py tarafından oluşturuldu.\n"
                        + yaml.safe_dump(DEFAULT_REFERENCE, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"uyarı: {path} yoktu, §5.19 değerleriyle oluşturuldu", file=sys.stderr)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def rel_improvement(nf: float, base: float, higher_is_better: bool) -> float:
    if not (np.isfinite(nf) and np.isfinite(base)) or base == 0:
        return math.nan
    return (nf - base) / base if higher_is_better else (base - nf) / base


def same_sign(a: float, b: float) -> float:
    """1.0 / 0.0 (trend uyumu), tanımsızsa NaN (CSV'de boş)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return math.nan
    return float(np.sign(a) == np.sign(b))


def _seed_mean(df: pd.DataFrame, method: str, col: str) -> tuple[float, int]:
    if df is None or col not in df.columns:
        return math.nan, 0
    v = df.loc[df["method"] == method, col].dropna()
    return (float(v.mean()), int(len(v))) if len(v) else (math.nan, 0)


def _row(metric, x, paper_nf, paper_base, paper_base_name, ours_nf, ours_base, n, higher, load=math.nan, note=""):
    rp = rel_improvement(paper_nf, paper_base, higher)
    ro = rel_improvement(ours_nf, ours_base, higher)
    return {"metric": metric, "x": x, "load": load, "paper_nfmtar": paper_nf, "paper_baseline": paper_base,
            "paper_baseline_name": paper_base_name, "ours_nfmtar": ours_nf, "ours_baseline": ours_base,
            "ours_baseline_name": OURS_BASE if np.isfinite(ours_base) else "", "ours_n_seeds": n,
            "rel_impr_paper": rp, "rel_impr_ours": ro, "trend_agreement": same_sign(rp, ro), "note": note}


def _slope(xs, ys) -> float:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else math.nan


def _curve_rows(metric: str, paper: Mapping, ours_nf: Mapping, ours_base: Mapping | None, baseline_name: str,
                note: str) -> list[dict]:
    """Eğri (x -> gecikme) satırları + eğim özeti. ours_*: {x: (değer, n)}."""
    rows = []
    p_nf = {int(k): float(v) for k, v in paper.get("NF-MTAR", {}).items()}
    p_base = {int(k): float(v) for k, v in paper.get(baseline_name, {}).items()} if baseline_name else {}
    for x in sorted(p_nf):
        o_nf, n = ours_nf.get(x, (math.nan, 0))
        o_b = (ours_base or {}).get(x, (math.nan, 0))[0]
        rows.append(_row(metric, x, p_nf[x], p_base.get(x, math.nan), baseline_name if p_base else "",
                         o_nf, o_b, n, higher=False, note=note))
    s_paper = _slope(list(p_nf), list(p_nf.values()))
    xs = sorted(ours_nf)
    s_ours = _slope(xs, [ours_nf[x][0] for x in xs])
    rows.append({"metric": f"{metric}_slope", "x": "", "load": math.nan, "paper_nfmtar": s_paper,
                 "paper_baseline": math.nan, "paper_baseline_name": "", "ours_nfmtar": s_ours,
                 "ours_baseline": math.nan, "ours_baseline_name": "", "ours_n_seeds": len(xs),
                 "rel_impr_paper": math.nan, "rel_impr_ours": math.nan, "trend_agreement": same_sign(s_paper, s_ours),
                 "note": "doğrusal eğim (ms / x birimi), NF-MTAR; trend = eğim işareti aynı mı"})
    return rows


def _by_x(df: pd.DataFrame | None, method: str, x_col: str | None, y_col: str) -> dict[int, tuple[float, int]]:
    if df is None or x_col is None or x_col not in df.columns or y_col not in df.columns:
        return {}
    sub = df[df["method"] == method].dropna(subset=[y_col])
    return {int(x): (float(g[y_col].mean()), len(g)) for x, g in sub.groupby(x_col)}


def find_packet_size_column(df: pd.DataFrame | None) -> str | None:
    """E4'te kritik paket boyutu run_batch ızgarasından gelir: grid_traffic.classes.critical.packet_bytes."""
    if df is None:
        return None
    cands = [c for c in df.columns if c.startswith("grid_") and c.endswith("packet_bytes")]
    return cands[0] if cands else None


def build_comparison(runs: pd.DataFrame | None, ref: Mapping[str, Any], packets: pd.DataFrame | None = None,
                     experiments: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Karşılaştırma tablosu. experiments: {'E1': etiket, 'E2': ..., 'E4': ...} (runs.csv experiment değerleri)."""
    exps = {"E1": "E1", "E2": "E2", "E4": "E4", **(experiments or {})}
    base_name = next((k for k, v in ref.get("method_mapping", {}).items() if v == OURS_BASE), "RPL")

    def exp_df(key):
        if runs is None or "experiment" not in runs:
            return None
        d = runs[runs["experiment"].astype(str) == exps[key]]
        return d if len(d) else None

    rows: list[dict] = []
    e1 = exp_df("E1")
    loads = sorted(e1["load_multiplier"].dropna().unique()) if e1 is not None else [math.nan]
    # Şekil 4a / 5a: sınıf bazında throughput ve gecikme
    for metric, ref_key, col, scale, higher in (
        ("throughput_pct", "throughput_pct_by_class", "norm_throughput", 100.0, True),
        ("latency_ms", "latency_ms_by_class", "lat_mean_ms", 1.0, False),
    ):
        pr = ref.get(ref_key, {})
        for load in loads:
            sub = e1[e1["load_multiplier"] == load] if e1 is not None else None
            for cls in CLASSES:
                o_nf, n = _seed_mean(sub, OURS_NF, f"{col}_{cls}")
                o_b, _ = _seed_mean(sub, OURS_BASE, f"{col}_{cls}")
                rows.append(_row(metric, cls, pr.get("NF-MTAR", {}).get(cls, math.nan),
                                 pr.get(base_name, {}).get(cls, math.nan), base_name, o_nf * scale, o_b * scale, n,
                                 higher, load=load,
                                 note="throughput = teslim edilen bayt / üretilen bayt (§5.13)" if higher else ""))
    # Tablo 4: NF-MTAR vs TSFIS-GWO (bizde TSFIS-GWO yok)
    t4 = ref.get("table4", {})
    t4_base = next((k for k in t4 if k != "NF-MTAR"), "")
    for metric, higher in (("throughput_pct", True), ("latency_ms", False), ("pdr_pct", True), ("qos_pct", True)):
        for load in loads:
            sub = e1[e1["load_multiplier"] == load] if e1 is not None else None
            if metric == "pdr_pct":
                o_nf, n = _seed_mean(sub, OURS_NF, "pdr_total")
                o_b, _ = _seed_mean(sub, OURS_BASE, "pdr_total")
                o_nf, o_b, note = o_nf * 100, o_b * 100, "bizim taban B1-plain (makalede TSFIS-GWO)"
            else:
                o_nf = o_b = math.nan
                n = 0
                note = "makale toplu değerin tanımını vermiyor; bizde eşlenmedi"
            r = _row(f"table4_{metric}", "all", t4.get("NF-MTAR", {}).get(metric, math.nan),
                     t4.get(t4_base, {}).get(metric, math.nan), t4_base, o_nf, o_b, n, higher, load=load, note=note)
            rows.append(r)
    # Şekil 6b: ölçeklenebilirlik (E2), kritik sınıf ortalama gecikmesi
    e2 = exp_df("E2")
    rows += _curve_rows("scalability_latency_ms", ref.get("scalability_latency_ms", {}),
                        _by_x(e2, OURS_NF, "n_nodes", "lat_mean_ms_critical"), None, "",
                        note="bizim: E2, kritik sınıf ort. gecikme (makale sınıfı belirtmiyor)")
    # Şekil 6d: atlama sayısı (E3 = E1 paket logları)
    if packets is not None and len(packets):
        pk = packets
        if "delivered" in pk:
            pk = pk[pk["delivered"].astype(bool)]
        if "after_warm_up" in pk:
            pk = pk[pk["after_warm_up"].astype(bool)]
        hop_nf = {int(h): (float(g["latency_ms"].mean()), len(g)) for h, g in pk[pk["method"] == OURS_NF].groupby("hops")}
        hop_b = {int(h): (float(g["latency_ms"].mean()), len(g)) for h, g in pk[pk["method"] == OURS_BASE].groupby("hops")}
    else:
        hop_nf, hop_b = {}, {}
    rows += _curve_rows("latency_vs_hops_ms", ref.get("latency_vs_hops_ms", {}), hop_nf, hop_b, base_name,
                        note="bizim: E1 paket logları, bütün sınıflar; ours_n_seeds = paket sayısı")
    # Şekil 6c: kritik paket boyutu (E4)
    e4 = exp_df("E4")
    rows += _curve_rows("latency_vs_packet_size_ms", ref.get("latency_vs_packet_size_ms", {}),
                        _by_x(e4, OURS_NF, find_packet_size_column(e4), "lat_mean_ms_critical"), None, "",
                        note="bizim: E4, kritik sınıf ort. gecikme")
    return pd.DataFrame(rows)


def load_packet_logs(directory: str | Path, runs: pd.DataFrame | None, experiment: str = "E1") -> pd.DataFrame | None:
    """results/packet_logs/<run_id>.parquet|.csv.gz -> tek tablo + method (runs.csv'deki run_id eşlemesinden)."""
    directory = Path(directory)
    files = sorted([*directory.glob("*.parquet"), *directory.glob("*.csv.gz")]) if directory.exists() else []
    if not files or runs is None:
        return None
    meta = runs[["run_id", "method", "experiment", "seed", "load_multiplier"]].astype({"experiment": str})
    meta = meta[meta["experiment"] == experiment].set_index("run_id")
    parts = []
    for f in files:
        rid = f.name.removesuffix(".parquet").removesuffix(".csv.gz")
        if rid not in meta.index:
            continue
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        m = meta.loc[rid]
        parts.append(df.assign(run_id=rid, method=m["method"], seed=m["seed"], load_multiplier=m["load_multiplier"]))
    return pd.concat(parts, ignore_index=True) if parts else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(PROJECT_ROOT / "results" / "tables" / "runs.csv"))
    parser.add_argument("--reference", default=str(PROJECT_ROOT / "configs" / "paper_reference.yaml"))
    parser.add_argument("--packet-logs", default=str(PROJECT_ROOT / "results" / "packet_logs"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "tables" / "paper_comparison.csv"))
    args = parser.parse_args(argv)
    ref = load_reference(args.reference)
    runs = pd.read_csv(args.runs) if Path(args.runs).exists() else None
    if runs is None:
        print(f"uyarı: {args.runs} yok; yalnızca makale sütunları yazılır", file=sys.stderr)
    table = build_comparison(runs, ref, load_packet_logs(args.packet_logs, runs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(table)} satır)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

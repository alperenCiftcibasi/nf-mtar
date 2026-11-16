"""M6 çalışma noktası kalibrasyonu: başlangıç enerjisi ve yük ızgarası seçimi, baseline özeti (CLAUDE.md §5.18a, §5.13).

Koşular experiments/run_batch.py ile yapılır ve runs.csv'ye yazılır; bu script yalnızca okur, kuralı uygular, tabloları
yazar ve (--write ile) seçimi config'e yazar. Kalibrasyon B1 ile yapılır (yöntemlerden bağımsız olsun, §5.18a).

Adımlar (alt komutlar):
  energy    §5.18a-2: {5, 10, 20, 50} J içinden, B1 yük 2,0'da 5 seed'in HEPSİNDE FND > süre olan (hiç ölüm yok) en küçük
            değer -> energy.initial_J. Koşular: --experiment m6_energy --grid energy.initial_J=5,10,20,50.
  load      §5.18a-3: B1 yük 1,0'da sınıf PDR'leri hep > 0,98 ise ızgara {1, 2, 4}; herhangi bir sınıfta < 0,80 ise
            {0,25, 0,5, 1}; aksi halde {0,5, 1, 2} -> experiments.E1.load_multipliers. "hep" ve "herhangi" iki okumayla
            değerlendirilir: (a) seed ortalaması, (b) her seed ayrı (en küçük seed değeri). İki okuma farklı ızgara
            verirse seçim yapılmaz (karar gerektirir) ve script hata ile çıkar.
  baselines §5.18a sonrası: seçilen noktada B1-plain, B1, B2 × ızgara yükleri × 5 seed; yöntem × yük × sınıf özeti
            (seed ortalaması ± std): PDR, ort./p95 gecikme, süre sınırı oranı, airtime payı, Φ_t ve Φ (§5.13),
            kayıp nedenleri.

Çıktılar: results/tables/m6_calibration.csv (step sütunu: energy / load), results/tables/m6_baselines.csv.

Kullanım:
    python experiments/calibration/m6_operating_point.py energy [--write]
    python experiments/calibration/m6_operating_point.py load [--write]
    python experiments/calibration/m6_operating_point.py baselines
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfmtar.config import load_config  # noqa: E402
from nfmtar.simulation import DROP_REASONS  # noqa: E402

CLASSES = ("critical", "regular", "noncritical")
ENERGY_GRID = (5.0, 10.0, 20.0, 50.0)            # §5.18a-2
LOAD_HIGH, LOAD_LOW = 0.98, 0.80                 # §5.18a-3 eşikleri
GRID_UP, GRID_DOWN, GRID_MID = [1.0, 2.0, 4.0], [0.25, 0.5, 1.0], [0.5, 1.0, 2.0]
CALIB_TABLE = PROJECT_ROOT / "results" / "tables" / "m6_calibration.csv"
BASELINE_TABLE = PROJECT_ROOT / "results" / "tables" / "m6_baselines.csv"


def runs(cfg, experiment: str) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg.run.runs_table
    df = pd.read_csv(path)
    df = df[df["experiment"] == experiment]
    if df.empty:
        raise SystemExit(f"{path}: experiment {experiment!r} satırı yok; önce run_batch.py ile koşturun")
    return df


def write_table(rows: pd.DataFrame, step: str) -> None:
    """m6_calibration.csv'de bu adımın satırlarını değiştirir (diğer adım korunur)."""
    rows = rows.assign(step=step)
    if CALIB_TABLE.exists():
        old = pd.read_csv(CALIB_TABLE)
        rows = pd.concat([old[old["step"] != step], rows], ignore_index=True)
    cols = ["step"] + [c for c in rows.columns if c != "step"]
    rows[cols].to_csv(CALIB_TABLE, index=False, float_format="%.4f")


def replace_yaml_line(pattern: str, replacement: str) -> None:
    """configs/default.yaml'da tek satırı yerinde değiştirir (yorumlar korunur)."""
    path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8", newline="") as fh:   # satır sonları (CRLF) korunur
        text = fh.read()
    new, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"{path}: desen {pattern!r} {n} kez eşleşti (1 bekleniyordu)")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)


# --------------------------------------------------------------------------- #


def step_energy(cfg, write: bool) -> None:
    df = runs(cfg, "m6_energy")
    duration = float(df["duration_s"].iloc[0])
    rows = []
    for e, g in df.groupby("grid_energy.initial_J"):
        fnd = g["fnd_s"].astype(float)
        rows.append({
            "initial_J": float(e), "load": float(g["load_multiplier"].iloc[0]), "n_seeds": len(g),
            "seeds_with_death": int(fnd.notna().sum()), "fnd_min_s": float(fnd.min()) if fnd.notna().any() else math.nan,
            "dead_nodes_mean": float(g["dead_nodes"].mean()), "dead_nodes_max": int(g["dead_nodes"].max()),
            "energy_total_J_mean": float(g["energy_total_J"].mean()),
            **{f"pdr_{c}_mean": float(g[f"pdr_{c}"].mean()) for c in CLASSES},
        })
    table = pd.DataFrame(rows).sort_values("initial_J")
    ok = table[(table["seeds_with_death"] == 0) & (table["n_seeds"] >= 5)]
    chosen = float(ok["initial_J"].min()) if not ok.empty else math.nan
    table["selected"] = table["initial_J"] == chosen
    write_table(table, "energy")
    print(f"energy step (B1, load {table['load'].iloc[0]:g}, duration {duration:g} s):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if math.isnan(chosen):
        raise SystemExit("hiçbir enerji değeri 5 seed'in hepsinde ölümsüz değil: §5.18a-2 ızgarası yetmiyor (karar gerekir)")
    print(f"-> selected energy.initial_J = {chosen:g} J")
    if write:
        replace_yaml_line(r"^(  initial_J: )[0-9.]+[^\r\n]*",
                          rf"\g<1>{chosen:.1f}                        # §5.18a-2 M6 kalibrasyonu (B1 yük 2,0'da 5 seed ölümsüz en küçük; M5: 2 J); E8'de 0.5 J")
        print("-> configs/default.yaml energy.initial_J written")


def classify(pdrs: dict[str, float]) -> list[float]:
    """§5.18a-3 kuralı: sınıf PDR'leri -> ızgara."""
    if all(v > LOAD_HIGH for v in pdrs.values()):
        return GRID_UP
    if any(v < LOAD_LOW for v in pdrs.values()):
        return GRID_DOWN
    return GRID_MID


def step_load(cfg, write: bool) -> None:
    df = runs(cfg, "m6_load")
    rows = []
    for load, g in df.groupby("load_multiplier"):
        row = {"load": float(load), "n_seeds": len(g), "dead_nodes_max": int(g["dead_nodes"].max())}
        for c in CLASSES:
            row[f"pdr_{c}_mean"] = float(g[f"pdr_{c}"].mean())
            row[f"pdr_{c}_min"] = float(g[f"pdr_{c}"].min())
            row[f"deadline_{c}_mean"] = float(g[f"deadline_ratio_{c}"].mean())
            row[f"lat_p95_ms_{c}_mean"] = float(g[f"lat_p95_ms_{c}"].mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("load")
    at1 = table[table["load"] == 1.0]
    if at1.empty:
        raise SystemExit("yük 1,0 koşusu yok")
    r = at1.iloc[0]
    by_mean = classify({c: r[f"pdr_{c}_mean"] for c in CLASSES})
    by_seed = classify({c: r[f"pdr_{c}_min"] for c in CLASSES})
    # her seed ayrı: "hep > 0,98" = en küçük seed değeri > 0,98; "herhangi < 0,80" = herhangi bir seed'de < 0,80
    table["rule_by_mean"] = str(by_mean)
    table["rule_by_seed"] = str(by_seed)
    write_table(table, "load")
    print("load step (B1):")
    print(table.drop(columns=["rule_by_mean", "rule_by_seed"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"rule at load 1.0: seed mean -> {by_mean}; per seed -> {by_seed}")
    if by_mean != by_seed:
        raise SystemExit("iki okuma farklı ızgara veriyor: §5.18a-3 'hep/herhangi' yorumu için karar gerekir")
    print(f"-> selected load grid {by_mean}")
    if write:
        replace_yaml_line(r"^(    load_multipliers: )\[[^\]]*\]",
                          rf"\g<1>[{', '.join(f'{v:g}' for v in by_mean)}]")
        print("-> configs/default.yaml experiments.E1.load_multipliers written")


def step_baselines(cfg) -> None:
    df = runs(cfg, "m6_baselines")
    metrics = ["pdr", "lat_mean_ms", "lat_p95_ms", "deadline_ratio", "window_ratio", "airtime_share", "phi",
               *[f"drop_{r}" for r in DROP_REASONS]]
    rows = []
    for (method, load), g in df.groupby(["method", "load_multiplier"]):
        for c in CLASSES:
            row = {"method": method, "load": float(load), "class": c, "n_seeds": len(g)}
            for m in metrics:
                v = g[f"{m}_{c}"].astype(float)
                row[m] = float(v.mean())
                if m in ("pdr", "lat_mean_ms", "lat_p95_ms", "deadline_ratio", "phi"):
                    row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else math.nan
            row["phi_total"] = float(g["phi"].mean())      # §5.13 Φ = Σ_t Φ_t / 3 (yöntem × yük için aynı)
            row["dead_nodes_mean"] = float(g["dead_nodes"].mean())
            row["energy_total_J_mean"] = float(g["energy_total_J"].mean())
            rows.append(row)
    order = {"B1-plain": 0, "B1": 1, "B2": 2}
    table = pd.DataFrame(rows)
    table = table.sort_values(["load", "method", "class"], key=lambda s: s.map(order) if s.name == "method" else
                              (s.map({c: i for i, c in enumerate(CLASSES)}) if s.name == "class" else s))
    table.to_csv(BASELINE_TABLE, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    show = ["load", "method", "class", "pdr", "lat_mean_ms", "lat_p95_ms", "deadline_ratio", "airtime_share", "phi",
            "phi_total", *[f"drop_{r}" for r in DROP_REASONS]]
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"-> {BASELINE_TABLE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["energy", "load", "baselines"])
    parser.add_argument("--write", action="store_true", help="secimi configs/default.yaml'a yaz")
    args = parser.parse_args(argv)
    cfg = load_config()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if args.step == "energy":
        step_energy(cfg, args.write)
    elif args.step == "load":
        step_load(cfg, args.write)
    else:
        step_baselines(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

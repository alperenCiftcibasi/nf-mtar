"""E9-lite: eski veri (data/e9_lite.npz) ile zengin veri (data/e9_rich.npz) yan yana (CLAUDE.md §5.21, P1 adım 7).

Girdi: results/tables/e9_lambda.csv (eski veri, λ taraması; burada λ = 1e-2 satırları, none ve B3 alınır) ve
results/tables/e9_lambda_rich.csv (experiments/e9_lambda.py --data data/e9_rich.npz --lams 1e-2 --tag rich).
Her koşul (B3, anchored, zero_l2, none) için iki veri kümesindeki test ölçütleri (MSE, Spearman, top-1, pişmanlık) ve
§5.20 yorumlanabilirlik ölçütleri. Test kümeleri farklıdır (eski: seed 6; zengin: seed 118–120), bu yüzden mutlak
değerler değil, koşulların kendi veri kümesindeki B3'e göre farkı (Δ pişmanlık = koşul − B3) karşılaştırılır.

Çıktı: results/tables/e9_rich_vs_lite.csv
Kullanım: python experiments/analysis/e9_rich_vs_lite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = PROJECT_ROOT / "results" / "tables"
COLS = ["test_mse", "test_spearman", "test_top1", "test_regret", "order_violations", "coverage_min",
        "distinguishability_max", "monotonicity_violation", "dev_c", "dev_sigma", "dev_y", "best_epoch"]


def pick(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["condition"].isin(["B3", "none"])) | ((df["lambda"] - 1e-2).abs() < 1e-9)].copy()
    b3 = float(df.loc[df["condition"] == "B3", "test_regret"].iloc[0])
    df["regret_minus_B3"] = df["test_regret"] - b3
    df.insert(0, "dataset", dataset)
    return df[["dataset", "condition", "lambda", "n_reps", *COLS, "regret_minus_B3"]]


def main() -> int:
    old, new = TABLES / "e9_lambda.csv", TABLES / "e9_lambda_rich.csv"
    for p in (old, new):
        if not p.exists():
            print(f"eksik: {p}", file=sys.stderr)
            return 1
    df = pd.concat([pick(old, "e9_lite"), pick(new, "e9_rich")], ignore_index=True)
    order = {"B3": 0, "anchored": 1, "zero_l2": 2, "none": 3}
    df = df.sort_values(["condition", "dataset"], key=lambda s: s.map(order) if s.name == "condition" else s)
    out = TABLES / "e9_rich_vs_lite.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

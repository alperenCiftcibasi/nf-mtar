"""Eğitim verisi özeti (CLAUDE.md §5.16, §5.21): girdilerin (x1–x5) ve kâhin skorunun dağılımı, veri kümeleri yan yana.

Her veri kümesi (varsayılan data/e9_lite.npz ve data/e9_rich.npz) ve her değişken için: ortalama, std, %5 / %50 / %95
yüzdelikleri, tam sıfır oranı ve grup içi std (aynı ağ durumu × topoloji içindeki adaylar arası std'nin ortalaması;
root seçimi grup içi sıralama olduğu için asıl bilgi taşıyan değişkenlik budur). Ayrıca satır, grup, seed sayısı ve
ortalama grup boyutu.

Çıktı: results/tables/e9_data_summary.csv
Kullanım: python experiments/analysis/e9_data_summary.py [--data data/e9_lite.npz data/e9_rich.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.engine.base import FEATURE_NAMES  # noqa: E402

TABLE = PROJECT_ROOT / "results" / "tables" / "e9_data_summary.csv"


def within_group_std(v: np.ndarray, groups: np.ndarray) -> float:
    """Grup içi std'nin grup ortalaması (grup boyutu >= 2)."""
    df = pd.DataFrame({"v": v, "g": groups})
    s = df.groupby("g")["v"].std(ddof=0)
    return float(s.mean())


def summarize(path: Path) -> list[dict]:
    d = np.load(path)
    X, y, g = d["X"], d["y"], d["group"]
    meta = {"dataset": path.stem, "rows": len(y), "groups": len(np.unique(g)), "seeds": len(np.unique(d["seed"])),
            "mean_group_size": len(y) / len(np.unique(g))}
    rows = []
    for name, v in [*((n, X[:, i]) for i, n in enumerate(FEATURE_NAMES)), ("y_oracle", y)]:
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        rows.append({**meta, "variable": name, "mean": v.mean(), "std": v.std(), "p5": p5, "p50": p50, "p95": p95,
                     "frac_zero": float(np.mean(v == 0)), "within_group_std": within_group_std(v, g)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=["data/e9_lite.npz", "data/e9_rich.npz"])
    args = parser.parse_args(argv)
    rows = []
    for p in args.data:
        path = PROJECT_ROOT / p
        if not path.exists():
            print(f"atlandı (yok): {path}")
            continue
        rows += summarize(path)
    df = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE, index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"-> {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M6 grafikleri ve B3 özeti (CLAUDE.md §5.13, §5.18a).

1. results/tables/b3_check.csv: runs.csv'deki b3_check koşularından (B3, yük 0,25 / 0,5, 5 seed) m6_baselines.csv
   sütunlarıyla yük × sınıf özeti + root değişim sayısı (topoloji başına ve toplam).
2. results/figures/m6_pdr_by_class.png ve m6_latency_by_class.png: sınıf başına panel, x = yük, gruplu çubuk = yöntem
   (B1-plain, B1, B2 ve varsa B3); seed ortalaması, hata çubuğu = seed std. Gecikme panelinde ortalama (çubuk) ve
   p95 (nokta).

Kullanım: python experiments/analysis/m6_plots.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.simulation import DROP_REASONS  # noqa: E402

TABLES = PROJECT_ROOT / "results" / "tables"
FIGS = PROJECT_ROOT / "results" / "figures"
CLASSES = ("critical", "regular", "noncritical")
METHOD_ORDER = ["B1-plain", "B1", "B2", "B3"]
COLORS = {"B1-plain": "#9e9e9e", "B1": "#4c72b0", "B2": "#55a868", "B3": "#c44e52"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """m6_baselines.csv biçiminde yöntem × yük × sınıf özeti (+ root değişimleri)."""
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
            row["phi_total"] = float(g["phi"].mean())
            for t in ("E", "L", "R"):
                row[f"root_changes_{t}"] = float(g[f"root_changes_{t}"].mean())
            row["root_changes_total"] = float(g["root_changes_total"].mean())
            row["dead_nodes_mean"] = float(g["dead_nodes"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def grouped_bars(table: pd.DataFrame, metric: str, ylabel: str, out: Path, p95: str | None = None) -> None:
    methods = [m for m in METHOD_ORDER if m in set(table["method"])]
    loads = sorted(table["load"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=(metric == "pdr"))
    width = 0.8 / len(methods)
    for ax, c in zip(axes, CLASSES):
        sub = table[table["class"] == c]
        for i, m in enumerate(methods):
            x, y, e, q = [], [], [], []
            for j, load in enumerate(loads):
                r = sub[(sub["method"] == m) & (sub["load"] == load)]
                if r.empty:
                    continue
                x.append(j + (i - (len(methods) - 1) / 2) * width)
                y.append(r[metric].iloc[0])
                e.append(r.get(f"{metric}_std", pd.Series([0.0])).iloc[0])
                if p95:
                    q.append(r[p95].iloc[0])
            ax.bar(x, y, width, yerr=e, capsize=2, color=COLORS.get(m), label=m)
            if p95:
                ax.plot(x, q, "k.", markersize=5, label="p95" if (i == 0 and c == CLASSES[0]) else None)
        ax.set_xticks(range(len(loads)), [f"{v:g}" for v in loads])
        ax.set_xlabel("load multiplier")
        ax.set_title(c)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    if metric == "pdr":
        axes[0].set_ylim(0, 1.05)
    (axes[-1] if metric == "pdr" else axes[0]).legend(fontsize=8, loc="upper right" if metric == "pdr" else "upper left")
    fig.suptitle(f"{ylabel} by traffic class (mean over 5 seeds, error bar = std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    runs = pd.read_csv(TABLES / "runs.csv")
    base = pd.read_csv(TABLES / "m6_baselines.csv")
    table = base
    b3 = runs[runs["experiment"] == "b3_check"]
    if not b3.empty:
        b3_table = summarize(b3)
        b3_table.to_csv(TABLES / "b3_check.csv", index=False, float_format="%.4f")
        print(f"-> {TABLES / 'b3_check.csv'} ({len(b3)} runs)")
        table = pd.concat([base, b3_table], ignore_index=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    grouped_bars(table, "pdr", "PDR", FIGS / "m6_pdr_by_class.png")
    grouped_bars(table, "lat_mean_ms", "latency (ms): mean bar, p95 dot", FIGS / "m6_latency_by_class.png",
                 p95="lat_p95_ms")
    print(f"-> {FIGS / 'm6_pdr_by_class.png'}\n-> {FIGS / 'm6_latency_by_class.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

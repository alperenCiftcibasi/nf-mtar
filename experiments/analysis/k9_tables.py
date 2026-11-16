"""K9 ε / güven kapısı taraması özeti ve NF-MTAR karşılaştırma tablosu (CLAUDE.md §5.18, §5.20 K9; §5.19 istatistik).

1. results/tables/k9_eps.csv: B3, yük 0,5, seed 1–5; ε ∈ {0,05, 0,1, 0,2} ve ε = 0,05 + confidence_gate (conf_min 0,3).
   ε = 0,05 satırları E1 koşularından (B3, yük 0,5, seed 1–5), diğerleri k9_eps (runs.csv) ve k9_gate
   (runs_k9_gate.csv) koşularından. Sütunlar: ε, gate, root değişimi E/L/R/toplam, sınıf PDR'leri, Φ (seed ortalaması).
2. results/tables/e1_nfmtar_comparison.csv: E1_stats.csv'den NF-MTAR − B3 / B1 / B2; yük başına ağ geneli Φ, sınıf
   PDR'leri ve root değişim sayısı: ortalama fark [%95 GA], Wilcoxon p (Holm), d_z, n.

Kullanım: python experiments/analysis/k9_tables.py [k9|nfmtar]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
CLASSES = ("critical", "regular", "noncritical")


def k9() -> None:
    runs = pd.read_csv(TABLES / "runs.csv", low_memory=False)
    base = runs[(runs.experiment == "E1") & (runs.method == "B3") & (runs.load_multiplier == 0.5) & (runs.seed <= 5)]
    eps = runs[runs.experiment == "k9_eps"]
    gate = pd.read_csv(TABLES / "runs_k9_gate.csv", low_memory=False) if (TABLES / "runs_k9_gate.csv").exists() else None
    parts = [(0.05, False, base)]
    for e, g in eps.groupby("grid_controller.switching.epsilon"):
        parts.append((float(e), False, g))
    if gate is not None:
        parts.append((0.05, True, gate))
    rows = []
    for e, gated, g in parts:
        row = {"epsilon": e, "confidence_gate": gated, "conf_min": 0.3 if gated else None, "n_seeds": len(g)}
        for t in ("E", "L", "R"):
            row[f"root_changes_{t}"] = g[f"root_changes_{t}"].mean()
        row["root_changes_total"] = g["root_changes_total"].mean()
        for c in CLASSES:
            row[f"pdr_{c}"] = g[f"pdr_{c}"].mean()
        row["phi"] = g["phi"].mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "k9_eps.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def nfmtar() -> None:
    d = pd.read_csv(TABLES / "E1_stats.csv")
    p = d[d.kind == "pairwise"]
    rows = []
    for load in (0.25, 0.5):
        for other in ("B3", "B1", "B2"):
            for cls, met in [("all", "phi"), *[(c, "pdr") for c in CLASSES], ("all", "root_changes_total")]:
                r = p[(p.load_multiplier == load) & (p["class"] == cls) & (p.metric == met)
                      & (((p.method == other) & (p.method_b == "NF-MTAR")) | ((p.method == "NF-MTAR") & (p.method_b == other)))]
                if len(r) != 1:
                    continue
                r = r.iloc[0]
                s = 1 if r.method == "NF-MTAR" else -1
                lo, hi = sorted([s * r.ci_lo, s * r.ci_hi])
                rows.append({"load": load, "comparison": f"NF-MTAR-{other}", "class": cls, "metric": met,
                             "mean_diff": s * r.mean_diff, "ci_lo": lo, "ci_hi": hi, "p_w_holm": r.p_w_holm,
                             "d_z": s * r.d_z, "n": int(r.n)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "e1_nfmtar_comparison.csv", index=False, float_format="%.5g")
    for _, r in df.iterrows():
        print(f"| {r['load']:g} | {r.comparison} | {r['class']} | {r.metric} | {r.mean_diff:+.3f} | "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p_w_holm:.3g} | {r.d_z:+.2f} | {r.n} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("k9", "both"):
        k9()
    if which in ("nfmtar", "both"):
        nfmtar()

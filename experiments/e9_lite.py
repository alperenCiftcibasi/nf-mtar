"""E9-lite — özgün katkı K1'in küçük ölçekli hali (CLAUDE.md §5.16, §5.20): yorumlanabilirliği koruyan eğitim.

Veri: data/e9_lite.npz (experiments/build_training_data.py; B1, yük 0,5, seed 1–4 eğitim, 5 doğrulama, 6 test).
Koşullar: B3 (eğitimsiz uzman), anchored, zero_l2, none; λ = 1e-2; Adam lr 0,01, en çok 200 epoch, doğrulamada
erken durdurma (patience 20). Her koşul için test MSE, Spearman, top-1, pişmanlık + §5.20 yorumlanabilirlik ölçütleri.
Çıktı: results/tables/e9_lite.csv, results/figures/e9_lite_terms.png (x3 terim şekilleri, uzman vs her mod),
results/models/e9_lite_<mod>.json.

Kullanım: python experiments/e9_lite.py [--lam 1e-2] [--epochs 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.config import load_config  # noqa: E402
from nfmtar.engine.fuzzy import TERM_LABELS, FuzzyEngine, FuzzyParams, FuzzySpec, gaussian_mf  # noqa: E402
from nfmtar.engine.interpretability import interpretability_report  # noqa: E402
from nfmtar.engine.trainer import evaluate, train  # noqa: E402

DATA = PROJECT_ROOT / "data" / "e9_lite.npz"
TABLE = PROJECT_ROOT / "results" / "tables" / "e9_lite.csv"
FIG = PROJECT_ROOT / "results" / "figures" / "e9_lite_terms.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args(argv)
    cfg = load_config()
    spec = FuzzySpec.from_config(cfg)
    d = np.load(DATA)
    tr, va, te = (d["split"] == k for k in (0, 1, 2))
    X, y, grp = d["X"], d["y"], d["group"]
    params = {"B3 (expert, untrained)": FuzzyParams.expert(spec)}
    epochs = {"B3 (expert, untrained)": 0}
    for mode in ("anchored", "zero_l2", "none"):
        res = train(spec, X[tr], y[tr], X[va], y[va], reg_mode=mode, lam=args.lam, max_epochs=args.epochs)
        params[mode] = res.params
        epochs[mode] = res.epochs
        res.params.save_json(PROJECT_ROOT / "results" / "models" / f"e9_lite_{mode}.json")
        print(f"{mode}: best epoch {res.epochs}, val MSE {res.best_val_mse:.5f}", flush=True)
    rows = []
    for name, p in params.items():
        f = FuzzyEngine(spec, p).predict_f(X[te])
        rows.append({"condition": name, "lambda": 0.0 if name.startswith("B3") or name == "none" else args.lam,
                     "best_epoch": epochs[name], **{f"test_{k}": v for k, v in evaluate(f, y[te], grp[te]).items()},
                     **interpretability_report(spec, p)})
    df = pd.DataFrame(rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE, index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"train/val/test rows {tr.sum()}/{va.sum()}/{te.sum()}; test groups {len(np.unique(grp[te]))}")

    # x3 (indeks 2) terim şekilleri
    xs = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3), sharey=True)
    for ax, (name, p) in zip(axes, params.items()):
        for j, label in enumerate(TERM_LABELS[spec.n_terms[2]]):
            ax.plot(xs, gaussian_mf(xs, p.centers[2][j], p.sigmas[2][j]), label=label)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("x3 depletion")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("membership")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"x3 term shapes: expert vs trained (lambda = {args.lam:g})", fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"-> {TABLE}\n-> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

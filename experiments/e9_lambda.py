"""E9 λ taraması (CLAUDE.md §5.20, küçük ölçek): data/e9_lite.npz ile anchored ve zero_l2 için λ ∈ {1e-3, 1e-2, 1e-1}
ve none; her koşul 2 tekrar (farklı mini yığın karıştırma seed'i), en çok 300 epoch, doğrulamada erken durdurma.

Çıktı: results/tables/e9_lambda.csv (e9_lite sütunları, tekrar ortalaması + std'ler için *_std),
results/figures/e9_pareto.png (x: monotonluk ihlali + sıra bozulması, y: test pişmanlığı; her koşul bir nokta; B3 dahil).

Kullanım: python experiments/e9_lambda.py [--reps 2] [--epochs 300] [--batch 4096] [--workers 4]
          [--data data/e9_lite.npz] [--lams 1e-3 1e-2 1e-1] [--modes anchored zero_l2 none] [--tag ""]
          [--patience 20] [--max-train-rows N]
--tag verilirse çıktılar e9_lambda_<tag>.csv (+ _reps.csv), e9_pareto_<tag>.png ve modeller
results/models/e9_<tag>_<mod>_<λ>_r<tekrar>.json.
--max-train-rows: eğitim kümesinden grup bazında alt örnek (büyük veride süre sınırı; doğrulama/test tam kalır).
§5.21 zengin veriyle E9-lite tekrarı: --data data/e9_rich.npz --lams 1e-2 --tag rich
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = PROJECT_ROOT / "data" / "e9_lite.npz"
TABLE = PROJECT_ROOT / "results" / "tables" / "e9_lambda.csv"
FIG = PROJECT_ROOT / "results" / "figures" / "e9_pareto.png"


def run_one(task: tuple) -> dict:
    import torch

    torch.set_num_threads(1)
    from nfmtar.config import load_config
    from nfmtar.engine.fuzzy import FuzzyEngine, FuzzyParams, FuzzySpec
    from nfmtar.engine.interpretability import interpretability_report
    from nfmtar.engine.trainer import evaluate, train

    mode, lam, rep, epochs, batch, data, patience, max_train, tag = task
    spec = FuzzySpec.from_config(load_config())
    d = np.load(data)
    tr, va, te = (d["split"] == k for k in (0, 1, 2))
    X, y, grp = d["X"], d["y"], d["group"]
    if max_train and tr.sum() > max_train:                  # grup bazında alt örnek (sabit seed: her koşulda aynı)
        groups = np.unique(grp[tr])
        keep = np.random.default_rng(0).choice(groups, size=int(len(groups) * max_train / tr.sum()), replace=False)
        tr = tr & np.isin(grp, keep)
    if mode == "B3":
        p, ep = FuzzyParams.expert(spec), 0
    else:
        res = train(spec, X[tr], y[tr], X[va], y[va], reg_mode=mode, lam=lam, max_epochs=epochs, seed=rep,
                    batch_size=batch, patience=patience)
        p, ep = res.params, res.epochs
        if tag:
            p.save_json(PROJECT_ROOT / "results" / "models" / f"e9_{tag}_{mode}_{lam:g}_r{rep}.json")
    f = FuzzyEngine(spec, p).predict_f(X[te])
    return {"condition": mode, "lambda": lam, "rep": rep, "best_epoch": ep, "n_train": int(tr.sum()),
            **{f"test_{k}": v for k, v in evaluate(f, y[te], grp[te]).items()}, **interpretability_report(spec, p)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--lams", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--modes", nargs="+", default=["anchored", "zero_l2", "none"])
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    data = args.data if Path(args.data).is_absolute() else str(PROJECT_ROOT / args.data)
    extra = (data, args.patience, args.max_train_rows, args.tag)
    tasks = [("B3", 0.0, 0, 0, 0, *extra)]
    for mode in args.modes:
        for lam in ([0.0] if mode == "none" else args.lams):
            tasks += [(mode, lam, r, args.epochs, args.batch, *extra) for r in range(args.reps)]
    table_path = TABLE.with_name(f"e9_lambda_{args.tag}.csv") if args.tag else TABLE
    fig_path = FIG.with_name(f"e9_pareto_{args.tag}.png") if args.tag else FIG
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, tasks))
    raw = pd.DataFrame(rows)
    num = [c for c in raw.columns if c not in ("condition", "lambda", "rep")]
    g = raw.groupby(["condition", "lambda"], sort=False)
    table = g[num].mean().join(g[["test_regret", "test_mse"]].std().add_suffix("_std")).reset_index()
    table.insert(2, "n_reps", g.size().values)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False, float_format="%.5f")
    raw.to_csv(table_path.with_name(table_path.stem + "_reps.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B3": "#9e9e9e", "anchored": "#4c72b0", "zero_l2": "#c44e52", "none": "#55a868"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in table.iterrows():
        x = r["monotonicity_violation"] + r["order_violations"]
        ax.scatter(x, r["test_regret"], color=colors[r["condition"]], s=40)
        label = r["condition"] if r["condition"] in ("B3", "none") else f"{r['condition']} {r['lambda']:g}"
        ax.annotate(label, (x, r["test_regret"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("interpretability violation (monotonicity violation + term-order violations)")
    ax.set_ylabel("test regret (oracle best - oracle chosen)")
    ax.set_title(f"E9{'-' + args.tag if args.tag else '-lite'}: regret vs interpretability (mean of reps)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"-> {table_path}\n-> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

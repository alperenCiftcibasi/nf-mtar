"""Tam E9 — özgün katkı K1 (CLAUDE.md §5.20, §5.22): yorumlanabilirliği koruyan eğitim, tam ızgara.

Veri: data/e9_rich.npz (experiments/build_training_data.py; seed 101–120, 14 eğitim / 3 doğrulama / 3 test, seed bazlı;
x2 = radyo meşguliyeti). Eğitim 200k satırlık sabit, grup bazlı alt örnekle (bütün koşullarda aynı; §5.22); doğrulama
ve test tam küme, parça parça değerlendirilir (bellek sınırı).

Koşullar (§5.22): reg_mode ∈ {none, zero_l2, anchored, anchored+cov}; λ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} (none tek
koşul, λ = 0); her koşul 3 tekrar (mini yığın karıştırma seed'i 0, 1, 2); ayrıca B3 (eğitimsiz uzman).
Optimizasyon e9_lambda.py ile aynı: Adam lr 0,01, mini yığın 4096, en çok 150 epoch, doğrulama MSE'sinde sabır 20.

Ölçütler (§5.22): ANA pişmanlık ve Spearman; yardımcı top-1 ve yakın-en-iyi oranı (seçilen ≥ en iyi − 0,05), MSE;
§5.20 yorumlanabilirlik ölçütleri. Doğrulama ve test için ayrı ayrı.
Seçim kuralı (önceden sabit): her mod için λ = ortalama doğrulama pişmanlığı en küçük olan. NF-MTAR ana modeli =
anchored+cov'un seçilen λ'sındaki, doğrulama pişmanlığı en küçük tekrar -> results/models/nfmtar_main.json.
Seçime test sonuçları girmez; anchored (cov'suz) daha iyi çıksa da ana model anchored+cov'dan alınır.
K1 ana ölçümü: anchored+cov − anchored farkı (pişmanlık ve kapsama), seçilen λ'larda ve aynı λ'da eşleştirilmiş.

Çıktılar:
- results/tables/e9_full_grid.csv      bütün ızgara, tekrar ortalaması ± std (val_* ve test_*)
- results/tables/e9_full_reps.csv      her eğitim tekrarı ayrı
- results/tables/e9_full_selected.csv  seçilen λ'ların test sonuçları + B3
- results/tables/e9_full_k1_cov_vs_anchored.csv  anchored+cov ve anchored yan yana, fark (cov − anchored) ± eşleştirilmiş std
- results/figures/e9_pareto.png        x: yorumlanabilirlik ihlali (monotonluk ihlali + sıra bozulması), y: test pişmanlığı
- results/figures/e9_terms_<girdi>.png uzman vs seçilen modların terim şekilleri (girdi başına)
- results/models/e9_full/<mod>_<λ>_r<tekrar>.json, results/models/nfmtar_main.json

Kullanım: python experiments/e9_full.py [--workers 6] [--reps 3] [--epochs 150] [--patience 20]
          python experiments/e9_full.py --from-csv   (eğitmeden, kayıtlı e9_full_reps.csv'den tablolar ve şekiller)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = PROJECT_ROOT / "data" / "e9_rich.npz"
TABLES = PROJECT_ROOT / "results" / "tables"
FIGS = PROJECT_ROOT / "results" / "figures"
MODELS = PROJECT_ROOT / "results" / "models"
MODES = ("none", "zero_l2", "anchored", "anchored+cov")
LAMBDAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EVAL_CHUNK = 5_000


def model_path(mode: str, lam: float, rep: int) -> Path:
    return MODELS / "e9_full" / f"{mode.replace('+', '_')}_{lam:g}_r{rep}.json"


def train_subsample(grp: np.ndarray, tr: np.ndarray, max_rows: int) -> np.ndarray:
    """Eğitim kümesinden grup bazında sabit alt örnek (rng 0; e9_lambda.py ile aynı kural)."""
    if not max_rows or tr.sum() <= max_rows:
        return tr
    groups = np.unique(grp[tr])
    keep = np.random.default_rng(0).choice(groups, size=int(len(groups) * max_rows / tr.sum()), replace=False)
    return tr & np.isin(grp, keep)


def run_one(task: tuple) -> dict:
    import torch

    torch.set_num_threads(1)
    from nfmtar.config import load_config
    from nfmtar.engine.fuzzy import FuzzyEngine, FuzzyParams, FuzzySpec
    from nfmtar.engine.interpretability import interpretability_report
    from nfmtar.engine.trainer import evaluate, train

    mode, lam, rep, epochs, patience, batch, max_train, data = task
    cfg = load_config()
    spec = FuzzySpec.from_config(cfg)
    margin = float(cfg.training["near_best_margin"])                 # §5.22 yakın-en-iyi
    d = np.load(data)
    X, y, grp, split = d["X"], d["y"], d["group"], d["split"]
    tr, va, te = (split == k for k in (0, 1, 2))
    tr = train_subsample(grp, tr, max_train)
    if mode == "B3":
        p, ep, path = FuzzyParams.expert(spec), 0, ""
    else:
        cov = cfg.training["loss"]["coverage"]
        res = train(spec, X[tr], y[tr], X[va], y[va], reg_mode=mode, lam=lam, max_epochs=epochs, patience=patience,
                    seed=rep, batch_size=batch, eval_chunk=EVAL_CHUNK, gamma_cov=float(cov["gamma_cov"]),
                    cov_eps=float(cov["epsilon"]), cov_grid=int(cov["grid_points"]))
        p, ep = res.params, res.epochs
        path = model_path(mode, lam, rep)
        path.parent.mkdir(parents=True, exist_ok=True)
        p.save_json(path)
    eng = FuzzyEngine(spec, p)
    row = {"condition": mode, "lambda": lam, "rep": rep, "best_epoch": ep, "n_train": int(tr.sum()),
           "model": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") if path else ""}
    for name, m in (("val", va), ("test", te)):
        f = eng.predict_f(X[m], EVAL_CHUNK)
        row.update({f"{name}_{k}": v for k, v in evaluate(f, y[m], grp[m], margin).items()})
    row.update(interpretability_report(spec, p))
    print(f"{mode} lam={lam:g} rep={rep}: epoch {ep}, val regret {row['val_regret']:.4f}, "
          f"test regret {row['test_regret']:.4f}", flush=True)
    return row


PRIMARY = ["test_regret", "test_spearman"]
SECONDARY = ["test_near_best", "test_top1", "test_mse"]
INTERP = ["coverage_min", "order_violations", "monotonicity_violation", "distinguishability_max", "dev_c", "dev_sigma",
          "dev_y"]


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in raw.columns if c not in ("condition", "lambda", "rep", "model")]
    g = raw.groupby(["condition", "lambda"], sort=False)
    out = g[num].mean().join(g[num].std().add_suffix("_std")).reset_index()
    out.insert(2, "n_reps", g.size().values)
    return out


def select(grid: pd.DataFrame) -> pd.DataFrame:
    """Önceden sabit kural: her mod için ortalama doğrulama pişmanlığı en küçük λ (§5.22); B3 olduğu gibi."""
    rows = [grid[grid["condition"] == "B3"]]
    for mode in MODES:
        sub = grid[grid["condition"] == mode]
        rows.append(sub.loc[[sub["val_regret"].idxmin()]])
    sel = pd.concat(rows, ignore_index=True)
    cols = ["condition", "lambda", "n_reps"]
    for c in [*PRIMARY, *SECONDARY, "val_regret", "val_spearman"]:
        cols += [c, f"{c}_std"]
    cols += [*INTERP, "best_epoch"]
    return sel[cols]


K1_METRICS = ["test_regret", "val_regret", "test_spearman", "test_near_best", "coverage_min", "monotonicity_violation"]


def k1_comparison(raw: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    """K1 ana ölçümü (§5.20, §5.22): anchored+cov ile anchored yan yana, fark = anchored+cov − anchored.

    İlk satır: her mod KENDİ seçilen λ'sında (doğrulama pişmanlığıyla seçilmiş; test sonuçları seçime girmez).
    Sonraki satırlar: aynı λ'da eşleştirilmiş karşılaştırma (λ ızgarası boyunca). Tekrarlar tekrar indeksiyle
    eşleştirilir (aynı başlangıç seed'i, aynı veri); d_<ölçüt>_std eşleştirilmiş farkların std'si.
    """
    def paired(lam_cov: float, lam_anc: float, label: str) -> dict:
        a = raw[(raw["condition"] == "anchored+cov") & np.isclose(raw["lambda"], lam_cov)].set_index("rep")
        b = raw[(raw["condition"] == "anchored") & np.isclose(raw["lambda"], lam_anc)].set_index("rep")
        reps = a.index.intersection(b.index)
        row = {"comparison": label, "lambda_cov": lam_cov, "lambda_anchored": lam_anc, "n_pairs": len(reps)}
        for m in K1_METRICS:
            d = a.loc[reps, m] - b.loc[reps, m]
            row.update({f"{m}_cov": a.loc[reps, m].mean(), f"{m}_anchored": b.loc[reps, m].mean(),
                        f"d_{m}": d.mean(), f"d_{m}_std": d.std()})
        return row

    lam = {m: float(sel.loc[sel["condition"] == m, "lambda"].iloc[0]) for m in ("anchored+cov", "anchored")}
    rows = [paired(lam["anchored+cov"], lam["anchored"], "selected_lambda_each")]
    rows += [paired(l, l, "same_lambda") for l in LAMBDAS]
    return pd.DataFrame(rows)


def plots(grid: pd.DataFrame, sel: pd.DataFrame, best_models: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nfmtar.config import load_config
    from nfmtar.engine.base import FEATURE_LABELS, FEATURE_NAMES
    from nfmtar.engine.fuzzy import TERM_LABELS, FuzzyParams, FuzzySpec, gaussian_mf

    colors = {"B3": "#9e9e9e", "none": "#55a868", "zero_l2": "#c44e52", "anchored": "#4c72b0",
              "anchored+cov": "#8172b2"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in grid.iterrows():
        x = r["monotonicity_violation"] + r["order_violations"]
        chosen = ((sel["condition"] == r["condition"]) & np.isclose(sel["lambda"], r["lambda"])).any()
        ax.scatter(x, r["test_regret"], color=colors[r["condition"]], s=70 if chosen else 22,
                   edgecolor="black" if chosen else "none", zorder=3 if chosen else 2)
        if chosen:
            label = r["condition"] if r["condition"] in ("B3", "none") else f"{r['condition']} λ={r['lambda']:g}"
            ax.annotate(label, (x, r["test_regret"]), fontsize=7, xytext=(5, 4), textcoords="offset points")
    for m, c in colors.items():
        ax.scatter([], [], color=c, label=m)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("interpretability violation (monotonicity violation + term-order violations)")
    ax.set_ylabel("test regret (oracle best − oracle chosen)")
    ax.set_title("E9: regret vs interpretability (mean of 3 reps; outlined = selected λ)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "e9_pareto.png", dpi=150)
    plt.close(fig)

    spec = FuzzySpec.from_config(load_config())
    params = {"B3 (expert)": FuzzyParams.expert(spec)}
    for mode, path in best_models.items():
        params[f"{mode}"] = FuzzyParams.load_json(path)
    xs = np.linspace(0, 1, 400)
    for i, name in enumerate(FEATURE_NAMES):
        fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 2.8), sharey=True)
        for ax, (title, p) in zip(axes, params.items()):
            for j, label in enumerate(TERM_LABELS[spec.n_terms[i]]):
                ax.plot(xs, gaussian_mf(xs, p.centers[i][j], p.sigmas[i][j]), label=label)
            ax.axhline(0.5, color="black", lw=0.6, ls=":")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f"{name} ({FEATURE_LABELS[name]})", fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("membership")
        axes[-1].legend(fontsize=6)
        fig.suptitle(f"{name}: expert vs selected trained models (dotted: coverage threshold 0.5)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / f"e9_terms_{name}.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--from-csv", action="store_true",
                        help="eğitmeden, kayıtlı e9_full_reps.csv'den tabloları, seçimi ve şekilleri yeniden üret")
    args = parser.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        raw = pd.read_csv(TABLES / "e9_full_reps.csv")
    else:
        common = (args.epochs, args.patience, args.batch, args.max_train_rows, args.data)
        tasks = [(m, lam, r, *common) for m in ("anchored+cov", "anchored", "zero_l2") for lam in LAMBDAS
                 for r in range(args.reps)]
        tasks += [("none", 0.0, r, *common) for r in range(args.reps)] + [("B3", 0.0, 0, *common)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            raw = pd.DataFrame(list(pool.map(run_one, tasks)))
        order = {m: i for i, m in enumerate(("B3", *MODES))}
        raw = raw.sort_values(["condition", "lambda", "rep"],
                              key=lambda s: s.map(order) if s.name == "condition" else s)
        raw.to_csv(TABLES / "e9_full_reps.csv", index=False, float_format="%.5f")
    grid = aggregate(raw)
    grid.to_csv(TABLES / "e9_full_grid.csv", index=False, float_format="%.5f")
    sel = select(grid)
    sel.to_csv(TABLES / "e9_full_selected.csv", index=False, float_format="%.5f")
    k1 = k1_comparison(raw, sel)
    k1.to_csv(TABLES / "e9_full_k1_cov_vs_anchored.csv", index=False, float_format="%.5f")

    # NF-MTAR ana modeli (§5.22, önceden sabit kural): YALNIZCA anchored+cov içinden; λ ve tekrar doğrulama
    # pişmanlığıyla seçilir, test sütunları seçime girmez. anchored daha iyi çıksa da kural değişmez.
    best_models: dict[str, Path] = {}
    for _, r in sel[sel["condition"] != "B3"].iterrows():
        reps = raw[(raw["condition"] == r["condition"]) & np.isclose(raw["lambda"], r["lambda"])]
        best_models[r["condition"]] = PROJECT_ROOT / reps.loc[reps["val_regret"].idxmin(), "model"]
    main_model = MODELS / "nfmtar_main.json"
    shutil.copyfile(best_models["anchored+cov"], main_model)
    plots(grid, sel, best_models)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(grid[["condition", "lambda", "val_regret", "test_regret", "test_spearman", "test_near_best",
                "coverage_min", "monotonicity_violation"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(sel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(k1[["comparison", "lambda_cov", "lambda_anchored", "test_regret_cov", "test_regret_anchored", "d_test_regret",
              "coverage_min_cov", "coverage_min_anchored", "d_coverage_min"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"NF-MTAR main model: {best_models['anchored+cov'].name} -> {main_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

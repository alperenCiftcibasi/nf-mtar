"""Kâhin duyarlılığı (CLAUDE.md §5.22): sabit ölçekli ikinci etiket, yalnızca seçilen koşullar için, çevrimdışı.

Asıl kâhin (§5.16) DEĞİŞMEZ. Duyarlılık etiketi grup içi min-max kullanmaz:
    y_fixed = w_d · teslim + w_l · (1 − clip(gecikme_s / π3_t, 0, 1)) + w_r · clip(dayanıklılık_s / H, 0, 1)
w = config oracle.weights (0,4 / 0,3 / 0,3), π3_t topolojinin maks. gecikmesi (§5.7: E 25, L 100, R 500 ms),
H = oracle.fixed_scale.resilience_horizon_s (1800 s: veri koşularının süresi). Ham bileşenler
data/e9_rich.npz'deki `components_raw` sütunlarından gelir (yeniden simülasyon yok).

Koşullar: B3 (uzman); e9_full'da seçilen anchored+cov ve none (λ e9_full_selected.csv'den), sabit ölçekli etiketle
yeniden eğitilir (3 tekrar; aynı 200k alt örnek ve optimizasyon). Ayrıca asıl etiketle eğitilmiş (e9_full) aynı
koşulların modelleri sabit ölçekli etikette değerlendirilir (etiket değişince sıralamanın taşınıp taşınmadığı).
Test seed'lerinde her iki etikete göre ölçütler. Sabit ölçekli etiket min-max'lanmadığı için pişmanlık ölçeği farklıdır;
karşılaştırma aynı etiket içinde, B3'e göre (regret_minus_B3) yapılır. Ayrıca iki etiketin uyumu: grup içi
Spearman ortalaması ve en iyi adayın aynı olduğu grupların oranı.

Çıktı: results/tables/e9_oracle_sensitivity.csv (+ _reps.csv)
Kullanım: python experiments/e9_oracle_sensitivity.py [--workers 6] [--reps 3]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from e9_full import EVAL_CHUNK, MODELS, TABLES, train_subsample  # noqa: E402

DATA = PROJECT_ROOT / "data" / "e9_rich.npz"
TOPO = ("E", "L", "R")


def fixed_scale_label(d, cfg) -> np.ndarray:
    """§5.22 sabit ölçekli kâhin: grup içi min-max yok; gecikme/π3, dayanıklılık/ufuk, [0,1]'e kırpılmış."""
    w = cfg.oracle["weights"]
    raw = d["components_raw"]
    pi3 = np.array([cfg.policy.for_topology(t).max_delay_ms / 1000.0 for t in TOPO])[d["topology"]]
    horizon = float(cfg.oracle["fixed_scale"]["resilience_horizon_s"])
    return (w["delivery"] * raw[:, 0] + w["latency"] * (1 - np.clip(raw[:, 1] / pi3, 0, 1))
            + w["resilience"] * np.clip(raw[:, 2] / horizon, 0, 1))


def label_agreement(y_a: np.ndarray, y_b: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    bounds = [*starts.tolist(), len(order)]
    rho, same = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ya, yb = y_a[order][a:b], y_b[order][a:b]
        if len(ya) < 2:
            continue
        if np.ptp(ya) > 0 and np.ptp(yb) > 0:
            rho.append(spearmanr(ya, yb).statistic)
        same.append(float(np.argmax(ya) == np.argmax(yb)))
    return {"label_spearman": float(np.mean(rho)), "label_same_best": float(np.mean(same))}


def run_one(task: tuple) -> dict:
    import torch

    torch.set_num_threads(1)
    from nfmtar.config import load_config
    from nfmtar.engine.fuzzy import FuzzyEngine, FuzzyParams, FuzzySpec
    from nfmtar.engine.interpretability import interpretability_report
    from nfmtar.engine.trainer import evaluate, train

    kind, mode, lam, rep, epochs, patience, batch, max_train, src_model = task
    cfg = load_config()
    spec = FuzzySpec.from_config(cfg)
    d = np.load(DATA)
    X, grp, split = d["X"], d["group"], d["split"]
    y_orig, y_fix = d["y"], fixed_scale_label(d, cfg)
    tr, va, te = (split == k for k in (0, 1, 2))
    tr = train_subsample(grp, tr, max_train)
    if kind == "expert":
        p, ep = FuzzyParams.expert(spec), 0
    elif kind == "trained_on_original":
        p, ep = FuzzyParams.load_json(PROJECT_ROOT / src_model), -1
    else:                                                         # trained_on_fixed_scale
        cov = cfg.training["loss"]["coverage"]
        res = train(spec, X[tr], y_fix[tr], X[va], y_fix[va], reg_mode=mode, lam=lam, max_epochs=epochs,
                    patience=patience, seed=rep, batch_size=batch, eval_chunk=EVAL_CHUNK,
                    gamma_cov=float(cov["gamma_cov"]), cov_eps=float(cov["epsilon"]), cov_grid=int(cov["grid_points"]))
        p, ep = res.params, res.epochs
        out = MODELS / "e9_sensitivity" / f"{mode.replace('+', '_')}_{lam:g}_r{rep}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        p.save_json(out)
    f = FuzzyEngine(spec, p).predict_f(X[te], EVAL_CHUNK)
    row = {"model": kind, "condition": mode, "lambda": lam, "rep": rep, "best_epoch": ep}
    margin = float(cfg.training["near_best_margin"])
    row.update({f"fixed_{k}": v for k, v in evaluate(f, y_fix[te], grp[te], margin).items()})
    row.update({f"orig_{k}": v for k, v in evaluate(f, y_orig[te], grp[te], margin).items()})
    rep_i = interpretability_report(spec, p)
    row.update({k: rep_i[k] for k in ("coverage_min", "order_violations", "monotonicity_violation")})
    print(f"{kind} {mode} lam={lam:g} rep={rep}: fixed regret {row['fixed_regret']:.4f}", flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    args = parser.parse_args(argv)
    from nfmtar.config import load_config

    cfg = load_config()
    sel = pd.read_csv(TABLES / "e9_full_selected.csv")
    reps = pd.read_csv(TABLES / "e9_full_reps.csv")
    common = (args.epochs, args.patience, args.batch, args.max_train_rows)
    tasks = [("expert", "B3", 0.0, 0, *common, "")]
    for mode in ("anchored+cov", "none"):
        lam = float(sel.loc[sel["condition"] == mode, "lambda"].iloc[0])
        tasks += [("trained_on_fixed_scale", mode, lam, r, *common, "") for r in range(args.reps)]
        src = reps[(reps["condition"] == mode) & np.isclose(reps["lambda"], lam)]
        tasks += [("trained_on_original", mode, lam, int(r.rep), *common, r.model) for r in src.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = pd.DataFrame(list(pool.map(run_one, tasks)))
    raw.to_csv(TABLES / "e9_oracle_sensitivity_reps.csv", index=False, float_format="%.5f")
    num = [c for c in raw.columns if c not in ("model", "condition", "lambda", "rep")]
    g = raw.groupby(["model", "condition", "lambda"], sort=False)
    out = g[num].mean().join(g[["fixed_regret", "fixed_spearman", "orig_regret"]].std().add_suffix("_std")).reset_index()
    out.insert(3, "n_reps", g.size().values)
    for lab in ("fixed", "orig"):
        b3 = float(out.loc[out["condition"] == "B3", f"{lab}_regret"].iloc[0])
        out[f"{lab}_regret_minus_B3"] = out[f"{lab}_regret"] - b3
    d = np.load(DATA)
    te = d["split"] == 2
    agree = label_agreement(d["y"][te], fixed_scale_label(d, cfg)[te], d["group"][te])
    for k, v in agree.items():
        out[k] = v
    out.to_csv(TABLES / "e9_oracle_sensitivity.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"label agreement on test groups: {agree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

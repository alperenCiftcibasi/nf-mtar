"""Top-1 = 0 teşhisi (CLAUDE.md §5.21): kâhin değiştirilmez, yalnızca ölçülür ve raporlanır.

Veri: data/e9_rich.npz (experiments/build_training_data.py; `components` = kâhinin ağırlıklı bileşenleri).
Değerlendirme doğrulama seed'lerinde (split = 1); test seed'lerine dokunulmaz.

(a) Kâhin bileşenleri: her grupta (aynı ağ durumu × topoloji) adaylar arası varyans; bileşen payı = var_k / Σ var
    (grup ortalaması), kovaryans payı = (var(ham) − Σ var_k) / var(ham); bileşenin argmax'ı ile kâhinin argmax'ının
    aynı olduğu grupların oranı; bileşen ile kâhin skoru arasında grup içi Pearson korelasyonu. Topoloji başına ve toplam.
(b) Üst sınır: eğitim seed'lerinde girdilerden kâhin skoruna HistGradientBoosting ve RandomForest. "pointwise" yalnızca
    x1–x5 (bulanık sistemle aynı bilgi: her aday tek başına puanlanır) -> bulanık sistemin ulaşabileceği üst sınırın
    tahmini; "context" ayrıca grup içi göreli girdiler (x − grup ortalaması, x − grup en büyüğü, grup içi yüzdelik sıra)
    ve topoloji -> aynı girdilerle grup bağlamı kullanan bir puanlayıcının sınırı. Doğrulamada R², MSE, Spearman, top-1,
    pişmanlık; karşılaştırma için B3 (uzman), eğitilmiş anchored (config engine.trained_model) ve rastgele seçim.
(c) Kâhinin en iyi adayı ile B3'ün seçtiği adayın özellik vektörleri: her topolojiden örnek gruplar (B3'ün yanıldığı)
    ve bütün doğrulama gruplarında ortalama fark (en iyi − B3 seçimi).

Çıktı: results/tables/top1_components.csv, top1_bounds.csv, top1_examples.csv, top1_feature_gap.csv
Kullanım: python experiments/analysis/top1_diagnosis.py [--data data/e9_rich.npz] [--max-train 600000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.config import load_config  # noqa: E402
from nfmtar.engine.base import FEATURE_NAMES  # noqa: E402
from nfmtar.engine.fuzzy import FuzzyEngine, FuzzyParams, FuzzySpec  # noqa: E402
from nfmtar.engine.trainer import evaluate  # noqa: E402

TABLES = PROJECT_ROOT / "results" / "tables"
TOPO = {0: "E", 1: "L", 2: "R"}
COMP = ("delivery", "latency", "resilience")


def group_frame(d, mask) -> pd.DataFrame:
    df = pd.DataFrame(d["X"][mask], columns=list(FEATURE_NAMES))
    df["y"] = d["y"][mask]
    df["group"] = d["group"][mask]
    df["topology"] = d["topology"][mask]
    df["node"] = d["node"][mask]
    if "components" in d.files:
        for i, c in enumerate(COMP):
            df[c] = d["components"][mask][:, i]
    return df


# --------------------------------------------------------------------------- (a)

def component_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raw"] = df[list(COMP)].sum(axis=1)
    g = df.groupby("group")
    var = g[[*COMP, "raw"]].var(ddof=0)
    topo = g["topology"].first()
    var = var[var["raw"] > 0]
    sum_var = var[list(COMP)].sum(axis=1)
    res = pd.DataFrame({f"share_{c}": var[c] / sum_var for c in COMP})
    res["cov_share"] = (var["raw"] - sum_var) / var["raw"]
    # argmax uyumu ve grup içi korelasyon
    idx_best = g["raw"].idxmax()
    for c in COMP:
        res[f"argmax_agree_{c}"] = (g[c].idxmax() == idx_best).reindex(res.index).astype(float)
        dc = df[c] - g[c].transform("mean")
        dr = df["raw"] - g["raw"].transform("mean")
        cov = (dc * dr).groupby(df["group"]).mean()
        sd = np.sqrt((dc ** 2).groupby(df["group"]).mean() * (dr ** 2).groupby(df["group"]).mean())
        res[f"corr_{c}"] = (cov / sd.replace(0, np.nan)).reindex(res.index)
    res["topology"] = topo.reindex(res.index).map(TOPO)
    out = res.groupby("topology").mean(numeric_only=True)
    out.loc["all"] = res.mean(numeric_only=True)
    out.insert(0, "n_groups", [*res.groupby("topology").size().tolist(), len(res)])
    return out.reset_index()


# --------------------------------------------------------------------------- (b)

def context_features(df: pd.DataFrame) -> np.ndarray:
    g = df.groupby("group")
    cols = [df[list(FEATURE_NAMES)].to_numpy()]
    for n in FEATURE_NAMES:
        cols.append((df[n] - g[n].transform("mean")).to_numpy()[:, None])
        cols.append((df[n] - g[n].transform("max")).to_numpy()[:, None])
        cols.append(g[n].rank(pct=True).to_numpy()[:, None])
    cols.append(df["topology"].to_numpy()[:, None].astype(float))
    return np.hstack(cols)


def r2(f, y) -> float:
    return float(1 - np.mean((y - f) ** 2) / np.var(y))


def random_choice(df: pd.DataFrame) -> dict[str, float]:
    g = df.groupby("group")["y"]
    n = g.size()
    n = n[n >= 2]
    return {"top1": float((1 / n).mean()), "regret": float((g.max() - g.mean())[n.index].mean())}


def bounds(tr: pd.DataFrame, va: pd.DataFrame, cfg, max_train: int, seed: int = 0) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    rng = np.random.default_rng(seed)
    if len(tr) > max_train:                        # grup bazında alt örnek (gruplar bütün kalsın; context için)
        groups = tr["group"].unique()
        keep = rng.choice(groups, size=int(len(groups) * max_train / len(tr)), replace=False)
        tr = tr[tr["group"].isin(keep)]
    rows = []
    yv, gv = va["y"].to_numpy(), va["group"].to_numpy()

    def add(name, features, f):
        rows.append({"model": name, "features": features, "r2": r2(f, yv), **evaluate(f, yv, gv)})

    spec = FuzzySpec.from_config(cfg)
    Xv = va[list(FEATURE_NAMES)].to_numpy()
    add("B3 expert fuzzy", "pointwise", FuzzyEngine(spec, FuzzyParams.expert(spec)).predict_f(Xv))
    trained = FuzzyParams.load_json(PROJECT_ROOT / cfg.engine["trained_model"])
    add(f"trained fuzzy ({Path(cfg.engine['trained_model']).stem})", "pointwise",
        FuzzyEngine(spec, trained).predict_f(Xv))
    for feats, Xt_, Xv_ in (("pointwise", tr[list(FEATURE_NAMES)].to_numpy(), Xv),
                            ("context", context_features(tr), context_features(va))):
        hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, random_state=seed)
        hgb.fit(Xt_, tr["y"].to_numpy())
        add("HistGradientBoosting", feats, hgb.predict(Xv_))
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_samples=min(1.0, 200_000 / len(tr)),
                                   n_jobs=-1, random_state=seed)
        rf.fit(Xt_, tr["y"].to_numpy())
        add("RandomForest", feats, rf.predict(Xv_))
        print(f"  {feats}: fitted on {len(tr)} rows", flush=True)
    rows.append({"model": "random choice", "features": "-", **random_choice(va)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (c)

def feature_gap(va: pd.DataFrame, cfg, n_examples: int = 2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = FuzzySpec.from_config(cfg)
    va = va.copy()
    va["f_B3"] = FuzzyEngine(spec, FuzzyParams.expert(spec)).predict_f(va[list(FEATURE_NAMES)].to_numpy())
    g = va.groupby("group")
    best = va.loc[g["y"].idxmax()].set_index("group")
    chosen = va.loc[g["f_B3"].idxmax()].set_index("group")
    wrong = best.index[best["node"] != chosen.loc[best.index, "node"]]
    diff = best.loc[wrong, list(FEATURE_NAMES)] - chosen.loc[wrong, list(FEATURE_NAMES)]
    diff["topology"] = best.loc[wrong, "topology"].map(TOPO)
    gap = diff.groupby("topology").mean()
    gap.loc["all"] = diff.mean(numeric_only=True)
    pos = (diff[list(FEATURE_NAMES)] > 0).groupby(diff["topology"]).mean().add_prefix("frac_best_higher_")
    pos.loc["all"] = (diff[list(FEATURE_NAMES)] > 0).mean().add_prefix("frac_best_higher_")
    gap = gap.add_prefix("mean_best_minus_B3_").join(pos)
    n_wrong = diff.groupby("topology").size()
    n_wrong["all"] = len(diff)
    n_all = va.groupby(va["topology"].map(TOPO))["group"].nunique()
    n_all["all"] = va["group"].nunique()
    gap.insert(0, "n_groups_B3_wrong", n_wrong.reindex(gap.index).to_numpy())
    gap.insert(1, "n_groups", n_all.reindex(gap.index).to_numpy())
    rng = np.random.default_rng(seed)
    ex = []
    cols = [*FEATURE_NAMES, "y", "f_B3", *(c for c in COMP if c in va)]
    for t in sorted(diff["topology"].unique()):
        gs = rng.choice(diff.index[diff["topology"] == t], size=min(n_examples, (diff["topology"] == t).sum()),
                        replace=False)
        for gid in gs:
            for role, frame in (("oracle_best", best), ("B3_chosen", chosen)):
                r = frame.loc[gid]
                ex.append({"group": int(gid), "topology": t, "role": role, "node": int(r["node"]),
                           "group_size": int(g.size()[gid]), **{c: float(r[c]) for c in cols}})
    return gap.reset_index(), pd.DataFrame(ex)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/e9_rich.npz")
    parser.add_argument("--max-train", type=int, default=600_000)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)
    cfg = load_config()
    d = np.load(PROJECT_ROOT / args.data)
    tr, va = group_frame(d, d["split"] == 0), group_frame(d, d["split"] == 1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    out = {}
    if "components" in d.files:
        out["components"] = component_shares(va)
        print("(a) oracle component variance shares (validation groups)\n",
              out["components"].to_string(index=False, float_format=fmt), flush=True)
    out["bounds"] = bounds(tr, va, cfg, args.max_train)
    print("(b) upper bounds on validation\n", out["bounds"].to_string(index=False, float_format=fmt), flush=True)
    out["feature_gap"], out["examples"] = feature_gap(va, cfg)
    print("(c) oracle best - B3 chosen (groups where B3 is wrong)\n",
          out["feature_gap"].to_string(index=False, float_format=fmt))
    print(out["examples"].to_string(index=False, float_format=fmt))
    TABLES.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        path = TABLES / f"top1_{k}{args.suffix}.csv"
        df.to_csv(path, index=False, float_format="%.5f")
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

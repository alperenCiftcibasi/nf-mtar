"""Eşleştirilmiş istatistik: runs.csv -> results/tables/<deney>_stats.csv (CLAUDE.md §5.19 "İstatistik").

Bütün yöntemler aynı seed'lerle koştuğu için gözlemler EŞLEŞTİRİLMİŞTİR (seed = blok). Bir "hücre" =
(deney, yük, [relay_mode], [variant], sınıf, metrik); karşılaştırma bir hücrenin içinde yöntemler arasındadır.

Hücre başına üç tür satır (`kind` sütunu):
- descriptive: yöntem başına n, ortalama, std, %95 GA (t dağılımı, df = n − 1).
- pairwise:    her yöntem çifti (A, B) için, ikisinin de değeri olan ortak seed'lerde fark d = A − B:
               fark ortalaması, std, SE = std/√n, t = fark/SE, %95 GA (fark ± t_{0.975, n−1}·SE), eşleştirilmiş t-testi
               p'si, Wilcoxon işaretli sıralar (W, p; scipy varsayılanı: sıfır farklar atılır), etki büyüklüğü
               d_z = fark ortalaması / fark std (= t/√n), ve Holm–Bonferroni düzeltilmiş p'ler (p_t_holm, p_w_holm).
- friedman:    ≥ 3 yöntem ve tam bloklar (bütün yöntemlerin değeri olan seed'ler) üzerinde Friedman χ², p.

Holm ailesi (CLAUDE.md'de tanımlı değil; burada seçilen ve parametreyle değiştirilebilen): varsayılan `cell` = bir
hücredeki bütün ikili karşılaştırmalar (Friedman sonrası post-hoc için olağan aile). `experiment` seçilirse bir deneyin
bütün hücrelerindeki ikili karşılaştırmalar tek aile olur (daha tutucu). t ve Wilcoxon p'leri ayrı ayrı düzeltilir.

Tutarlılık kontrolü (§5.19; makalede bulunan hatalara düşmemek için) `check_consistency` ile, çıktı yazılmadan önce
çalışır ve tutarsızlıkta `ConsistencyError` verir:
- t = fark/SE ve d_z = t/√n (yeniden hesap), GA yarı genişliği = t_krit·SE, GA ortalamayı içerir;
- p ∈ [0, 1]; Holm p ≥ ham p, ≤ 1 ve ham p sırasıyla monoton;
- Wilcoxon 0 ≤ W ≤ n'(n'+1)/2 (n' = sıfır olmayan fark sayısı);
- Friedman 0 ≤ χ² ≤ n(k − 1) (n blok, k yöntem); descriptive GA ortalamayı içerir.

Kullanım:
    python experiments/analysis/stats.py [--runs results/tables/runs.csv] [--experiments E1 E2]
        [--out-dir results/tables] [--holm-family cell|experiment]
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSES: tuple[str, ...] = ("critical", "regular", "noncritical")   # E / L / R (§5.6)
NETWORK_CLASS = "all"                                               # ağ geneli metrikler için sınıf etiketi
METHOD_ORDER: tuple[str, ...] = ("B1-plain", "B1", "B2", "B3", "NF-MTAR", "NF-MTAR-paper")   # §5.12, §5.19 E5

# Varsayılan metrikler (§5.13). Sınıf metrikleri runs.csv'de `<metrik>_<sınıf>` sütunlarıdır.
CLASS_METRICS: tuple[str, ...] = (
    "pdr", "norm_throughput", "throughput_kbps", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms",
    "deadline_ratio", "window_ratio", "airtime_share", "hops_mean", "phi",
)
NETWORK_METRICS: tuple[str, ...] = (
    "pdr_total", "phi", "energy_total_J", "fnd_s", "hnd_s", "dead_nodes", "root_changes_total",
)
CELL_KEYS: tuple[str, ...] = ("experiment", "load_multiplier", "relay_mode", "variant")
ALPHA_CI = 0.05   # %95 GA (configs/default.yaml experiments.statistics.ci_level)
RTOL = 1e-6


class ConsistencyError(ValueError):
    """İstatistik tablosu iç tutarlılık kontrolünü geçemedi (§5.19)."""


# --------------------------------------------------------------------------- #
# Veri: runs.csv -> uzun biçim
# --------------------------------------------------------------------------- #


def load_runs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_runs(df)


def normalize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik isteğe bağlı anahtar sütunlarını doldurur (variant, relay_mode, experiment)."""
    df = df.copy()
    for key in ("experiment", "relay_mode", "variant"):
        df[key] = df[key].fillna("").astype(str) if key in df.columns else ""
    if "load_multiplier" not in df.columns:
        df["load_multiplier"] = math.nan
    return df


def to_long(runs: pd.DataFrame, class_metrics: Sequence[str] = CLASS_METRICS,
            network_metrics: Sequence[str] = NETWORK_METRICS) -> pd.DataFrame:
    """runs.csv -> (hücre anahtarları, method, seed, class, metric, value); bulunmayan sütunlar atlanır."""
    runs = normalize_runs(runs)
    ids = [*CELL_KEYS, "method", "seed"]
    parts = []
    for m in class_metrics:
        for cls in CLASSES:
            col = f"{m}_{cls}"
            if col in runs.columns:
                parts.append(runs[ids].assign(**{"class": cls, "metric": m, "value": runs[col].astype(float)}))
    for m in network_metrics:
        if m in runs.columns:
            parts.append(runs[ids].assign(**{"class": NETWORK_CLASS, "metric": m, "value": runs[m].astype(float)}))
    if not parts:
        return pd.DataFrame(columns=[*ids, "class", "metric", "value"])
    return pd.concat(parts, ignore_index=True)


def method_sort_key(m: str) -> tuple[int, str]:
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m)


# --------------------------------------------------------------------------- #
# Temel istatistikler
# --------------------------------------------------------------------------- #


def mean_ci(x: Sequence[float], alpha: float = ALPHA_CI) -> dict[str, float]:
    """Ortalama ve t dağılımlı (1 − alpha) GA; n < 2'de GA NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = float(x.mean()) if n else math.nan
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    if n > 1:
        half = float(sps.t.ppf(1 - alpha / 2, n - 1)) * sd / math.sqrt(n)
    else:
        half = math.nan
    return {"n": n, "mean": mean, "sd": sd, "ci_lo": mean - half, "ci_hi": mean + half}


def paired_compare(a: Sequence[float], b: Sequence[float], alpha: float = ALPHA_CI) -> dict[str, float]:
    """Eşleştirilmiş karşılaştırma, d = a − b (aynı sırada seed'ler; NaN içeren çiftler atılır).

    Farkların std'si 0 ise (bütün farklar eşit) t, d_z, GA ve p tanımsızdır (NaN).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_compare: a ve b aynı uzunlukta olmalı")
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    out: dict[str, float] = {"n": n, "mean_a": float(a[ok].mean()) if n else math.nan,
                             "mean_b": float(b[ok].mean()) if n else math.nan,
                             "mean_diff": float(d.mean()) if n else math.nan}
    sd = float(d.std(ddof=1)) if n > 1 else math.nan
    out["sd_diff"] = sd
    nan_keys = ("se", "t", "df", "p_t", "ci_lo", "ci_hi", "d_z")
    if n < 2 or not sd > 0:
        out.update({k: math.nan for k in nan_keys})
    else:
        se = sd / math.sqrt(n)
        t = out["mean_diff"] / se
        tcrit = float(sps.t.ppf(1 - alpha / 2, n - 1))
        out.update({
            "se": se, "t": t, "df": n - 1,
            "p_t": float(2 * sps.t.sf(abs(t), n - 1)),
            "ci_lo": out["mean_diff"] - tcrit * se, "ci_hi": out["mean_diff"] + tcrit * se,
            "d_z": out["mean_diff"] / sd,
        })
    nz = int(np.count_nonzero(d))
    out["n_nonzero"] = nz
    if nz >= 1:
        res = sps.wilcoxon(d)   # zero_method='wilcox': sıfır farklar atılır
        out["w"] = float(res.statistic)
        out["p_w"] = float(res.pvalue)
    else:
        out["w"] = math.nan
        out["p_w"] = math.nan
    return out


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni düzeltmesi (adım adım aşağı). NaN p'ler ailede sayılmaz ve NaN kalır.

    Sıralı p_(1) ≤ … ≤ p_(m) için p̃_(i) = max_{j ≤ i} min(1, (m − j + 1)·p_(j)).
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def friedman(blocks: np.ndarray) -> dict[str, float]:
    """Friedman testi; blocks: (n_blok, k_yöntem), NaN içermez. k < 3 veya n < 2 ise NaN."""
    n, k = blocks.shape
    if k < 3 or n < 2:
        return {"n": n, "k": k, "chi2": math.nan, "p": math.nan}
    if np.all(blocks == blocks[:, :1]):   # her blokta bütün yöntemler eşit: test tanımsız
        return {"n": n, "k": k, "chi2": math.nan, "p": math.nan}
    res = sps.friedmanchisquare(*blocks.T)
    return {"n": n, "k": k, "chi2": float(res.statistic), "p": float(res.pvalue)}


# --------------------------------------------------------------------------- #
# Hücre analizi
# --------------------------------------------------------------------------- #


def analyze_cell(cell: pd.DataFrame, methods: Sequence[str] | None = None) -> list[dict]:
    """Bir hücrenin (tek deney/yük/sınıf/metrik) descriptive + pairwise + friedman satırları."""
    dup = cell.duplicated(["method", "seed"])
    if dup.any():
        raise ValueError(f"aynı (yöntem, seed) birden çok kez: {cell.loc[dup, ['method', 'seed']].values.tolist()[:5]}")
    wide = cell.pivot(index="seed", columns="method", values="value")
    present = sorted(wide.columns, key=method_sort_key)
    methods = [m for m in (methods or present) if m in present]
    rows: list[dict] = []
    for m in methods:
        rows.append({"kind": "descriptive", "method": m, **mean_ci(wide[m].to_numpy())})
    for ma, mb in itertools.combinations(methods, 2):
        r = paired_compare(wide[ma].to_numpy(), wide[mb].to_numpy())
        rows.append({"kind": "pairwise", "method": ma, "method_b": mb, **r})
    if len(methods) >= 3:
        complete = wide[methods].dropna()
        r = friedman(complete.to_numpy())
        rows.append({"kind": "friedman", "method": ",".join(methods), "n": r["n"], "k": r["k"],
                     "chi2": r["chi2"], "p": r["p"]})
    return rows


def compute_stats(runs: pd.DataFrame, experiments: Iterable[str] | None = None, holm_family: str = "cell",
                  class_metrics: Sequence[str] = CLASS_METRICS,
                  network_metrics: Sequence[str] = NETWORK_METRICS) -> dict[str, pd.DataFrame]:
    """Deney başına istatistik tablosu; çıkmadan önce tutarlılık kontrolü yapılır."""
    if holm_family not in ("cell", "experiment"):
        raise ValueError("holm_family: 'cell' veya 'experiment'")
    long = to_long(runs, class_metrics, network_metrics).dropna(subset=["value"])
    if experiments is not None:
        long = long[long["experiment"].isin(list(experiments))]
    group_keys = [*CELL_KEYS, "class", "metric"]
    out: dict[str, pd.DataFrame] = {}
    for exp, exp_df in long.groupby("experiment", sort=True):
        rows = []
        for key, cell in exp_df.groupby(group_keys, sort=True, dropna=False):
            base = dict(zip(group_keys, key))
            rows.extend({**base, **r} for r in analyze_cell(cell))
        table = pd.DataFrame(rows)
        table = apply_holm(table, holm_family)
        check_consistency(table)
        out[str(exp)] = table
    return out


def apply_holm(table: pd.DataFrame, family: str = "cell") -> pd.DataFrame:
    table = table.copy()
    table["p_t_holm"] = math.nan
    table["p_w_holm"] = math.nan
    pw = table["kind"] == "pairwise"
    if not pw.any():
        return table
    fam_keys = [*CELL_KEYS, "class", "metric"] if family == "cell" else ["experiment"]
    for _, g in table[pw].groupby(fam_keys, dropna=False):
        idx = g.index
        table.loc[idx, "p_t_holm"] = holm(table.loc[idx, "p_t"].to_numpy(dtype=float))
        table.loc[idx, "p_w_holm"] = holm(table.loc[idx, "p_w"].to_numpy(dtype=float))
    table.attrs["holm_family"] = family
    return table


# --------------------------------------------------------------------------- #
# Tutarlılık kontrolü (§5.19)
# --------------------------------------------------------------------------- #


def _close(a: float, b: float, rtol: float = RTOL, atol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def check_consistency(table: pd.DataFrame, alpha: float = ALPHA_CI) -> None:
    """Tablo satırlarını yeniden hesapla doğrular; ilk tutarsızlıkta ConsistencyError (satır ve neden)."""
    errors: list[str] = []

    def bad(i, msg):
        errors.append(f"satır {i}: {msg}")

    for i, r in table.iterrows():
        kind = r["kind"]
        if kind == "descriptive":
            if _isnum(r.get("ci_lo")) and not (r["ci_lo"] - 1e-12 <= r["mean"] <= r["ci_hi"] + 1e-12):
                bad(i, f"GA ortalamayı içermiyor ({r['ci_lo']}, {r['mean']}, {r['ci_hi']})")
        elif kind == "pairwise":
            n = r["n"]
            if _isnum(r.get("t")):
                se = r["sd_diff"] / math.sqrt(n)
                if not _close(r["se"], se):
                    bad(i, f"SE = sd/√n değil ({r['se']} vs {se})")
                if not _close(r["t"], r["mean_diff"] / r["se"]):
                    bad(i, f"t ≠ fark/SE ({r['t']} vs {r['mean_diff'] / r['se']})")
                if not _close(r["d_z"], r["t"] / math.sqrt(n)):
                    bad(i, f"d_z ≠ t/√n ({r['d_z']} vs {r['t'] / math.sqrt(n)})")
                if r["df"] != n - 1:
                    bad(i, f"df ≠ n − 1 ({r['df']}, n = {n})")
                tcrit = float(sps.t.ppf(1 - alpha / 2, n - 1))
                if not _close((r["ci_hi"] - r["ci_lo"]) / 2, tcrit * r["se"]):
                    bad(i, "GA yarı genişliği ≠ t_krit·SE")
                if not (r["ci_lo"] <= r["mean_diff"] <= r["ci_hi"]):
                    bad(i, "GA fark ortalamasını içermiyor")
                p_expected = float(2 * sps.t.sf(abs(r["t"]), n - 1))
                if not _close(r["p_t"], p_expected, rtol=1e-6, atol=1e-15):
                    bad(i, f"p_t t ve df ile tutarsız ({r['p_t']} vs {p_expected})")
            if _isnum(r.get("w")):
                nz = r["n_nonzero"]
                if not (0 <= r["w"] <= nz * (nz + 1) / 2):
                    bad(i, f"Wilcoxon W = {r['w']} [0, n'(n'+1)/2 = {nz * (nz + 1) / 2}] dışında")
            for pcol in ("p_t", "p_w", "p_t_holm", "p_w_holm"):
                if _isnum(r.get(pcol)) and not (0.0 <= r[pcol] <= 1.0):
                    bad(i, f"{pcol} = {r[pcol]} [0, 1] dışında")
            for raw, adj in (("p_t", "p_t_holm"), ("p_w", "p_w_holm")):
                if adj in r and _isnum(r.get(raw)) and _isnum(r.get(adj)) and r[adj] < r[raw] - 1e-15:
                    bad(i, f"{adj} < {raw}")
        elif kind == "friedman":
            if _isnum(r.get("chi2")):
                upper = r["n"] * (r["k"] - 1)
                if not (-1e-9 <= r["chi2"] <= upper + 1e-9):
                    bad(i, f"Friedman χ² = {r['chi2']} [0, n(k−1) = {upper}] dışında")
                if not (0.0 <= r["p"] <= 1.0):
                    bad(i, f"Friedman p = {r['p']} [0, 1] dışında")
        else:
            bad(i, f"bilinmeyen kind {kind!r}")

    # Holm monotonluğu: aile içinde ham p sırasıyla düzeltilmiş p azalmamalı
    if "p_t_holm" in table.columns and (table["kind"] == "pairwise").any():
        pw = table[table["kind"] == "pairwise"]
        fam = table.attrs.get("holm_family", "cell")
        fam_keys = [*CELL_KEYS, "class", "metric"] if fam == "cell" else ["experiment"]
        fam_keys = [k for k in fam_keys if k in pw.columns]
        for _, g in pw.groupby(fam_keys, dropna=False):
            for raw, adj in (("p_t", "p_t_holm"), ("p_w", "p_w_holm")):
                s = g[[raw, adj]].dropna().sort_values(raw, kind="stable")
                if (np.diff(s[adj].to_numpy()) < -1e-15).any():
                    errors.append(f"{adj} ham p sırasıyla monoton değil (aile {g.index.tolist()[:3]}…)")

    if errors:
        raise ConsistencyError("; ".join(errors[:10]) + (f" (+{len(errors) - 10} daha)" if len(errors) > 10 else ""))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def safe_name(exp: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", exp) or "unlabeled"


def write_stats(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for exp, table in tables.items():
        p = out_dir / f"{safe_name(exp)}_stats.csv"
        table.to_csv(p, index=False)
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(PROJECT_ROOT / "results" / "tables" / "runs.csv"))
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "tables"))
    parser.add_argument("--holm-family", choices=["cell", "experiment"], default="cell")
    args = parser.parse_args(argv)
    if not Path(args.runs).exists():
        print(f"runs.csv yok: {args.runs}", file=sys.stderr)
        return 1
    tables = compute_stats(load_runs(args.runs), args.experiments, args.holm_family)
    for p in write_stats(tables, args.out_dir):
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

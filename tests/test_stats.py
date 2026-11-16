"""Testler: experiments/analysis/stats.py — eşleştirilmiş karşılaştırma, Friedman, Holm, tutarlılık kontrolü (CLAUDE.md §5.19).

- Sentetik eşleştirilmiş veride bilinen sonuçlar (elle hesap + scipy çapraz kontrolü),
- Holm–Bonferroni'nin elle hesaplanmış örneği,
- tutarlılık kontrolünün bozuk girdide hata vermesi.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ANALYSIS = Path(__file__).resolve().parents[1] / "experiments" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load("stats")


def synthetic_runs(n_seeds: int = 12, seed: int = 0) -> pd.DataFrame:
    """Üç yöntem, iki yük, eşleştirilmiş (seed etkisi ortak) sentetik runs.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for load in (0.5, 1.0):
        block = rng.normal(0.0, 0.05, n_seeds)        # seed (blok) etkisi: bütün yöntemlerde ortak
        for method, shift in (("B1", 0.0), ("B3", 0.03), ("NF-MTAR", 0.06)):
            noise = rng.normal(0.0, 0.01, n_seeds)
            for i, s in enumerate(range(1, n_seeds + 1)):
                row = {"run_id": f"E1-{method}-s{s}-l{load:g}", "experiment": "E1", "method": method, "seed": s,
                       "load_multiplier": load, "relay_mode": "strict"}
                for cls, base in (("critical", 0.8), ("regular", 0.7), ("noncritical", 0.5)):
                    row[f"pdr_{cls}"] = base + shift + block[i] + noise[i]
                    row[f"lat_mean_ms_{cls}"] = 50.0 - 100 * shift + 10 * block[i] + noise[i]
                row["pdr_total"] = 0.7 + shift + block[i] + noise[i]
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bilinen sonuçlar
# --------------------------------------------------------------------------- #


def test_paired_compare_hand_computed():
    # d = [1, 2, 3, 4, 5]: ortalama 3, sd = √2.5, SE = √0.5, t = 3/√0.5 = 4.2426, d_z = 3/√2.5 = 1.8974
    a = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = st.paired_compare(a, b)
    assert r["n"] == 5
    assert r["mean_diff"] == pytest.approx(3.0)
    assert r["sd_diff"] == pytest.approx(math.sqrt(2.5))
    assert r["se"] == pytest.approx(math.sqrt(0.5))
    assert r["t"] == pytest.approx(3.0 / math.sqrt(0.5))
    assert r["d_z"] == pytest.approx(3.0 / math.sqrt(2.5))
    tcrit = sps.t.ppf(0.975, 4)
    assert r["ci_lo"] == pytest.approx(3.0 - tcrit * math.sqrt(0.5))
    assert r["ci_hi"] == pytest.approx(3.0 + tcrit * math.sqrt(0.5))
    # Wilcoxon: bütün farklar pozitif -> W = min(R+, R−) = 0; kesin iki yönlü p = 2/2^5 = 0.0625
    assert r["w"] == 0.0
    assert r["p_w"] == pytest.approx(0.0625)


def test_paired_compare_matches_scipy():
    rng = np.random.default_rng(42)
    a = rng.normal(1.0, 1.0, 20)
    b = a - rng.normal(0.3, 0.5, 20)
    r = st.paired_compare(a, b)
    ref_t = sps.ttest_rel(a, b)
    ref_w = sps.wilcoxon(a - b)
    assert r["t"] == pytest.approx(ref_t.statistic)
    assert r["p_t"] == pytest.approx(ref_t.pvalue)
    assert r["w"] == pytest.approx(ref_w.statistic)
    assert r["p_w"] == pytest.approx(ref_w.pvalue)
    ci = ref_t.confidence_interval(0.95)
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((ci.low, ci.high))
    assert r["d_z"] == pytest.approx(r["t"] / math.sqrt(20))


def test_paired_compare_drops_nan_pairs_and_degenerate():
    r = st.paired_compare([1.0, np.nan, 3.0, 4.0], [0.0, 1.0, np.nan, 3.0])
    assert r["n"] == 2 and r["mean_diff"] == pytest.approx(1.0)
    assert math.isnan(r["t"])                                  # iki fark da 1: sd = 0 -> t tanımsız
    r0 = st.paired_compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])   # bütün farklar sıfır
    assert math.isnan(r0["t"]) and math.isnan(r0["w"])


def test_mean_ci_t_distribution():
    x = [1.0, 2.0, 3.0, 4.0]
    r = st.mean_ci(x)
    half = sps.t.ppf(0.975, 3) * np.std(x, ddof=1) / 2.0
    assert r["mean"] == 2.5
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((2.5 - half, 2.5 + half))


def test_friedman_matches_scipy():
    rng = np.random.default_rng(3)
    blocks = rng.normal(size=(15, 4)) + np.array([0.0, 0.2, 0.5, 1.0])
    r = st.friedman(blocks)
    ref = sps.friedmanchisquare(*blocks.T)
    assert r["chi2"] == pytest.approx(ref.statistic)
    assert r["p"] == pytest.approx(ref.pvalue)
    # mükemmel sıralama: χ² üst sınırı n(k − 1)
    perfect = np.tile(np.arange(4.0), (10, 1))
    assert st.friedman(perfect)["chi2"] == pytest.approx(10 * 3)


def test_compute_stats_on_synthetic_runs():
    runs = synthetic_runs()
    tables = st.compute_stats(runs)
    assert set(tables) == {"E1"}
    t = tables["E1"]
    cell = t[(t["load_multiplier"] == 1.0) & (t["class"] == "critical") & (t["metric"] == "pdr")]
    assert set(cell["kind"]) == {"descriptive", "pairwise", "friedman"}
    assert (cell["kind"] == "pairwise").sum() == 3                       # 3 yöntem -> 3 çift
    # yöntem sırası METHOD_ORDER'a göre: B1, B3, NF-MTAR
    assert cell[cell["kind"] == "descriptive"]["method"].tolist() == ["B1", "B3", "NF-MTAR"]
    pair = cell[(cell["kind"] == "pairwise") & (cell["method"] == "B1") & (cell["method_b"] == "NF-MTAR")].iloc[0]
    sub = runs[runs["load_multiplier"] == 1.0].pivot(index="seed", columns="method", values="pdr_critical")
    ref = sps.ttest_rel(sub["B1"], sub["NF-MTAR"])
    assert pair["t"] == pytest.approx(ref.statistic)
    assert pair["mean_diff"] == pytest.approx(-0.06, abs=0.01)           # bilinen kayma
    assert pair["p_t_holm"] >= pair["p_t"]
    fr = cell[cell["kind"] == "friedman"].iloc[0]
    assert fr["chi2"] == pytest.approx(sps.friedmanchisquare(sub["B1"], sub["B3"], sub["NF-MTAR"]).statistic)
    assert fr["p"] < 0.001
    # ağ geneli metrik 'all' sınıfında
    assert ((t["class"] == "all") & (t["metric"] == "pdr_total")).any()


def test_compute_stats_rejects_duplicate_method_seed():
    runs = synthetic_runs()
    runs = pd.concat([runs, runs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="birden çok"):
        st.compute_stats(runs)


def test_write_stats(tmp_path):
    paths = st.write_stats(st.compute_stats(synthetic_runs()), tmp_path)
    assert [p.name for p in paths] == ["E1_stats.csv"]
    assert len(pd.read_csv(paths[0])) > 0


# --------------------------------------------------------------------------- #
# Holm–Bonferroni
# --------------------------------------------------------------------------- #


def test_holm_hand_example():
    # sıralı: 0.005·4 = 0.02; 0.01·3 = 0.03; 0.03·2 = 0.06; 0.04·1 = 0.04 -> max(0.06, 0.04) = 0.06
    p = [0.01, 0.04, 0.03, 0.005]
    assert st.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_nan_and_clipping():
    out = st.holm([0.02, np.nan, 0.5])                     # m = 2
    assert out[0] == pytest.approx(0.04) and math.isnan(out[1]) and out[2] == pytest.approx(0.5)
    assert st.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert st.holm([]).size == 0


def test_holm_family_experiment_is_more_conservative():
    runs = synthetic_runs()
    cell = st.compute_stats(runs, holm_family="cell")["E1"]
    exp = st.compute_stats(runs, holm_family="experiment")["E1"]
    pw_c = cell[cell["kind"] == "pairwise"]["p_t_holm"].to_numpy()
    pw_e = exp[exp["kind"] == "pairwise"]["p_t_holm"].to_numpy()
    assert np.all(pw_e >= pw_c - 1e-15)


# --------------------------------------------------------------------------- #
# Tutarlılık kontrolü bozuk girdide hata verir
# --------------------------------------------------------------------------- #


@pytest.fixture
def table():
    return st.compute_stats(synthetic_runs())["E1"]


def _first(table, kind):
    return table.index[table["kind"] == kind][0]


def test_consistency_passes_on_clean_table(table):
    st.check_consistency(table)


@pytest.mark.parametrize("column, factor, match", [
    ("t", 1.5, "t ≠ fark/SE"),
    ("d_z", 2.0, "d_z ≠ t/√n"),
    ("se", 0.5, "SE"),
    ("p_t", 0.1, "p_t"),
])
def test_consistency_detects_pairwise_tampering(table, column, factor, match):
    i = _first(table, "pairwise")
    table.loc[i, column] = table.loc[i, column] * factor
    with pytest.raises(st.ConsistencyError, match=match):
        st.check_consistency(table)


def test_consistency_detects_wilcoxon_out_of_range(table):
    i = _first(table, "pairwise")
    nz = table.loc[i, "n_nonzero"]
    table.loc[i, "w"] = nz * (nz + 1) / 2 + 1
    with pytest.raises(st.ConsistencyError, match="Wilcoxon"):
        st.check_consistency(table)


def test_consistency_detects_friedman_out_of_range(table):
    i = _first(table, "friedman")
    table.loc[i, "chi2"] = table.loc[i, "n"] * (table.loc[i, "k"] - 1) + 5
    with pytest.raises(st.ConsistencyError, match="Friedman"):
        st.check_consistency(table)


def test_consistency_detects_holm_below_raw(table):
    i = _first(table, "pairwise")
    table.loc[i, "p_t_holm"] = table.loc[i, "p_t"] / 2
    with pytest.raises(st.ConsistencyError, match="p_t_holm"):
        st.check_consistency(table)


def test_consistency_detects_descriptive_ci(table):
    i = _first(table, "descriptive")
    table.loc[i, "ci_lo"] = table.loc[i, "mean"] + 1
    with pytest.raises(st.ConsistencyError, match="GA"):
        st.check_consistency(table)


def test_consistency_detects_p_out_of_unit_interval(table):
    i = _first(table, "pairwise")
    table.loc[i, "p_w"] = 1.5
    with pytest.raises(st.ConsistencyError, match="p_w"):
        st.check_consistency(table)


# --------------------------------------------------------------------------- #
# Makale karşılaştırması: göreli iyileşme ve trend uyumu (paper_comparison.py)
# --------------------------------------------------------------------------- #

pc = _load("paper_comparison")


def test_relative_improvement_sign_convention():
    # pozitif = NF-MTAR daha iyi: throughput büyük-iyi, gecikme küçük-iyi
    assert pc.rel_improvement(95.2, 71.4, higher_is_better=True) == pytest.approx((95.2 - 71.4) / 71.4)
    assert pc.rel_improvement(18.4, 67.3, higher_is_better=False) == pytest.approx((67.3 - 18.4) / 67.3)
    assert math.isnan(pc.rel_improvement(1.0, math.nan, True))
    assert pc.same_sign(0.3, 0.1) == 1.0 and pc.same_sign(0.3, -0.1) == 0.0 and math.isnan(pc.same_sign(0.3, math.nan))


def test_build_comparison_on_synthetic_runs():
    runs = synthetic_runs()
    runs["method"] = runs["method"].replace({"B1": "B1-plain"})        # makalenin RPL'si <-> B1-plain
    runs["norm_throughput_critical"] = runs["pdr_critical"]
    table = pc.build_comparison(runs, pc.DEFAULT_REFERENCE)
    row = table[(table["metric"] == "throughput_pct") & (table["x"] == "critical") & (table["load"] == 1.0)].iloc[0]
    sub = runs[runs["load_multiplier"] == 1.0]
    assert row["ours_nfmtar"] == pytest.approx(100 * sub.loc[sub["method"] == "NF-MTAR", "pdr_critical"].mean())
    assert row["ours_n_seeds"] == 12
    assert row["trend_agreement"] == 1.0                               # ikisinde de NF-MTAR daha iyi
    # E2/E4/paket logu yok: makale sütunu dolu, bizim sütunumuz boş
    sc = table[table["metric"] == "scalability_latency_ms"]
    assert sc["paper_nfmtar"].notna().all() and sc["ours_nfmtar"].isna().all()

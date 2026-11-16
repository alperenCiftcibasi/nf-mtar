"""Testler: engine/fuzzy.py, engine/fuzzy_torch.py, engine/base.py, engine/baselines.py (CLAUDE.md §5.11, §5.14, §5.15, §5.16).

§5.14 test listesi: üyelikler [0,1], merkezde 1, komşular 0,5'te kesişir; monotonluk (x3 artınca skor artmaz, x4 artınca
azalmaz); güven ∈ [1/750, 1]; NumPy ↔ PyTorch eşdeğerliği; ∂f/∂y_k = w_k/Σw (autograd vs elle).
Kalibrasyon raporu (makale Kural 1/47/98): fark log'a yazılır, test başarısız olmaz (§5.14).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch

from nfmtar.config import load_config
from nfmtar.engine.base import FEATURE_NAMES, NodeFeatures, ScoreOutput, ScoringStrategy
from nfmtar.engine.baselines import FuzzyStrategy, StaticRootStrategy, b3_strategy, strategy_for_method
from nfmtar.engine.fuzzy import (
    DONT_CARE,
    FuzzyEngine,
    FuzzyParams,
    FuzzySpec,
    extract_features,
    full_rule_table,
    gaussian_mf,
    half_crossing_sigma,
    nominal_rates_pps,
)
from nfmtar.engine.fuzzy_torch import TorchFuzzyEngine, torch_infer
from nfmtar.network.radio import RadioModel
from tests.test_router import make_net

N_RANDOM = 1000


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def engine(cfg):
    return FuzzyEngine.from_config(cfg)


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(12345).random((N_RANDOM, 5))


def perturbed_params(spec: FuzzySpec, seed: int = 7) -> FuzzyParams:
    """'Eğitilmiş' benzeri parametreler: merkez/σ/y rastgele oynatılmış (eşdeğerlik testleri uzman simetrisine dayanmasın)."""
    rng = np.random.default_rng(seed)
    p = FuzzyParams.expert(spec)
    return FuzzyParams(
        centers=[c + rng.normal(0, 0.03, c.shape) for c in p.centers],
        sigmas=[s * rng.uniform(0.8, 1.25, s.shape) for s in p.sigmas],
        y=np.clip(p.y + rng.normal(0, 0.05, p.y.shape), 0.11, 0.99),
        table=p.table,
    )


# ----------------------------------------------------------------------------- üyelik fonksiyonları (Denk. 9)


def test_sigma_values_match_claude_md():
    """§5.14: 5 terim σ ≈ 0,150; 3 terim ≈ 0,300; 2 terim ≈ 0,601."""
    assert round(half_crossing_sigma(5), 3) == 0.150
    assert round(half_crossing_sigma(3), 3) == 0.300
    assert round(half_crossing_sigma(2), 3) == 0.601


def test_memberships_in_unit_interval_one_at_center_half_at_crossing(engine, X):
    """§5.14: μ ∈ [0,1]; terim merkezinde μ = 1; komşu terimler orta noktada 0,5'te kesişir; MATLAB gaussmf DEĞİL."""
    for m in engine.memberships(X):
        assert np.all((m >= 0) & (m <= 1))
    for i, (c, s) in enumerate(zip(engine.params.centers, engine.params.sigmas)):
        assert np.allclose(gaussian_mf(c, c, s), 1.0)
        mid = (c[:-1] + c[1:]) / 2
        left, right = gaussian_mf(mid, c[:-1], s[:-1]), gaussian_mf(mid, c[1:], s[1:])
        assert np.allclose(left, 0.5) and np.allclose(right, 0.5), FEATURE_NAMES[i]
    # Denk. 9 (paydada 2σ² yok): x = c + σ'da μ = e^-1
    assert math.isclose(gaussian_mf(0.3, 0.0, 0.3), math.exp(-1.0))


# ----------------------------------------------------------------------------- kural tabanı


def test_full_rule_table_and_expert_consequents(engine, cfg):
    """§5.14: 5×5×5×2×3 = 750 kuralın tamamı, tekrarsız; y_k = 0,1 + 0,9 Σ a_i g_i ∈ [0,1; 1]."""
    t = engine.params.table
    assert t.shape == (750, 5) and len({tuple(r) for r in t}) == 750
    assert (t.max(axis=0) + 1).tolist() == [5, 5, 5, 2, 3]
    assert t[0].tolist() == [0, 0, 0, 0, 0] and t[1].tolist() == [0, 0, 0, 0, 1]   # x5 en hızlı
    y = engine.params.y
    assert y.min() >= 0.1 - 1e-12 and y.max() <= 1.0 + 1e-12
    best = engine.rule_index([4, 0, 0, 1, 2])     # x1 VeryHigh, x2/x3 VeryLow, x4 High, x5 High
    worst = engine.rule_index([0, 4, 4, 0, 0])
    assert math.isclose(y[best], 1.0) and math.isclose(y[worst], 0.1)
    # elle bir kural: x1 High(g=.75), x2 Low(g=.75), x3 Medium(g=.5), x4 Low(g=0), x5 High(g=1)
    k = engine.rule_index([3, 1, 2, 0, 2])
    a = cfg.fuzzy["consequent"]["weights"]
    expected = 0.1 + 0.9 * (a["x1"] * 0.75 + a["x2"] * 0.75 + a["x3"] * 0.5 + a["x4"] * 0.0 + a["x5"] * 1.0)
    assert math.isclose(y[k], expected)


def test_rule_to_text(engine):
    """Canlı demo için insan okunur kural metni."""
    k = engine.rule_index([4, 0, 0, 1, 2])
    txt = engine.rule_to_text(k)
    assert txt == (f"R{k}: IF x1 (demand) is VeryHigh AND x2 (utilization) is VeryLow AND x3 (depletion) is VeryLow "
                   "AND x4 (link_quality) is High AND x5 (connectivity) is High THEN 1.00")
    assert engine.rule_to_text(0, with_labels=False).startswith("R0: IF x1 is VeryLow AND x2 is VeryLow")


# ----------------------------------------------------------------------------- çıkarım (Denk. 11–13)


def test_inference_matches_explicit_formula(engine, X):
    """Denk. 11–13: w_k = Π μ; f = Σ w y / Σ w; f_norm = clip((f − .1)/.9); güven = max w / Σ w — döngüyle elle hesap."""
    x = X[3]
    p = engine.params
    w = np.array([np.prod([math.exp(-((x[i] - p.centers[i][r[i]]) / p.sigmas[i][r[i]]) ** 2) for i in range(5)])
                  for r in p.table])
    f = float(w @ p.y / w.sum())
    out = engine.evaluate(x)
    assert math.isclose(out.f, f, rel_tol=1e-12)
    assert math.isclose(out.score, min(1.0, max(0.0, (f - 0.1) / 0.9)), rel_tol=1e-12)
    assert math.isclose(out.confidence, w.max() / w.sum(), rel_tol=1e-12)
    assert out.top_rule_index == int(np.argmax(w))


def test_evaluate_equals_batch_and_accepts_node_features(engine, X):
    batch = engine.evaluate_batch(X[:20])
    for b in range(20):
        nf = NodeFeatures(b, *X[b])
        single = engine.evaluate(nf)
        assert single.score == pytest.approx(batch.score[b], abs=1e-15)
        assert single.top_rule_index == batch.top_rule_index[b]
    assert np.array_equal(engine.evaluate_batch([NodeFeatures(b, *X[b]) for b in range(20)]).score, batch.score)


def test_monotonicity_x3_and_x4_random_points(engine, X):
    """§5.14: 1000 rastgele noktada x3 artınca skor ARTMAZ, x4 artınca skor AZALMAZ (diğer girdiler sabit)."""
    rng = np.random.default_rng(99)
    for i, sign in ((2, -1), (3, +1)):   # x3 (indeks 2): artmaz; x4 (indeks 3): azalmaz
        lo = X.copy()
        hi = X.copy()
        hi[:, i] = lo[:, i] + rng.random(N_RANDOM) * (1.0 - lo[:, i])
        d = engine.evaluate_batch(hi).f - engine.evaluate_batch(lo).f
        assert np.all(sign * d >= -1e-12), (FEATURE_NAMES[i], d.min(), d.max())
        assert np.mean(sign * d > 1e-9) > 0.9            # gerçekten etkili (sabit değil)


def test_monotonicity_other_directions(engine, X):
    """Uzman sonuçlarıyla yön tutarlılığı: x1, x5 artınca azalmaz; x2 artınca artmaz."""
    for i, sign in ((0, +1), (1, -1), (4, +1)):
        hi = X.copy()
        hi[:, i] = np.minimum(1.0, X[:, i] + 0.2)
        d = engine.evaluate_batch(hi).f - engine.evaluate_batch(X).f
        assert np.all(sign * d >= -1e-12), FEATURE_NAMES[i]


def test_confidence_bounds(engine, X):
    """§5.14 güven ∈ [1/750, 1]; skor ∈ [0,1]; uç noktalarda da."""
    corners = np.array([[a, b, c, d, e] for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1) for e in (0, 1)], float)
    for pts in (X, corners, np.full((1, 5), 0.5)):
        out = engine.evaluate_batch(pts)
        assert np.all(out.confidence >= 1 / 750 - 1e-15) and np.all(out.confidence <= 1 + 1e-15)
        assert np.all((out.score >= 0) & (out.score <= 1))


def test_node_features_validation():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        NodeFeatures(0, 1.2, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        NodeFeatures(0, float("nan"), 0, 0, 0, 0)


# ----------------------------------------------------------------------------- PyTorch eşdeğeri (§5.14, §5.16)


@pytest.mark.parametrize("params_kind", ["expert", "perturbed"])
def test_numpy_and_torch_agree_within_1e6(cfg, X, params_kind):
    """§5.14: aynı girdide NumPy ve PyTorch çıktıları 1e-6 içinde (f, f_norm, güven, baskın kural)."""
    spec = FuzzySpec.from_config(cfg)
    params = FuzzyParams.expert(spec) if params_kind == "expert" else perturbed_params(spec)
    np_out = FuzzyEngine(spec, params).evaluate_batch(X)
    tm = TorchFuzzyEngine.from_config(cfg, params)
    with torch.no_grad():
        t_out = tm(torch.from_numpy(X))
    assert np.max(np.abs(t_out["f"].numpy() - np_out.f)) < 1e-6
    assert np.max(np.abs(t_out["f_norm"].numpy() - np_out.score)) < 1e-6
    assert np.max(np.abs(t_out["confidence"].numpy() - np_out.confidence)) < 1e-6
    assert np.array_equal(t_out["top_rule"].numpy(), np_out.top_rule_index)
    back = tm.to_params()   # ters dönüşümler (softplus⁻¹, logit) parametreleri korur
    assert all(np.allclose(a, b, atol=1e-9) for a, b in zip(back.sigmas, params.sigmas))
    assert all(np.allclose(a, b, atol=1e-12) for a, b in zip(back.centers, params.centers))
    assert np.allclose(back.y, params.y, atol=1e-9)


def test_autograd_df_dy_equals_normalized_firing(cfg, X):
    """§5.16: ∂f/∂y_k = w_k / Σ w (autograd, elle hesapla karşılaştırma; NumPy motorunun w'su ile)."""
    spec = FuzzySpec.from_config(cfg)
    params = perturbed_params(spec, seed=3)
    np_engine = FuzzyEngine(spec, params)
    tm = TorchFuzzyEngine.from_config(cfg, params)
    for b in (0, 17, 500):
        x = torch.from_numpy(X[b:b + 1])
        y = tm.consequents().detach().clone().requires_grad_(True)
        out = torch_infer(x, [c.detach() for c in tm.centers], [s.detach() for s in tm.sigmas()], y, tm.cols(),
                          spec.y_min, spec.y_span)
        out["f"].sum().backward()
        w = np_engine.firing_strengths(X[b:b + 1])[0]
        assert np.allclose(y.grad.numpy(), w / w.sum(), atol=1e-12)


def test_torch_parametrization_constraints(cfg):
    """§5.16: σ = softplus(s) + 1e-3 > 0 ve y_k = 0,1 + 0,9·sigmoid(u) ∈ [0,1; 1] — parametreler ne olursa olsun."""
    tm = TorchFuzzyEngine.from_config(cfg)
    with torch.no_grad():
        for s in tm.s:
            s.fill_(-50.0)
        tm.u[::2] = 80.0
        tm.u[1::2] = -80.0
    assert all(bool((s >= 1e-3).all()) for s in tm.sigmas())
    y = tm.consequents().detach()
    assert float(y.min()) >= 0.1 and float(y.max()) <= 1.0
    with torch.no_grad():
        out = tm(torch.rand(10, 5, dtype=torch.float64))
    assert torch.isfinite(out["f"]).all()   # σ = 1e-3'te bütün w_k 0'a taşar; log uzayı normalizasyonu 0/0'ı önler
    spec = FuzzySpec.from_config(cfg)
    np_out = FuzzyEngine(spec, tm.to_params()).evaluate_batch(np.random.default_rng(0).random((10, 5)))
    assert np.all(np.isfinite(np_out.f))


def test_params_json_roundtrip(cfg, tmp_path, X):
    spec = FuzzySpec.from_config(cfg)
    p = perturbed_params(spec)
    path = p.save_json(tmp_path / "model.json")
    q = FuzzyParams.load_json(path)
    assert np.array_equal(FuzzyEngine(spec, p).evaluate_batch(X).f, FuzzyEngine(spec, q).evaluate_batch(X).f)


# ----------------------------------------------------------------------------- pruned (don't care)


def test_pruned_expert_has_no_merges(cfg, X):
    """Uzman sonuçlarında bütün a_i > 0: hiçbir girdi bütün terimlerde aynı sonucu vermez -> 750 kural kalır."""
    full, pruned = FuzzyEngine.from_config(cfg), FuzzyEngine.from_config(cfg, pruned=True)
    assert pruned.n_rules == 750 and pruned.pruned
    assert np.allclose(full.evaluate_batch(X).f, pruned.evaluate_batch(X).f)


def test_pruned_merges_inactive_input(cfg):
    """x5'in ağırlığı 0 olan sonuçlarda her (x1..x4) grubu x5'in 3 teriminde aynı -> 250 don't-care kuralı;
    don't-care üyeliği 1, metin x5'i yazmaz; x5'i değiştirmek pruned motorun çıkışını değiştirmez."""
    spec = FuzzySpec.from_config(cfg)
    table = full_rule_table(spec.n_terms)
    spec0 = FuzzySpec(spec.n_terms, spec.directions, (0.2, 0.2, 0.3, 0.3, 0.0), spec.y_min, spec.y_span,
                      spec.neighborhood_hops, spec.max_neighbors)
    from nfmtar.engine.fuzzy import expert_consequents

    p = FuzzyParams.expert(spec)
    p = FuzzyParams(p.centers, p.sigmas, expert_consequents(spec0, table), table)
    eng = FuzzyEngine(spec, p, pruned=True)
    assert eng.n_rules == 250 and np.all(eng.params.table[:, 4] == DONT_CARE)
    assert "x5" not in eng.rule_to_text(0)
    a, b = np.array([[0.3, 0.2, 0.1, 0.9, 0.0]]), np.array([[0.3, 0.2, 0.1, 0.9, 1.0]])
    assert eng.evaluate_batch(a).f[0] == pytest.approx(eng.evaluate_batch(b).f[0], abs=1e-15)
    tm = TorchFuzzyEngine(spec, eng.params)   # PyTorch eşdeğeri don't-care'i de destekler
    with torch.no_grad():
        assert abs(float(tm(torch.from_numpy(a))["f"][0]) - eng.evaluate_batch(a).f[0]) < 1e-6


# ----------------------------------------------------------------------------- girdi çıkarımı (§5.14 girdi tablosu)


def test_extract_features_hand_computed(cfg):
    """§5.14 tanımları elle: dört kritik sensör bir doğru üzerinde 40 m aralıkla (yalnızca komşular bağlı), Edge-E n0'ın
    10 m solunda. x1 = kapalı 2-atlama komşuluğu hızı / toplam; x4 = komşulara ort. PRR; x5 = komşu/15 (edge router
    komşu sayılmaz); x3 = tükenme; x2 dışarıdan; H dışarıdan. Ölü düğüm komşu sayılmaz ve özellik almaz."""
    radio = RadioModel.from_config(cfg)
    spec = FuzzySpec.from_config(cfg)
    net, part = make_net(cfg, [(10, 0, "critical"), (50, 0, "critical"), (90, 0, "critical"), (130, 0, "critical")],
                         energy_frac={3: 0.25})
    rates = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    occ = {1: 0.5, 2: 1.7}
    feats = extract_features(spec, net, part, "E", rates, radio.prr, occ, {0: 1, 1: 2, 2: 3, 3: 4})
    assert sorted(feats) == [0, 1, 2, 3]
    f0, f1, f2, f3 = (feats[i] for i in range(4))
    assert f0.x1 == pytest.approx(6 / 10) and f1.x1 == pytest.approx(1.0) and f3.x1 == pytest.approx(9 / 10)
    assert f0.x5 == pytest.approx(1 / 15) and f1.x5 == pytest.approx(2 / 15)       # edge router komşu değil
    assert f0.x4 == pytest.approx(radio.prr(40.0)) and f1.x4 == pytest.approx(radio.prr(40.0))
    assert f1.x2 == 0.5 and f2.x2 == 1.0 and f0.x2 == 0.0                         # kırpma, varsayılan 0
    assert f3.x3 == pytest.approx(0.75) and f0.x3 == 0.0
    assert [f.hops_to_edge for f in (f0, f1, f2, f3)] == [1, 2, 3, 4]
    net.nodes[2].kill(0.0)
    feats = extract_features(spec, net, part, "E", rates, radio.prr)
    assert sorted(feats) == [0, 1, 3]
    assert feats[3].x5 == 0.0 and feats[3].x4 == 0.0 and feats[3].x1 == pytest.approx(4 / 7)
    assert feats[1].x5 == pytest.approx(1 / 15) and feats[3].hops_to_edge is None


def test_extract_features_real_network_in_unit_interval(cfg):
    """Gerçek ağda (seed 1) bütün topolojilerin bütün canlı sensörleri için x ∈ [0,1]; motor değerlendirir."""
    from nfmtar.network.graph import TOPOLOGY_ORDER
    from nfmtar.network.partition import build_partitioned_network
    from nfmtar.router import Router

    pn = build_partitioned_network(cfg, seed=1)
    radio = RadioModel.from_config(cfg)
    spec = FuzzySpec.from_config(cfg)
    rates = nominal_rates_pps(cfg, pn.net)
    router = Router.for_method(cfg, radio, "B3")
    engine = FuzzyEngine(spec)
    for t in TOPOLOGY_ORDER:
        reach = router.reach_from_edge(pn.net, pn.part, t)
        feats = extract_features(spec, pn.net, pn.part, t, rates, radio.prr, hops_to_edge=reach.hops)
        assert set(feats) == set(pn.part.members[t])
        out = engine.evaluate_batch(list(feats.values()))
        assert np.all((out.score >= 0) & (out.score <= 1)) and len(out.score) == len(feats)
        assert all(feats[v].hops_to_edge == reach.hops[v] for v in reach.candidates)


# ----------------------------------------------------------------------------- stratejiler (§5.11, §5.12, §5.15)


def test_strategies_for_methods(cfg):
    """§5.12 tablosu: B1-plain/B1/B2 sabit root (puanlama yok); B3 uzman bulanık, dinamik root; NF-MTAR parametre ister."""
    for m, cost in (("B1-plain", "hop"), ("B1", "hop"), ("B2", "etx_depletion")):
        s = strategy_for_method(cfg, m)
        assert isinstance(s, StaticRootStrategy) and isinstance(s, ScoringStrategy)
        assert not s.dynamic_root() and s.path_cost == cost and s.name == m
        with pytest.raises(RuntimeError, match="sabit root"):
            s.score(NodeFeatures(0, 0.5, 0.5, 0.5, 0.5, 0.5))
    assert strategy_for_method(cfg, "B1-plain").vsdn_policy is False
    b3 = strategy_for_method(cfg, "B3")
    assert isinstance(b3, FuzzyStrategy) and isinstance(b3, ScoringStrategy) and b3.dynamic_root()
    nf = NodeFeatures(1, 0.9, 0.1, 0.2, 0.95, 0.6, hops_to_edge=2)
    out = b3.score(nf)
    ref = FuzzyEngine.from_config(cfg).evaluate(nf)
    assert isinstance(out, ScoreOutput) and out.score == ref.score and out.confidence == ref.confidence
    assert out.top_rule == ref.top_rule_index
    assert b3_strategy(cfg).score(nf) == out
    with pytest.raises(ValueError, match="eğitilmiş"):
        strategy_for_method(cfg, "NF-MTAR")
    trained = strategy_for_method(cfg, "NF-MTAR", perturbed_params(FuzzySpec.from_config(cfg)))
    assert trained.dynamic_root() and trained.name == "NF-MTAR" and trained.score(nf).score != out.score
    with pytest.raises(KeyError, match="bilinmeyen"):
        strategy_for_method(cfg, "RPL")


# ----------------------------------------------------------------------------- makale kuralları kalibrasyon raporu (§5.14)


def test_paper_rules_calibration_report_logged(engine, cfg, caplog):
    """§5.14: makalenin Kural 1/47/98 terimleri -> bizim y_k ve terim merkezlerinde çıkış; fark log'a, test başarısız olmaz.
    Yalnızca raporun yapısı doğrulanır (kural bulundu, değerler aralıkta)."""
    from nfmtar.engine.fuzzy import paper_rule_report

    rows = paper_rule_report(engine, cfg.fuzzy["calibration_check"]["paper_rules"])
    assert [r["paper_rule"] for r in rows] == [1, 47, 98]
    with caplog.at_level(logging.INFO):
        for r in rows:
            assert 0.1 - 1e-12 <= r["our_y"] <= 1.0 + 1e-12 and 0.0 <= r["f_norm_at_centers"] <= 1.0
            logging.getLogger(__name__).info("paper rule %d: paper %.2f, ours %.3f (diff %+.3f); f at centers %.3f",
                                             r["paper_rule"], r["paper_y"], r["our_y"], r["diff_y"], r["f_at_centers"])

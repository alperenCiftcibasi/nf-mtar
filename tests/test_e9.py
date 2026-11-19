"""Testler: engine/oracle.py, engine/trainer.py, engine/interpretability.py (CLAUDE.md §5.16, §5.20)."""

from __future__ import annotations

import numpy as np
import pytest

from nfmtar.config import load_config
from nfmtar.engine.fuzzy import FuzzyEngine, FuzzyParams, FuzzySpec
from nfmtar.engine.interpretability import interpretability_report
from nfmtar.engine.oracle import oracle_components, oracle_scores
from nfmtar.engine.trainer import coverage_penalty, evaluate, train
from nfmtar.network.radio import RadioModel
from nfmtar.router import Router
from tests.test_router import make_net

LINE = [(10, 0, "critical"), (50, 0, "critical"), (90, 0, "critical"), (130, 0, "critical")]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def spec(cfg):
    return FuzzySpec.from_config(cfg)


def _oracle(cfg, energy_frac=None):
    net, part = make_net(cfg, LINE, energy_frac=energy_frac)
    router = Router.for_method(cfg, RadioModel.from_config(cfg), "B3")
    reach = router.reach_from_edge(net, part, "E")
    return oracle_scores(cfg, router, net, part, "E", reach, {v: 1.0 for v in range(4)})


def test_oracle_range_and_energy_monotone(cfg):
    """§5.16: skor aday başına min-max ile [0,1; 1,0]; bir düğümün enerjisi düşünce kâhin skoru artmaz
    (doğrusal dizilim: yollar enerjiden bağımsız, yalnızca dayanıklılık değişir)."""
    full = _oracle(cfg)
    assert set(full) == {0, 1, 2, 3}
    vals = np.array(list(full.values()))
    assert vals.min() == pytest.approx(0.1) and vals.max() == pytest.approx(1.0)
    for v in range(4):
        tired = _oracle(cfg, energy_frac={v: 0.2})
        assert tired[v] <= full[v] + 1e-12, v


def test_oracle_components_consistent_with_scores(cfg):
    """§5.21 teşhisi: oracle_components'in son skoru oracle_scores ile aynı; ağırlıklı bileşenlerin toplamının
    min-max'ı skordur; bileşenler [0, ağırlık] aralığında."""
    net, part = make_net(cfg, LINE, energy_frac={1: 0.3})
    router = Router.for_method(cfg, RadioModel.from_config(cfg), "B3")
    reach = router.reach_from_edge(net, part, "E")
    rates = {v: 1.0 for v in range(4)}
    comp = oracle_components(cfg, router, net, part, "E", reach, rates)
    scores = oracle_scores(cfg, router, net, part, "E", reach, rates)
    assert [scores[v] for v in comp["candidates"]] == pytest.approx(comp["score"].tolist())
    raw = comp["w_delivery"] + comp["w_latency"] + comp["w_resilience"]
    assert comp["score"] == pytest.approx(0.1 + 0.9 * (raw - raw.min()) / (raw.max() - raw.min()))
    for key, w in (("w_delivery", "delivery"), ("w_latency", "latency"), ("w_resilience", "resilience")):
        assert comp[key].min() >= -1e-12 and comp[key].max() <= cfg.oracle["weights"][w] + 1e-12


def test_trainer_anchor_strength_and_metrics(spec):
    """§5.16: anchored'da λ büyüdükçe uzmandan sapma |Δθ| küçülür (küçük sentetik veri); evaluate ölçütleri tutarlı."""
    rng = np.random.default_rng(0)
    X = rng.random((300, 5))
    y = np.clip(0.9 - 0.8 * X[:, 0], 0.1, 1.0)          # uzmana ters bir hedef: eğitim parametreleri oynatmak zorunda
    dev = {}
    for lam in (1e-3, 1.0):
        res = train(spec, X[:200], y[:200], X[200:], y[200:], reg_mode="anchored", lam=lam, max_epochs=30,
                    patience=30)
        r = interpretability_report(spec, res.params)
        dev[lam] = r["dev_c"] + r["dev_sigma"] + r["dev_y"]
    assert dev[1.0] < dev[1e-3]
    f = FuzzyEngine(spec).evaluate_batch(X).f
    m = evaluate(f, f, np.repeat(np.arange(30), 10))     # kendisiyle: MSE 0, top-1 1, pişmanlık 0, Spearman 1
    assert m["mse"] == 0 and m["top1"] == 1 and m["regret"] == 0 and m["spearman"] == pytest.approx(1.0)


def test_interpretability_expert_and_corrupted(spec):
    """§5.20: uzmanda sıra bozulması 0, komşu kesişim 0,5, monotonluk ihlali 0, sapma 0; elle bozulunca ihlal > 0."""
    p0 = FuzzyParams.expert(spec)
    r0 = interpretability_report(spec, p0)
    assert r0["order_violations"] == 0 and r0["monotonicity_violation"] == 0
    assert r0["distinguishability_max"] == pytest.approx(0.5, abs=2e-3) and r0["coverage_min"] == pytest.approx(0.5, abs=2e-3)
    assert r0["dev_c"] == r0["dev_sigma"] == r0["dev_y"] == 0
    bad = p0.copy()
    bad.centers[0][[1, 2]] = bad.centers[0][[2, 1]]       # x1 Low <-> Medium merkezleri yer değiştirdi
    bad.y = bad.y[::-1].copy()                            # sonuçlar ters
    r1 = interpretability_report(spec, bad)
    assert r1["order_violations"] == 1 and r1["monotonicity_violation"] > 0.5 and r1["dev_y"] > 0


def test_coverage_penalty_zero_when_covered_positive_when_narrowed(spec):
    """§5.22 menteşe kapsama cezası: uzman parametrelerinde (komşu terimler 0,5'te kesişir) ceza 0; σ'lar yarıya
    indirilince ara noktalarda max μ < 0,5 olur ve ceza > 0; σ küçüldükçe ceza artar."""
    import torch

    p = FuzzyParams.expert(spec)
    c = [torch.as_tensor(x) for x in p.centers]
    full = coverage_penalty(c, [torch.as_tensor(s) for s in p.sigmas])
    assert float(full) < 1e-12
    half = coverage_penalty(c, [torch.as_tensor(s) * 0.5 for s in p.sigmas])
    quarter = coverage_penalty(c, [torch.as_tensor(s) * 0.25 for s in p.sigmas])
    assert 0 < float(half) < float(quarter)
    assert interpretability_report(spec, p)["coverage_min"] >= 0.5 - 1e-9


def test_anchored_cov_keeps_coverage_better_than_anchored(spec):
    """§5.22 anchored+cov: aynı sentetik veride (σ daraltmayı ödüllendiren keskin hedef) kapsama cezası, anchored'a göre
    eğitim sonrası kapsamayı daha yüksek tutar; mod adı geçerli."""
    rng = np.random.default_rng(0)
    X = rng.random((3000, 5))
    y = 0.1 + 0.9 * (np.sin(12 * X[:, 2]) > 0)                         # x3'te keskin basamaklar
    kw = dict(lam=1e-3, max_epochs=40, patience=40, batch_size=512, seed=0)
    a = train(spec, X[:2400], y[:2400], X[2400:], y[2400:], reg_mode="anchored", **kw)
    b = train(spec, X[:2400], y[:2400], X[2400:], y[2400:], reg_mode="anchored+cov", **kw)
    ca = interpretability_report(spec, a.params)["coverage_min"]
    cb = interpretability_report(spec, b.params)["coverage_min"]
    assert cb > ca

"""Testler: traffic.py — varış süreçleri, boyutlar, kritik bakım epizotları, ortak rastgele sayılar, paket kaydı (CLAUDE.md §5.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfmtar.config import ConfigError, load_config
from nfmtar.network.partition import build_partitioned_network
from nfmtar.traffic import (
    TrafficTrace,
    build_traffic_trace,
    episode_on_fraction,
    expected_rate_pps,
    generate_trace,
)

CLASSES = ("critical", "regular", "noncritical")
NO_EPISODES = {"traffic.critical_episodes.enabled": False}


def sources(n_per_class: int) -> dict[int, str]:
    """Küçük kaynak kümesi: kimlikler sınıf sırasıyla ardışık (ağdaki gibi)."""
    return {i: CLASSES[i // n_per_class] for i in range(3 * n_per_class)}


def per_source_keys(trace: TrafficTrace, traffic_class: str) -> dict[int, tuple]:
    return {src: tuple(p.key() for p in trace.by_source[src]) for src in trace.sources_by_class[traffic_class]}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def long_trace(cfg):
    """10 kaynak/sınıf, 20 000 s: hız doğrulamaları için (yaklaşık 2 500 epizot)."""
    return generate_trace(cfg, sources(10), seed=1, duration_s=20_000.0)


# ----------------------------------------------------------------------------- config (§5.6)


def test_traffic_config_matches_claude_md(cfg):
    """§5.6 tablo: boyutlar, temel hızlar, süreç türleri, epizot parametreleri."""
    t = cfg.traffic
    assert t.load_multiplier == 1.0
    assert t.classes["critical"].packet_bytes == 64 and t.classes["critical"].rate_pps == 1.0
    assert t.classes["critical"].arrival.process == "poisson"
    assert t.classes["regular"].packet_bytes == (128, 256) and t.classes["regular"].rate_pps == 1.0
    assert t.classes["regular"].arrival.process == "periodic"
    assert t.classes["regular"].arrival.jitter_frac == 0.10 and t.classes["regular"].arrival.random_phase
    assert t.classes["noncritical"].packet_bytes == (512, 1024) and t.classes["noncritical"].rate_pps == 0.2
    assert t.classes["noncritical"].arrival.process == "poisson"
    ep = t.critical_episodes
    assert ep.enabled and ep.traffic_class == "critical"
    assert (ep.mean_interval_s, ep.duration_s, ep.rate_pps) == (300, 30, 10)


def test_traffic_config_validation():
    with pytest.raises(ConfigError, match="process"):
        load_config(overrides={"traffic.classes.critical.arrival": {"process": "bursty"}})
    with pytest.raises(ConfigError, match="poisson"):
        load_config(overrides={"traffic.classes.critical.arrival": {"process": "poisson", "jitter_frac": 0.1}})
    with pytest.raises(ConfigError, match="jitter_frac"):
        load_config(overrides={"traffic.classes.regular.arrival": {"process": "periodic", "jitter_frac": 1.5}})
    with pytest.raises(ConfigError, match="packet_bytes"):
        load_config(overrides={"traffic.classes.regular.packet_bytes": [256, 128]})
    with pytest.raises(ConfigError, match="rate_pps"):
        load_config(overrides={"traffic.classes.regular.rate_pps": 0})
    with pytest.raises(ConfigError, match="traffic_class"):
        load_config(overrides={"traffic.critical_episodes.traffic_class": "urgent"})
    with pytest.raises(ConfigError, match="pozitif"):
        load_config(overrides={"traffic.critical_episodes.duration_s": -1})
    with pytest.raises(ConfigError, match="load_multiplier"):
        load_config(overrides={"traffic.load_multiplier": 0})


# ----------------------------------------------------------------------------- hızlar (§5.6)


def test_mean_rate_per_class_matches_expected_including_episodes(cfg, long_trace):
    """§5.6: uzun izde sınıf başına düğüm başı ortalama hız, beklenen değere (epizotlar dahil) %5 içinde."""
    ep = cfg.traffic.critical_episodes
    p_on = episode_on_fraction(ep)
    assert math.isclose(p_on, 1 - math.exp(-30 / 300))
    assert math.isclose(expected_rate_pps(cfg.traffic, "critical"), 1.0 * (1 - p_on) + 10 * p_on)
    assert expected_rate_pps(cfg.traffic, "regular") == 1.0 and expected_rate_pps(cfg.traffic, "noncritical") == 0.2
    for cls in CLASSES:
        measured, expected = long_trace.mean_rate_pps(cls), expected_rate_pps(cfg.traffic, cls)
        assert abs(measured - expected) / expected < 0.05, (cls, measured, expected)
    # epizot sınıfında pencere içi hız ~10, dışı ~1; epizot zaman oranı ~p_on
    assert abs(long_trace.episode_rate_pps("critical", inside=True) - 10) / 10 < 0.05
    assert abs(long_trace.episode_rate_pps("critical", inside=False) - 1.0) < 0.05
    on_time = sum(b - a for w in long_trace.episodes.values() for a, b in w)
    assert abs(on_time / (10 * 20_000.0) - p_on) / p_on < 0.10
    # epizotlar kapalıyken kritik hız temel hıza iner
    off = generate_trace(load_config(overrides=NO_EPISODES), sources(10), seed=1, duration_s=20_000.0)
    assert abs(off.mean_rate_pps("critical") - 1.0) < 0.02 and not off.episodes
    assert not any(p.in_episode for p in off.packets)


def test_regular_interarrival_within_10_percent(long_trace):
    """§5.6 düzenli trafik: ardışık varış aralıkları 1 s ± %10 içinde; ilk varış [0, 1 s) rastgele faz."""
    for src in long_trace.sources_by_class["regular"]:
        times = np.array([p.created_s for p in long_trace.by_source[src]])
        gaps = np.diff(times)
        assert gaps.min() >= 0.9 - 1e-9 and gaps.max() <= 1.1 + 1e-9
        assert gaps.std() > 0.01                    # titreşim gerçekten var
        assert 0.0 <= times[0] < 1.0


def test_episode_windows_merge_and_clip(cfg):
    """§5.6: pencereler sıralı, örtüşmez (birleşik), [0, süre] içinde, her biri en az 30 s (kırpılan sonuncu hariç)."""
    trace = generate_trace(cfg, {0: "critical"}, seed=3, duration_s=5_000.0)
    windows = trace.episodes[0]
    assert len(windows) >= 5
    for (a, b), nxt in zip(windows, list(windows[1:]) + [None]):
        assert 0.0 <= a < b <= 5_000.0
        assert b - a >= 30.0 - 1e-9 or b == 5_000.0
        if nxt is not None:
            assert nxt[0] > b
    for p in trace.packets:  # epizot bayrağı pencereyle tutarlı
        assert p.in_episode == any(a <= p.created_s < b for a, b in windows)


# ----------------------------------------------------------------------------- ortak rastgele sayılar (§5.6)


def test_same_seed_same_trace_and_class_isolation(cfg):
    """§5.6: aynı seed aynı iz; bir sınıfın hızı değişince diğer sınıfların izleri (id hariç) aynı kalır; farklı seed farklı iz."""
    src = sources(4)
    a = generate_trace(cfg, src, seed=5, duration_s=600.0)
    b = generate_trace(cfg, src, seed=5, duration_s=600.0)
    assert a.packets == b.packets and a.episodes == b.episodes
    assert [p.packet_id for p in a.packets] == list(range(len(a.packets)))
    c = generate_trace(load_config(overrides={"traffic.classes.regular.rate_pps": 2.0}), src, seed=5, duration_s=600.0)
    assert per_source_keys(c, "critical") == per_source_keys(a, "critical")
    assert per_source_keys(c, "noncritical") == per_source_keys(a, "noncritical")
    assert per_source_keys(c, "regular") != per_source_keys(a, "regular")
    assert c.episodes == a.episodes
    d = generate_trace(cfg, src, seed=6, duration_s=600.0)
    assert d.packets != a.packets
    # epizot parametresi yalnızca epizot sınıfını etkiler
    e = generate_trace(load_config(overrides={"traffic.critical_episodes.rate_pps": 20}), src, seed=5, duration_s=600.0)
    assert e.episodes == a.episodes
    assert per_source_keys(e, "regular") == per_source_keys(a, "regular")
    assert per_source_keys(e, "noncritical") == per_source_keys(a, "noncritical")


def test_load_multiplier_scales_base_rates_only(cfg):
    """§5.6: load_multiplier = 2 temel hızları iki katına çıkarır; epizot pencereleri ve epizot hızı değişmez."""
    src = sources(10)
    cfg2 = load_config(overrides={"traffic.load_multiplier": 2.0})
    one = generate_trace(cfg, src, seed=2, duration_s=20_000.0)
    two = generate_trace(cfg2, src, seed=2, duration_s=20_000.0)
    assert two.load_multiplier == 2.0 and two.episodes == one.episodes
    assert expected_rate_pps(cfg2.traffic, "regular") == 2.0 and math.isclose(expected_rate_pps(cfg2.traffic, "noncritical"), 0.4)
    p_on = episode_on_fraction(cfg2.traffic.critical_episodes)
    assert math.isclose(expected_rate_pps(cfg2.traffic, "critical"), 2.0 * (1 - p_on) + 10 * p_on)
    ratio_regular = two.mean_rate_pps("regular") / one.mean_rate_pps("regular")
    ratio_non = two.mean_rate_pps("noncritical") / one.mean_rate_pps("noncritical")
    assert abs(ratio_regular - 2.0) < 0.02 and abs(ratio_non - 2.0) < 0.10
    assert abs(two.episode_rate_pps("critical", inside=False) / one.episode_rate_pps("critical", inside=False) - 2.0) < 0.10
    assert abs(two.episode_rate_pps("critical", inside=True) - 10) / 10 < 0.05
    assert abs(one.episode_rate_pps("critical", inside=True) - 10) / 10 < 0.05
    for s in one.sources_by_class["regular"]:  # düzenli: periyot 0,5 s ± %10
        gaps = np.diff([p.created_s for p in two.by_source[s]])
        assert gaps.min() >= 0.45 - 1e-9 and gaps.max() <= 0.55 + 1e-9


# ----------------------------------------------------------------------------- paket kaydı (§5.6)


def test_packet_record_fields(cfg):
    """§5.6: id'ler zaman sırasında, boyutlar dağılımda, son tarih = oluşturulma + π3; dizinler tutarlı."""
    trace = generate_trace(cfg, sources(3), seed=9, duration_s=300.0)
    pi3 = {cls: cfg.policy.for_topology(cfg.traffic.classes[cls].topology).max_delay_ms / 1000.0 for cls in CLASSES}
    assert pi3 == {"critical": 0.025, "regular": 0.1, "noncritical": 0.5}
    ids_by_source = {s: {q.packet_id for q in v} for s, v in trace.by_source.items()}
    prev = -1.0
    for i, p in enumerate(trace.packets):
        assert p.packet_id == i and p.created_s >= prev and 0.0 <= p.created_s < 300.0
        prev = p.created_s
        assert math.isclose(p.deadline_s, p.created_s + pi3[p.traffic_class])
        assert p.packet_id in ids_by_source[p.source]
    sizes = {cls: {p.size_bytes for p in trace.packets_of_class(cls)} for cls in CLASSES}
    assert sizes["critical"] == {64}
    assert min(sizes["regular"]) >= 128 and max(sizes["regular"]) <= 256 and len(sizes["regular"]) > 50
    assert min(sizes["noncritical"]) >= 512 and max(sizes["noncritical"]) <= 1024
    assert sum(len(v) for v in trace.by_source.values()) == len(trace.packets)
    df = trace.to_dataframe()
    assert len(df) == len(trace.packets) and set(df.columns) >= {"packet_id", "source", "traffic_class", "size_bytes", "created_s", "deadline_s"}
    rec = trace.to_record()
    assert rec["n_packets_total"] == len(trace.packets) and rec["offered_kbps_noncritical"] > 0


def test_build_from_network_uses_sensor_classes(cfg):
    """§5.6 kanonik giriş: izdeki kaynaklar ağın sensörleri, sınıfları düğümün ürettiği sınıf; süre varsayılanı simulation.duration_s."""
    pn = build_partitioned_network(cfg, seed=1)
    trace = build_traffic_trace(cfg, pn.net, seed=1)
    assert trace.duration_s == cfg.simulation.duration_s == 600.0
    for cls in CLASSES:
        assert sorted(trace.sources_by_class[cls]) == pn.net.sensors_of_class(cls)
    assert set(trace.by_source) == set(pn.net.sensor_ids)
    assert set(trace.episodes) <= set(pn.net.sensors_of_class("critical"))
    assert build_traffic_trace(cfg, pn.net, seed=1).packets == trace.packets

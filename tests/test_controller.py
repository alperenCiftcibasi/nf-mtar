"""Testler: engine/rank.py + controller.py (minimal M9) ve B3'ün simülasyona bağlanması (CLAUDE.md §5.17, §5.18)."""

from __future__ import annotations

import math

import numpy as np

from nfmtar.config import load_config
from nfmtar.engine.rank import rank_candidates, rank_values
from nfmtar.simulation import run_simulation


def test_rank_formula_hand_computed():
    """§5.17 R = w1(1 − f) + w2·H/Hmax + w3·C/Cmax; argmin, eşitlikte küçük id."""
    R = rank_values([0.9, 0.5], [1, 2], [0.5, 1.0], (0.4, 0.4, 0.2))
    assert np.allclose(R, [0.4 * 0.1 + 0.4 * 0.5 + 0.2 * 0.5, 0.4 * 0.5 + 0.4 * 1.0 + 0.2 * 1.0])
    rr = rank_candidates([7, 3], [0.5, 0.5], [0.1, 0.1], [1, 1], [1.0, 1.0], (0.4, 0.4, 0.2))
    assert rr.candidates == (3, 7) and rr.best == 3                     # eşitlikte küçük id


def test_b3_smoke_root_health_trigger_and_hysteresis():
    """B3 (uzman bulanık, dinamik root) simülasyonda koşar; root öldürülünce sağlık tetikleyicisiyle hemen yeni root;
    histerezis: ilk 30 s'de periyodik değişimler min_dwell (2·π11) ve ε kuralına uyar."""
    cfg = load_config(overrides={"simulation.duration_s": 30.0, "run.warm_up_s": 5.0, "traffic.load_multiplier": 0.25,
                                 "network.n_nodes": 60})
    res = run_simulation(cfg, "B3", seed=2, kill_roots_at_s=15.0)
    deaths = {u for _, u in res.deaths}
    health = [rc for rc in res.root_changes if rc["reason"].startswith("death:")]
    assert len(health) == 3 and all(rc["old"] in deaths and rc["new"] not in deaths for rc in health)
    assert all(math.isclose(rc["time_s"], 15.0) for rc in health)
    assert sum(ps.delivered for ps in res.packets if ps.packet.created_s > 15.0) > 0   # kurtarma
    # periyodik değişimler arasında en az min_dwell
    for t in ("E", "L", "R"):
        times = [rc["time_s"] for rc in res.root_changes if rc["topology"] == t and rc["reason"] == "periodic"]
        dwell = 2 * cfg.policy.for_topology(t).route_update_s
        assert all(b - a >= dwell - 1e-9 for a, b in zip(times, times[1:]))


# --------------------------------------------------------------------------- #
# §5.18 tetikleyici 2: trafik değişimi (TrafficMonitor)
# --------------------------------------------------------------------------- #

from nfmtar.controller import TrafficMonitor  # noqa: E402
from nfmtar.network.partition import build_partitioned_network  # noqa: E402
from nfmtar.traffic import build_traffic_trace  # noqa: E402


def _feed(monitor: TrafficMonitor, packets, t_end: float, step: float = 1.0) -> list[tuple[float, str]]:
    """İzi oluşturulma sırasıyla bildirir, her `step`'te kontrol eder (simülasyondaki sırayla aynı)."""
    fires, i, k = [], 0, 1
    pk = sorted(packets, key=lambda p: p.created_s)
    while k * step <= t_end + 1e-9:
        t = k * step
        while i < len(pk) and pk[i].created_s <= t:
            monitor.observe(pk[i].traffic_class, pk[i].created_s)
            i += 1
        fires += [(t, c) for c in monitor.check(t)]
        k += 1
    return fires


def test_traffic_monitor_hand_example():
    """Eşit aralıklı 5 paket/s, t = 100'de 15 paket/s. t = 100 + k'de n_önce = 50, n_şimdi = 50 + 10k.
    Saf formül (z = 0): 10k/50 > 0,3 ilk k = 2'de -> t = 102. z = 3: ayrıca 10k > 3·sqrt(100 + 10k) gerekir;
    k = 2: 20 < 32,9, k = 3: 30 < 34,2, k = 4: 40 > 35,5 -> t = 104. Öncesinde tetiklenmez; λ değerleri raporlanır."""
    times = [i * 0.2 for i in range(1, 500)] + [100 + i / 15 for i in range(1, 600)]
    for z, expected in ((0.0, 102), (3.0, 104)):
        m = TrafficMonitor(["c"], delta_t_s=10.0, tau=0.3, eps0_pps=0.01, cooldown_s=10.0, significance_z=z)
        fires = []
        for k in range(1, 121):
            for x in [x for x in times if k - 1 < x <= k]:
                m.observe("c", x)
            fires += [k for _ in m.check(float(k))]
        assert fires[0] == expected
        ev = m.events[0]
        assert math.isclose(ev["lambda_prev_pps"], 5.0) and ev["relative_change"] > 0.3
    assert m.rates("c", 120.0) == (15.0, 15.0)


def test_traffic_trigger_fires_on_episode_start_not_on_stationary_traffic():
    """Gerçek izler (60 düğüm, yük 0,25, 600 s, seed 1–3). Durağan trafikte (epizotlar kapalı) kritik ve düzenli
    sınıf hiç tetiklenmez, kritik olmayan koşu başına en çok 2 (3σ gürültü); z = 0 (saf formül) ise onu sürekli
    tetikler. Epizotlar açıkken, [s − 2Δt, s + Δt] aralığında başka epizot sınırı (başlangıç/bitiş) olmayan her
    epizot başlangıcı s'den sonraki Δt + 1 s içinde kritik sınıf tetiklenir (çakışan bir epizot bitişi artışı
    dengeleyebildiği için yalnızca yalıtılmış başlangıçlar sınanır)."""
    for seed in (1, 2, 3):
        base = {"traffic.load_multiplier": 0.25, "network.n_nodes": 60}
        cfg = load_config(overrides={**base, "traffic.critical_episodes.enabled": False})
        tr = build_traffic_trace(cfg, build_partitioned_network(cfg, seed).net, seed, 600.0)
        fires = _feed(TrafficMonitor.from_config(cfg, tr.classes), tr.packets, 600.0)
        assert not [f for f in fires if f[1] != "noncritical"] and len(fires) <= 2
        raw = TrafficMonitor.from_config(cfg, tr.classes)
        raw.z = 0.0
        assert sum(1 for _, c in _feed(raw, tr.packets, 600.0) if c == "noncritical") >= 10

        cfg = load_config(overrides=base)
        tr = build_traffic_trace(cfg, build_partitioned_network(cfg, seed).net, seed, 600.0)
        m = TrafficMonitor.from_config(cfg, tr.classes)
        crit = [t for t, c in _feed(m, tr.packets, 600.0) if c == "critical"]
        edges = sorted(x for ws in tr.episodes.values() for w in ws for x in w)
        starts = sorted(w[0] for ws in tr.episodes.values() for w in ws)
        isolated = [s for s in starts if 2 * m.delta_t_s <= s <= 600.0 - m.delta_t_s - 1
                    and not any(s - 2 * m.delta_t_s <= e <= s + m.delta_t_s and e != s for e in edges)]
        assert isolated
        for s in isolated:
            assert any(s <= t <= s + m.delta_t_s + 1.0 for t in crit), (seed, s)


def test_traffic_trigger_in_simulation_only_for_dynamic_root():
    """B3 (dinamik root): epizotlu kısa koşuda trafik tetikleyicisi çalışır ve her olay kritik sınıfa aittir
    (durağan sınıflar tetiklenmez); trafik nedenli root değişimleri yalnızca tetikleme anlarında olur.
    B1 (sabit root) trafiği izlemez."""
    cfg = load_config(overrides={"simulation.duration_s": 120.0, "run.warm_up_s": 5.0, "traffic.load_multiplier": 0.25,
                                 "network.n_nodes": 60})
    res = run_simulation(cfg, "B3", seed=2)
    assert res.traffic_triggers and {e["traffic_class"] for e in res.traffic_triggers} == {"critical"}
    times = {e["time_s"] for e in res.traffic_triggers}
    assert all(rc["time_s"] in times for rc in res.root_changes if rc["reason"].startswith("traffic:"))
    assert run_simulation(cfg, "B1", seed=2).traffic_triggers == []

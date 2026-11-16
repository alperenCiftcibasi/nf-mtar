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

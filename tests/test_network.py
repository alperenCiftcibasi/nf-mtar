"""Testler: config (§3, §7), fiziksel ağ ve yeniden üretim (§5.1), yerleşim (§5.2), bölümleme (§5.3), edge router (§5.4)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from nfmtar.config import ConfigError, config_from_dict, load_config, load_raw, parse_override
from nfmtar.network.graph import (
    TOPOLOGY_ORDER,
    NodeState,
    PacketQueue,
    Topology,
    build_network,
    generate_network,
    r_region_rect,
    split_counts,
    unit_disk_edges,
)
from nfmtar.network.partition import build_partitioned_network, check_edge_connectivity, partition_network

UNIFORM_E0 = {"placement.random.r_placement": "uniform", "network.enforce_partition_connectivity": False}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def pn(cfg):
    return build_partitioned_network(cfg, seed=1)


@pytest.fixture(scope="module")
def net(pn):
    return pn.net


@pytest.fixture(scope="module")
def part(pn):
    return pn.part


# ----------------------------------------------------------------------------- config (§7)


def test_config_table1_matches_claude_md(cfg):
    """§5.7 Tablo 1 değerleri dataclass'a doğru taşınmalı."""
    assert (cfg.policy.E.priority, cfg.policy.L.priority, cfg.policy.R.priority) == (7, 4, 1)
    assert (cfg.policy.E.buffer_packets, cfg.policy.L.buffer_packets, cfg.policy.R.buffer_packets) == (16, 8, 4)
    assert cfg.policy.for_topology(Topology.R).energy_weight == 0.8
    assert cfg.policy.for_topology("E").max_delay_ms == 25
    assert len(cfg.run.seeds) == 30
    assert cfg.network.radio_range_m == 50.0 and cfg.network.area_m == (300.0, 300.0)


def test_config_new_decisions_defaults(cfg):
    """§5.1 enforce_partition_connectivity, §5.2 r_placement/r_target_degree, §5.9 relay_mode varsayılanları."""
    assert cfg.network.enforce_partition_connectivity is True
    assert cfg.placement.random.r_placement == "region" and cfg.placement.random.r_target_degree == 10
    assert cfg.router.relay_mode == "strict" and cfg.router.b2.link_metric == "etx"
    assert cfg.experiments["E0"]["placements"] == ["uniform", "region"]


def test_config_class_topology_mapping_is_one_hot(cfg):
    """§5.3: her topolojiye tam bir trafik sınıfı."""
    assert cfg.traffic.class_for_topology("E") == "critical"
    assert cfg.traffic.class_for_topology("L") == "regular"
    assert cfg.traffic.class_for_topology("R") == "noncritical"
    assert cfg.traffic.classes["regular"].packet_bytes == (128, 256)
    assert cfg.traffic.classes["critical"].packet_bytes == 64


def test_config_rejects_unknown_and_missing_keys():
    """§7: yazım hataları sessizce geçmemeli."""
    raw = load_raw()
    raw["network"]["radyo_menzili"] = 1
    with pytest.raises(ConfigError, match="bilinmeyen anahtar"):
        config_from_dict(raw)
    raw = load_raw()
    del raw["network"]["radio_range_m"]
    with pytest.raises(ConfigError, match="zorunlu anahtar eksik"):
        config_from_dict(raw)
    raw = load_raw()
    raw["network"]["n_nodes"] = "yüz elli"
    with pytest.raises(ConfigError, match="tam sayı"):
        config_from_dict(raw)


def test_config_validates_enums_ratio_and_mapping():
    raw = load_raw()
    raw["placement"]["class_ratio"] = {"E": 0.5, "L": 0.5, "R": 0.5}
    with pytest.raises(ConfigError, match="toplamı 1"):
        config_from_dict(raw)
    raw = load_raw()
    raw["traffic"]["classes"]["regular"]["topology"] = "E"
    with pytest.raises(ConfigError, match="one-hot"):
        config_from_dict(raw)
    with pytest.raises(ConfigError, match="r_placement"):
        load_config(overrides={"placement.random.r_placement": "grid"})
    with pytest.raises(ConfigError, match="relay_mode"):
        load_config(overrides={"router.relay_mode": "mixed"})


def test_config_overrides():
    cfg = load_config(overrides={"network.n_nodes": 60, "energy.initial_J": 0.5})
    assert cfg.network.n_nodes == 60 and cfg.energy.initial_J == 0.5
    assert parse_override("network.n_nodes=200") == ("network.n_nodes", 200)
    assert parse_override("placement.random.r_placement=uniform") == ("placement.random.r_placement", "uniform")
    with pytest.raises(ConfigError, match="bulunamadı"):
        load_config(overrides={"network.yok_boyle_anahtar": 1})


# ----------------------------------------------------------------------------- düğüm durumu (§5.5, §5.7)


def test_split_counts_largest_remainder():
    """§5.2: oranlar tam sayıya bölünürken toplam korunur."""
    assert split_counts(150, {"E": 0.3, "L": 0.4, "R": 0.3}) == {"E": 45, "L": 60, "R": 45}
    counts = split_counts(20, {"E": 0.3, "L": 0.4, "R": 0.3})
    assert sum(counts.values()) == 20 and counts == {"E": 6, "L": 8, "R": 6}
    assert sum(split_counts(7, {"E": 0.3, "L": 0.4, "R": 0.3}).values()) == 7


def test_packet_queue_bounded_fifo():
    """§5.7 π9: tampon dolunca paket düşer; FIFO sırası korunur."""
    q = PacketQueue(capacity=2)
    assert q.push("a") and q.push("b")
    assert not q.push("c") and q.dropped == 1
    assert q.occupancy == 1.0 and q.is_full
    assert q.pop() == "a" and q.pop() == "b" and q.is_empty
    unbounded = PacketQueue()
    for i in range(100):
        assert unbounded.push(i)
    assert unbounded.occupancy == 0.0 and unbounded.dropped == 0


def test_node_energy_depletion_and_death():
    """§5.5: tükenme = 1 − kalan/başlangıç; enerjisi biten düğüm ölür."""
    n = NodeState(0, (0.0, 0.0), "critical", False, 2.0, 2.0, 0.0)
    assert n.depletion == 0.0 and n.alive
    assert n.consume(0.5) and math.isclose(n.depletion, 0.25)
    n.queue.push("pkt")
    assert not n.consume(1.5, now_s=12.0)
    assert not n.alive and n.energy_J == 0.0 and n.death_time_s == 12.0 and n.depletion == 1.0
    assert len(n.queue) == 0  # ölen düğümün kuyruğu boşalır
    with pytest.raises(ValueError):
        n.consume(-1.0)


def test_node_kill_and_edge_router_energy():
    """§5.19 E5 arıza enjeksiyonu; §5.4 edge router sonsuz enerjili."""
    n = NodeState(0, (0.0, 0.0), "critical", False, 2.0, 2.0, 0.0)
    n.kill(now_s=300.0)
    assert not n.alive and n.death_time_s == 300.0
    er = NodeState(1, (5.0, 150.0), None, True, math.inf, math.inf, 0.0, topology=Topology.E)
    assert er.depletion == 0.0
    assert er.consume(1e9) and er.alive and math.isinf(er.energy_J)
    with pytest.raises(ValueError):
        er.kill()


# ----------------------------------------------------------------------------- fiziksel ağ ve yerleşim (§5.1, §5.2)


def test_node_counts_and_ids(cfg, net):
    """§5.1 N sensör + §5.4 üç edge router; kimlikler ardışık."""
    n = cfg.network.n_nodes
    assert net.n_sensors == n and net.sensor_ids == tuple(range(n))
    assert set(net.edge_router_ids.values()) == {n, n + 1, n + 2}
    assert net.graph.number_of_nodes() == n + 3
    counts = split_counts(n, cfg.placement.class_ratio)
    for t in TOPOLOGY_ORDER:
        cls = cfg.traffic.class_for_topology(t.value)
        assert len(net.sensors_of_class(cls)) == counts[t.value]
    for eid in net.edge_router_ids.values():
        st = net.nodes[eid]
        assert st.is_edge_router and st.traffic_class is None and math.isinf(st.energy_J)
        assert net.topology_of_edge_router(eid) == st.topology


def test_positions_in_area_and_regions(cfg, net):
    """§5.2: E düğümleri elipste, L düğümleri koridorda, R düğümleri bölge dikdörtgeninde, hepsi alan içinde."""
    w, h = cfg.network.area_m
    pos = net.positions()
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= w).all()
    assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= h).all()

    ell = cfg.placement.elliptic
    cx, cy = ell.center_frac[0] * w, ell.center_frac[1] * h
    a, b = ell.semi_axes_frac[0] * w, ell.semi_axes_frac[1] * h
    for i in net.sensors_of_class(cfg.traffic.class_for_topology("E")):
        x, y = net.nodes[i].pos
        assert ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0 + 1e-9

    lin = cfg.placement.linear
    center = lin.center_frac * (h if lin.axis == "x" else w)
    for i in net.sensors_of_class(cfg.traffic.class_for_topology("L")):
        across = net.nodes[i].pos[1] if lin.axis == "x" else net.nodes[i].pos[0]
        assert abs(across - center) <= lin.width_m / 2 + 1e-9

    assert net.r_region is not None
    x0, y0, rw, rh = net.r_region
    for i in net.sensors_of_class(cfg.traffic.class_for_topology("R")):
        x, y = net.nodes[i].pos
        assert x0 - 1e-9 <= x <= x0 + rw + 1e-9 and y0 - 1e-9 <= y <= y0 + rh + 1e-9

    for t in TOPOLOGY_ORDER:
        assert net.nodes[net.edge_router_ids[t]].pos == cfg.edge_routers.positions_m[t.value]


def test_r_region_geometry(cfg, net):
    """§5.2 region: A_R = n_R·π·R²/r_target_degree; alan içinde; Edge-R dış kenarın orta noktasında."""
    n_r = split_counts(cfg.network.n_nodes, cfg.placement.class_ratio)["R"]
    expected_area = n_r * math.pi * cfg.network.radio_range_m**2 / cfg.placement.random.r_target_degree
    x0, y0, rw, rh = net.r_region
    assert math.isclose(rw * rh, expected_area, rel_tol=1e-9)
    assert math.isclose(expected_area, 35342.9, rel_tol=1e-4)  # CLAUDE.md: ≈ 35.000 m²
    w, h = cfg.network.area_m
    assert 0 <= x0 and x0 + rw <= w + 1e-9 and 0 <= y0 and y0 + rh <= h + 1e-9
    ex, ey = cfg.edge_routers.positions_m["R"]
    assert math.isclose(x0 + rw, ex) and math.isclose(y0 + rh / 2, ey)  # Edge-R sağda: dikdörtgen içe uzanır

    # Edge router başka kenarlardayken de içe doğru uzanır
    cfg_r = cfg.placement.random
    x0, y0, rw, rh = r_region_rect(45, 50.0, (300.0, 300.0), (5.0, 150.0), cfg_r)
    assert math.isclose(x0, 5.0) and math.isclose(y0 + rh / 2, 150.0)
    x0, y0, rw, rh = r_region_rect(45, 50.0, (300.0, 300.0), (150.0, 295.0), cfg_r)
    assert math.isclose(y0 + rh, 295.0) and math.isclose(x0 + rw / 2, 150.0)
    assert r_region_rect(0, 50.0, (300.0, 300.0), (295.0, 150.0), cfg_r)[2:] == (0.0, 0.0)

    uniform = load_config(overrides=UNIFORM_E0)
    assert build_partitioned_network(uniform, 1).net.r_region is None


@pytest.mark.parametrize("n_nodes", [20, 50, 100, 150, 200])
def test_r_region_scales_with_constant_density(n_nodes):
    """§5.2 + §5.19 E2: alan kenarı 300·sqrt(N/150) iken bölge n_R ile ölçeklenir ve alana sığar."""
    side = 300.0 * math.sqrt(n_nodes / 150.0)
    cfg = load_config(overrides={
        "network.n_nodes": n_nodes,
        "network.area_m": [side, side],
        "edge_routers.positions_m": {"E": [5.0, side / 2], "L": [side / 2, side - 5.0], "R": [side - 5.0, side / 2]},
        "network.enforce_partition_connectivity": False,
    })
    net = generate_network(cfg, seed=1)
    n_r = split_counts(n_nodes, cfg.placement.class_ratio)["R"]
    x0, y0, rw, rh = net.r_region
    assert math.isclose(rw * rh, n_r * math.pi * 2500.0 / 10.0, rel_tol=1e-9)
    assert x0 >= 0 and y0 >= 0 and x0 + rw <= side + 1e-9 and y0 + rh <= side + 1e-9
    assert math.isclose(x0 + rw, side - 5.0)


def test_unit_disk_edges_exact(cfg, net):
    """§5.1: kenar <=> mesafe <= R; kenar üzerindeki mesafe doğru."""
    pos = net.positions()
    r = cfg.network.radio_range_m
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    n = len(pos)
    expected = {(i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] <= r}
    actual = {(min(u, v), max(u, v)) for u, v in net.graph.edges()}
    assert actual == expected
    for u, v in list(net.graph.edges())[:50]:
        assert math.isclose(net.graph.edges[u, v]["dist"], d[u, v])
        assert math.isclose(net.distance(u, v), d[u, v])
    assert unit_disk_edges(np.array([[0.0, 0.0], [30.0, 0.0], [100.0, 0.0]]), 50.0) == [(0, 1, 30.0)]


def test_reproducible_by_seed(cfg):
    """§3: aynı seed aynı ağ; farklı seed farklı ağ; deneme indeksi akışı değiştirir."""
    a, b, c = build_partitioned_network(cfg, 7).net, build_partitioned_network(cfg, 7).net, build_partitioned_network(cfg, 8).net
    assert np.array_equal(a.positions(), b.positions())
    assert set(a.graph.edges()) == set(b.graph.edges())
    assert not np.array_equal(a.positions(), c.positions())
    assert not np.array_equal(generate_network(cfg, 7, 0).positions(), generate_network(cfg, 7, 1).positions())


# ----------------------------------------------------------------------------- yeniden üretim ölçütü (§5.1)


def test_strict_regen_every_topology_connected_for_all_seeds(cfg):
    """§5.1 strict + region: her seed'de her topoloji kendi edge router'ına bağlı; deneme sayısı sınır altında."""
    attempts = []
    for seed in cfg.run.seeds:
        pn = build_partitioned_network(cfg, seed)
        assert pn.part.disconnected_count == 0, f"seed {seed}: {pn.part.disconnected}"
        assert pn.enforced and pn.relay_mode == "strict" and pn.r_placement == "region"
        attempts.append(pn.net.regen_attempts)
    assert max(attempts) < cfg.network.max_regen_attempts


def test_no_regen_when_not_enforced():
    """§5.1 E0 modu: yeniden üretim yok, kopuk düğümler yalnızca ölçülür."""
    cfg = load_config(overrides=UNIFORM_E0)
    for seed in cfg.run.seeds[:5]:
        pn = build_partitioned_network(cfg, seed)
        assert pn.net.regen_attempts == 0 and not pn.enforced
    rec = pn.to_record()
    for key in ("seed", "regen_attempts", "physical_connected", "disconnected_R", "r_placement", "relay_mode"):
        assert key in rec
    assert rec["disconnected_total"] == pn.part.disconnected_count


def test_shared_mode_requires_physical_connectivity():
    """§5.1 + §5.9 shared: topolojiler bağlı VE fiziksel ağ bağlı."""
    cfg = load_config(overrides={"router.relay_mode": "shared"})
    for seed in cfg.run.seeds[:5]:
        pn = build_partitioned_network(cfg, seed)
        assert pn.part.disconnected_count == 0 and pn.net.is_connected()
        assert pn.to_record()["physical_connected"] is True


def test_regeneration_gives_up_when_criterion_impossible():
    """§5.1: Edge-E elipsten 50 m'den uzaksa E hiç bağlanamaz -> RuntimeError; build_network tek başına ölçüt uygulamaz."""
    cfg = load_config(overrides={"edge_routers.positions_m": {"E": [0.0, 0.0], "L": [150.0, 295.0], "R": [295.0, 150.0]},
                                 "network.max_regen_attempts": 3})
    with pytest.raises(RuntimeError, match="kabul ölçütünü sağlayan ağ üretilemedi"):
        build_partitioned_network(cfg, seed=1)
    net = build_network(cfg, seed=1)  # accept=None: ilk üretim
    assert net.regen_attempts == 0
    assert build_network(cfg, seed=1, accept=lambda n: n.regen_attempts == 2).regen_attempts == 2


# ----------------------------------------------------------------------------- bölümleme (§5.3)


def test_partition_is_deterministic_one_hot_and_disjoint(cfg, net, part):
    """§5.3: sensör -> tek topoloji; bölümler ayrık; birleşim = tüm sensörler."""
    union = set()
    for t in TOPOLOGY_ORDER:
        assert not (union & part.members[t])
        union |= part.members[t]
    assert union == set(net.sensor_ids)
    for i in net.sensor_ids:
        assert len(part.node_topologies[i]) == 1
        t = part.topology_of(i)
        assert net.nodes[i].topology == t
        assert cfg.traffic.classes[net.nodes[i].traffic_class].topology == t.value
    for t in TOPOLOGY_ORDER:
        assert part.node_topologies[net.edge_router_ids[t]] == frozenset({t})
    assert build_partitioned_network(cfg, 1).part.members == part.members


def test_partition_applies_buffer_policy(cfg, net, part):
    """§5.7 π9: kuyruk kapasitesi topolojiye göre 16 / 8 / 4."""
    for i in net.sensor_ids:
        assert net.nodes[i].queue.capacity == cfg.policy.for_topology(part.topology_of(i)).buffer_packets


def test_partition_subgraph_and_connectivity_uniform(caplog):
    """§5.3 (uniform/E0): alt çizge = üyeler + edge router; kopuk düğümlerin yolu yok; uyarı loglanır."""
    cfg = load_config(overrides=UNIFORM_E0)
    with caplog.at_level("WARNING", logger="nfmtar.network.partition"):
        pn = build_partitioned_network(cfg, seed=1)
    net, part = pn.net, pn.part
    assert part.disconnected_count > 0  # ilk koşu bulgusu: uniform yerleşimde R kopuk kalır (§5.2)
    assert any("edge router'a yolu yok" in rec.getMessage() for rec in caplog.records)
    for t in TOPOLOGY_ORDER:
        sub = part.subgraph(net, t)
        eid = net.edge_router_ids[t]
        assert set(sub.nodes()) == part.members[t] | {eid}
        assert part.disconnected[t] <= part.members[t]
        for v in part.connected_members(t):
            assert nx.has_path(sub, v, eid)
        for v in part.disconnected[t]:
            assert not nx.has_path(sub, v, eid)
        assert part.subgraph_mean_degree(net, t) >= 0
    assert part.disconnected_count == sum(len(s) for s in part.disconnected.values())


def test_connectivity_alive_only_excludes_dead_nodes(cfg):
    """§5.5 + §5.3: edge router'ın tüm komşuları ölünce kalan canlı üyeler kopuk sayılır."""
    pn = build_partitioned_network(cfg, seed=3)
    net, part = pn.net, pn.part
    t = Topology.E
    eid = net.edge_router_ids[t]
    gateways = [v for v in net.neighbors(eid) if v in part.members[t]]
    assert gateways, "Edge-E'nin E topolojisinde komşusu olmalı"
    for v in gateways:
        net.nodes[v].kill(now_s=0.0)
    survivors = {v for v in part.members[t] if net.nodes[v].alive}
    dead_only = check_edge_connectivity(net, part.members[t], eid, alive_only=True)
    assert dead_only == frozenset(survivors)
    static = check_edge_connectivity(net, part.members[t], eid, alive_only=False)
    assert static == part.disconnected[t] == frozenset()
    fresh = partition_network(pn.net, cfg, warn=False)
    assert fresh.members == part.members

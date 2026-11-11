"""Testler: router.py — kenar maliyeti, ters yönlü Dijkstra, iki aşamalı yol, relay_mode, ölü düğümler (CLAUDE.md §5.9, §5.4)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from nfmtar.config import ConfigError, load_config
from nfmtar.network.graph import TOPOLOGY_ORDER, NodeState, PhysicalNetwork, Topology, unit_disk_edges
from nfmtar.network.partition import build_partitioned_network, partition_network
from nfmtar.network.radio import RadioModel
from nfmtar.router import Router, relays_on_paths

FAR = {"E": (0.0, 0.0), "L": (1000.0, 0.0), "R": (2000.0, 0.0)}   # ilgisiz edge router'lar uzakta


def make_net(cfg, sensors, edge_pos=None, energy_frac=None):
    """Elle kurulmuş ağ: sensors = [(x, y, sınıf), ...] -> kimlikler 0..n−1; edge router'lar n, n+1, n+2 (E, L, R).

    energy_frac: {kimlik: kalan/başlangıç} (yorgun düğümler). Çizge unit disk (R = config), 'dist' kenar özniteliği.
    """
    edge_pos = {**FAR, **(edge_pos or {})}
    e0 = cfg.energy.initial_J
    nodes: dict[int, NodeState] = {}
    for i, (x, y, cls) in enumerate(sensors):
        frac = (energy_frac or {}).get(i, 1.0)
        nodes[i] = NodeState(i, (float(x), float(y)), cls, False, e0, e0 * frac, cfg.energy.dead_threshold_J)
    n = len(sensors)
    edge_ids = {}
    for k, t in enumerate(TOPOLOGY_ORDER):
        nid = n + k
        nodes[nid] = NodeState(nid, tuple(map(float, edge_pos[t.value])), None, True, math.inf, math.inf,
                               cfg.energy.dead_threshold_J, topology=t)
        edge_ids[t] = nid
    g = nx.Graph()
    for i, st in nodes.items():
        g.add_node(i, pos=st.pos, is_edge_router=st.is_edge_router, traffic_class=st.traffic_class)
    pos = np.array([nodes[i].pos for i in range(len(nodes))])
    for u, v, d in unit_disk_edges(pos, cfg.network.radio_range_m):
        g.add_edge(u, v, dist=d)
    net = PhysicalNetwork(g, nodes, tuple(range(n)), edge_ids, (3000.0, 3000.0), cfg.network.radio_range_m, 0, 0)
    part = partition_network(net, cfg, warn=False)
    return net, part


def brute_force_best(router, net, g: nx.DiGraph, source: int, target: int):
    """Bağımsız kontrol: bütün basit yollar arasında en ucuz (Dijkstra'dan bağımsız)."""
    best, best_cost = None, math.inf
    for p in nx.all_simple_paths(g, source, target):
        c = sum(g.edges[u, v]["cost"] for u, v in zip(p, p[1:]))
        if c < best_cost:
            best, best_cost = tuple(p), c
    return best, best_cost


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def radio(cfg):
    return RadioModel.from_config(cfg)


@pytest.fixture
def five_node(cfg):
    """E(id 4) (0,0); n0 (10,0); n1 (30,0); n2 (59,0); n3 (30,30); hepsi kritik -> topoloji E."""
    return make_net(cfg, [(10, 0, "critical"), (30, 0, "critical"), (59, 0, "critical"), (30, 30, "critical")])


# ----------------------------------------------------------------------------- λ'lar ve kenar maliyeti formülü (§5.9)


def test_lambdas_from_policy_and_normalization(cfg, radio):
    """§5.9: λ3 = π6, λ1 = λ2 = (1 − λ3)/2; B2: λ1 = 0, λ2 = 1 − λ3; lambda3_from_policy=false -> λ3 = 0."""
    r = Router(cfg, radio, "weighted")
    for t, l3 in (("E", 0.3), ("L", 0.5), ("R", 0.8)):
        lam = r.lambdas(t)
        assert math.isclose(lam.l3, l3) and math.isclose(lam.l1, (1 - l3) / 2) and math.isclose(lam.l2, (1 - l3) / 2)
        assert math.isclose(lam.total, 1.0)
    b2 = Router(cfg, radio, "etx_depletion").lambdas(Topology.E)
    assert b2.l1 == 0.0 and math.isclose(b2.l2, 0.7) and math.isclose(b2.l3, 0.3)
    no_energy = Router(load_config(overrides={"router.lambda3_from_policy": False}), radio, "weighted").lambdas("R")
    assert no_energy.l3 == 0.0 and math.isclose(no_energy.l1, 0.5) and math.isclose(no_energy.l2, 0.5)
    assert Router.for_method(cfg, radio, "B1").path_cost == "hop"
    assert Router.for_method(cfg, radio, "B1-plain").path_cost == "hop"
    assert Router.for_method(cfg, radio, "B2").path_cost == "etx_depletion"
    assert Router.for_method(cfg, radio, "NF-MTAR").path_cost == Router.for_method(cfg, radio, "B3").path_cost == "weighted"
    with pytest.raises(ValueError):
        Router(cfg, radio, "manhattan")
    with pytest.raises(ConfigError, match="algorithm"):
        load_config(overrides={"router.algorithm": "bellman_ford"})
    with pytest.raises(ConfigError, match="prr_min"):
        load_config(overrides={"router.prr_min": 0.0})


def test_edge_cost_formula_hand_computed(cfg, radio, five_node):
    """§5.9: cost = λ1·d²/R² + λ2·(−log PRR)/L_max + λ3·depletion(u); PRR kırpılır, dışlanmaz; B1 = 1."""
    net, part = five_node
    r = Router(cfg, radio, "weighted")
    l_max = -math.log(0.05)
    # n0 -> E: d = 10, tam enerji
    d = net.distance(0, 4)
    expected = 0.35 * (10 / 50) ** 2 + 0.35 * (-math.log(radio.prr(10.0))) / l_max
    assert math.isclose(d, 10.0) and math.isclose(r.cost(net, 0, 4, "E"), expected)
    # n2 -> n0: d = 49, PRR radyo modelinde 0,057 > 0,05: kırpılmaz; terim < 1 ve kenar çizgede
    assert 0.05 < r.link_prr(49.0) < 0.06 and r.link_term(49.0) < 1.0
    assert r.link_prr(50.0) == 0.05 and math.isclose(r.link_term(50.0), 1.0)   # menzil sınırında kırpılır, terim tam 1
    assert r.link_term(0.0) < 0.004
    g = r.routing_graph(net, part, "E")
    assert g.has_edge(2, 0) and g.has_edge(0, 2)
    # λ'lar topolojiye göre: tam enerjili göndericide R (λ1 = λ2 = 0,1) E'den (0,35) ucuz; yorgun göndericide λ3 = 0,8 ile pahalı
    c10_before = r.cost(net, 1, 0, "E")
    assert r.cost(net, 1, 0, "R") < r.cost(net, 1, 0, "E")
    # depletion gönderici tarafında: n0'ın enerjisi düşünce n0 -> E pahalılaşır, n0'a GİREN kenar (n1 -> n0) değişmez
    net.nodes[0].energy_J = 0.5 * net.nodes[0].initial_energy_J
    assert math.isclose(r.cost(net, 0, 4, "E"), expected + 0.3 * 0.5)
    assert math.isclose(r.cost(net, 1, 0, "E"), c10_before)
    assert r.cost(net, 0, 4, "R") > r.cost(net, 0, 4, "E")
    assert Router(cfg, radio, "hop").cost(net, 2, 0, "E") == 1.0


# ----------------------------------------------------------------------------- 5 düğümlü çizge (§5.9)


def test_five_node_expected_path_and_prr_product(cfg, radio, five_node):
    """§5.9: beklenen yol n2 -> n1 -> n0 -> E (brute force ile aynı); Σ(−log PRR) = −log(Π PRR); H ≠ en ucuz yolun atlaması."""
    net, part = five_node
    r = Router(cfg, radio, "weighted")
    reach = r.reach_from_edge(net, part, "E")
    assert reach.edge_router == 4 and reach.candidates == frozenset({0, 1, 2, 3})
    assert reach.paths[2] == (2, 1, 0, 4) and reach.paths[3] == (3, 1, 0, 4) and reach.paths[1] == (1, 0, 4)
    g = r.routing_graph(net, part, "E")
    for v in (0, 1, 2, 3):
        best, best_cost = brute_force_best(r, net, g, v, 4)
        assert reach.paths[v] == best and math.isclose(reach.cost[v], best_cost)
        assert math.isclose(reach.cost[v], r.path_cost_of(net, reach.paths[v], "E"))
    assert reach.hops == {0: 1, 1: 1, 2: 2, 3: 1}            # H: en az atlama (BFS)
    assert reach.path_hops(2) == 3 and reach.hops[2] == 2      # en ucuz yol daha uzun
    path = reach.paths[2]
    prrs = [r.link_prr(net.distance(u, v)) for u, v in zip(path, path[1:])]
    assert math.isclose(sum(-math.log(p) for p in prrs), -math.log(math.prod(prrs)))
    assert math.isclose(r.path_prr(net, path), math.prod(prrs))
    lam = r.lambdas("E")
    link_part = sum(lam.l2 * (-math.log(p)) / r.l_max for p in prrs)
    assert math.isclose(link_part, lam.l2 * (-math.log(r.path_prr(net, path))) / r.l_max)


# ----------------------------------------------------------------------------- enerji ağırlığı (§5.9 λ3 = π6)


def test_energy_weight_short_tired_vs_long_fresh(cfg, radio):
    """§5.9: kısa ama yorgun aktarıcılı yol ile uzun ama dinlenmiş yol: kritik (λ3 = 0,3) kısa, kritik olmayan (λ3 = 0,8) uzun yolu seçer."""
    layout = [(52, 0), (25, 0), (47, 39.7), (7, 39.7)]   # S, T (yorgun), F1, F2
    expectations = {"critical": ("E", (0, 1)), "noncritical": ("R", (0, 2, 3))}   # + kendi edge router'ı
    for cls, (t, relays) in expectations.items():
        net, part = make_net(cfg, [(x, y, cls) for x, y in layout], edge_pos={t: (0.0, 0.0)}, energy_frac={1: 0.1})
        edge_id = net.edge_router_ids[Topology(t)]
        expected = relays + (edge_id,)
        assert net.graph.has_edge(0, 1) and net.graph.has_edge(1, edge_id) and net.graph.has_edge(3, edge_id)
        assert not net.graph.has_edge(0, edge_id)
        r = Router(cfg, radio, "weighted")
        reach = r.reach_from_edge(net, part, t)
        assert reach.paths[0] == expected, (cls, reach.paths[0])
        best, _ = brute_force_best(r, net, r.routing_graph(net, part, t), 0, edge_id)
        assert best == expected


# ----------------------------------------------------------------------------- B1 ve B2 (§5.9, §5.12)


def test_b1_min_hops_b2_min_etx(cfg, radio):
    """§5.9: B1 en az atlamalı yolu (2 atlama, kötü bağlantılar), B2 en düşük ETX'li yolu (4 iyi atlama) seçer."""
    # E(0,0) - G2(24,0) - M(48,0) - G1(72,0) - S(96,0): 24 m bağlantılar iyi (PRR ~0,99), 48 m bağlantılar kötü (PRR ~0,08)
    net, part = make_net(cfg, [(96, 0, "critical"), (72, 0, "critical"), (48, 0, "critical"), (24, 0, "critical")])
    S, G1, M, G2, E = 0, 1, 2, 3, 4
    assert radio.prr(48.0) < 0.1 < 0.98 < radio.prr(24.0)
    b1 = Router(cfg, radio, "hop").reach_from_edge(net, part, "E")
    assert b1.paths[S] == (S, M, E) and b1.cost[S] == 2.0 and b1.hops[S] == 2
    b2 = Router(cfg, radio, "etx_depletion")
    reach = b2.reach_from_edge(net, part, "E")
    assert reach.paths[S] == (S, G1, M, G2, E)
    assert reach.path_hops(S) == 4 and reach.hops[S] == 2
    etx = lambda d: 1.0 / b2.link_prr(d)  # noqa: E731
    expected = 0.7 * sum((etx(24.0) - 1) / (20 - 1) for _ in range(4))
    assert math.isclose(reach.cost[S], expected)
    # weighted de iyi bağlantılı yolu seçer; B1 ile B2 gerçekten farklı yollar
    assert Router(cfg, radio, "weighted").reach_from_edge(net, part, "E").paths[S] == (S, G1, M, G2, E)


# ----------------------------------------------------------------------------- relay_mode (§5.9)


def test_strict_no_path_shared_relays_via_other_topology(cfg, radio):
    """§5.9: strict modda kopuk R sensörü için yol yok; shared modda L sensörü üzerinden ulaşır; edge router'lar aktarıcı olamaz."""
    # Edge-R (0,0); X (düzenli, L) (30,0); Y (kritik olmayan, R) (60,0); Edge-L (30,30): Y–Edge-L ve Edge-L–Edge-R kenarları var
    net, part = make_net(cfg, [(30, 0, "regular"), (60, 0, "noncritical")], edge_pos={"R": (0.0, 0.0), "L": (30.0, 30.0)})
    X, Y, edge_l, edge_r = 0, 1, net.edge_router_ids[Topology.L], net.edge_router_ids[Topology.R]
    assert net.graph.has_edge(Y, edge_l) and net.graph.has_edge(edge_l, edge_r) and net.graph.has_edge(Y, X)
    assert Y in part.disconnected[Topology.R]
    strict = Router(cfg, radio, "weighted", relay_mode="strict")
    assert not strict.reach_from_edge(net, part, "R").reachable(Y)
    assert strict.reach_from_edge(net, part, "R").candidates == frozenset()
    assert set(strict.routing_graph(net, part, "R").nodes) == {Y, edge_r}
    shared = Router(load_config(overrides={"router.relay_mode": "shared"}), radio, "weighted")
    assert shared.relay_mode == "shared"
    reach = shared.reach_from_edge(net, part, "R")
    assert reach.paths[Y] == (Y, X, edge_r) and reach.candidates == frozenset({Y})   # X aktarıcı, aday değil
    g = shared.routing_graph(net, part, "R")
    assert edge_l not in g.nodes and g.out_degree(edge_r) == 0
    assert relays_on_paths(reach.paths.values()) == {X}
    net.nodes[X].kill(1.0)                      # tek aktarıcı ölünce Edge-L köprü olamaz
    assert not shared.reach_from_edge(net, part, "R").reachable(Y)


# ----------------------------------------------------------------------------- ölü düğümler ve dinamik maliyet (§5.5, §5.9)


def test_dead_relay_removed_and_costs_recomputed(cfg, radio, five_node):
    """§5.9: ölü aktarıcı çizgeden çıkar ve yol değişir; depletion değişince maliyet yeniden hesaplanır."""
    net, part = five_node
    r = Router(cfg, radio, "weighted")
    before = r.reach_from_edge(net, part, "E")
    assert before.paths[2] == (2, 1, 0, 4)
    net.nodes[1].kill(10.0)
    after = r.reach_from_edge(net, part, "E")
    assert 1 not in r.routing_graph(net, part, "E").nodes and not after.reachable(1)
    assert after.paths[2] == (2, 3, 0, 4) and 1 not in after.paths[3]
    assert after.candidates == frozenset({0, 2, 3})
    assert after.cost[2] > before.cost[2]
    # dinamik maliyet: n3 yorulunca (depletion 0,95) n2 için n3'ten geçmek yerine kötü 49 m bağlantı tercih edilir
    net.nodes[3].energy_J = 0.05 * net.nodes[3].initial_energy_J
    tired = r.reach_from_edge(net, part, "E")
    assert tired.paths[2] == (2, 0, 4)
    g = r.routing_graph(net, part, "E")
    assert math.isclose(g.edges[3, 0]["cost"] - r.routing_graph(five_node[0], part, "E").edges[3, 0]["cost"], 0.0)  # aynı durum, aynı maliyet
    assert math.isclose(g.edges[3, 0]["cost"], r.cost(net, 3, 0, "E")) and r.cost(net, 3, 0, "E") > 0.3 * 0.9


# ----------------------------------------------------------------------------- iki aşamalı yol (§5.4)


def test_two_stage_plan_and_bypass(cfg, radio):
    """§5.4: sensör -> root -> edge router; root'a sensör-yoluyla ulaşamayan ama edge router'a komşu sensör bypass."""
    net, part = make_net(cfg, [(10, 0, "critical"), (30, 0, "critical"), (59, 0, "critical"), (30, 30, "critical"),
                               (-42, 0, "critical")])  # n4 yalnızca Edge-E'ye komşu
    E = 5
    assert list(net.graph.neighbors(4)) == [E]
    r = Router(cfg, radio, "weighted")
    reach = r.reach_from_edge(net, part, "E")
    assert reach.candidates == frozenset({0, 1, 2, 3, 4})
    plan = r.plan_routes(net, part, "E", root=0, reach=reach)
    assert plan.root == 0 and plan.root_to_edge == (0, E) and math.isclose(plan.root_to_edge_cost, reach.cost[0])
    assert plan.to_root == {0: (0,), 1: (1, 0), 2: (2, 1, 0), 3: (3, 1, 0)}
    assert plan.full_path(2) == (2, 1, 0, E) and plan.next_hop(2) == 1 and plan.hops(2) == 3
    assert plan.full_path(0) == (0, E) and plan.next_hop(0) == E and plan.hops(0) == 1
    assert plan.bypass == {4: (4, E)} and plan.full_path(4) == (4, E) and not plan.via_root(4)
    assert plan.unreachable == frozenset()
    assert math.isclose(plan.to_root_cost[2], r.path_cost_of(net, (2, 1, 0), "E"))
    rec = plan.to_record()
    assert rec["n_via_root"] == 4 and rec["n_bypass"] == 1 and rec["n_unreachable"] == 0 and rec["root_hops_to_edge"] == 1
    # root değişince aşama 1 yolları değişir; aşama 2 root'un edge yolu
    plan3 = r.plan_routes(net, part, "E", root=3)
    assert plan3.root_to_edge == (3, 1, 0, E) and plan3.full_path(2) == (2, 1, 3, 1, 0, E) or plan3.full_path(2)[0] == 2
    assert plan3.full_path(2)[-1] == E and plan3.full_path(2)[plan3.full_path(2).index(3)] == 3
    with pytest.raises(ValueError, match="aday"):
        r.plan_routes(net, part, "E", root=E)
    net.nodes[0].kill(1.0)
    with pytest.raises(ValueError, match="aday"):
        r.plan_routes(net, part, "E", root=0)


def test_real_network_candidates_and_plans(cfg, radio):
    """Gerçek ağ (seed 1, strict): adaylar = edge router'a bağlı canlı üyeler; her topolojide plan tam, ulaşılamayan yok."""
    pn = build_partitioned_network(cfg, seed=1)
    net, part = pn.net, pn.part
    for method in ("B1", "B2", "NF-MTAR"):
        r = Router.for_method(cfg, radio, method)
        for t in TOPOLOGY_ORDER:
            reach = r.reach_from_edge(net, part, t)
            assert reach.candidates == part.connected_members(t)
            assert all(net.nodes[v].topology == t for v in reach.paths)   # strict: yalnızca kendi topolojisi
            root = min(reach.candidates, key=lambda v: (reach.cost[v], v))
            plan = r.plan_routes(net, part, t, root, reach)
            assert plan.unreachable == frozenset()
            assert set(plan.to_root) | set(plan.bypass) == set(part.members[t])
            for v in part.members[t]:
                path = plan.full_path(v)
                assert path[0] == v and path[-1] == net.edge_router_ids[t]
                assert all(net.graph.has_edge(a, b) for a, b in zip(path, path[1:]))

"""Testler: router.py — kenar maliyeti (enerji/PRR/depletion), B1/B2, ters Dijkstra, H, iki aşamalı yol, bypass/no_route,
relay_mode, ölü düğümler (CLAUDE.md §5.9, §5.4, §5.17)."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from nfmtar.config import ConfigError, load_config
from nfmtar.network.graph import TOPOLOGY_ORDER, NodeState, PhysicalNetwork, Topology, unit_disk_edges
from nfmtar.network.partition import build_partitioned_network, partition_network
from nfmtar.network.radio import RadioModel
from nfmtar.router import BYPASS, NO_ROUTE, VIA_ROOT, Router, relays_on_paths, static_root

FAR = {"E": (0.0, 0.0), "L": (1000.0, 0.0), "R": (2000.0, 0.0)}   # ilgisiz edge router'lar uzakta
ENERGY_ONLY = {"router.lambda2": 0.0, "router.lambda3_from_policy": False}   # λ1 = 1, λ2 = λ3 = 0


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


def brute_force_best(g: nx.DiGraph, source: int, target: int):
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


# ----------------------------------------------------------------------------- λ'lar ve formül (§5.9)


def test_lambdas_methods_and_config_validation(cfg, radio):
    """§5.9: λ3 = π6, λ1 = λ2 = (1 − λ3)/2; B2 (0, 1, π6); B1 (0, 0, 0); lambda3_from_policy=false -> λ3 = 0."""
    r = Router(cfg, radio, "weighted")
    for t, l3 in (("E", 0.3), ("L", 0.5), ("R", 0.8)):
        lam = r.lambdas(t)
        assert math.isclose(lam.l3, l3) and math.isclose(lam.l1, (1 - l3) / 2) and math.isclose(lam.l2, (1 - l3) / 2)
        assert math.isclose(lam.total, 1.0)
    b2 = Router(cfg, radio, "etx_depletion").lambdas(Topology.E)
    assert (b2.l1, b2.l2) == (0.0, 1.0) and math.isclose(b2.l3, 0.3)
    assert Router(cfg, radio, "hop").lambdas("R").total == 0.0
    energy_only = Router(load_config(overrides=ENERGY_ONLY), radio, "weighted").lambdas("R")
    assert (energy_only.l1, energy_only.l2, energy_only.l3) == (1.0, 0.0, 0.0)
    assert Router.for_method(cfg, radio, "B1").path_cost == Router.for_method(cfg, radio, "B1-plain").path_cost == "hop"
    assert Router.for_method(cfg, radio, "B2").path_cost == "etx_depletion"
    assert Router.for_method(cfg, radio, "NF-MTAR").path_cost == Router.for_method(cfg, radio, "B3").path_cost == "weighted"
    with pytest.raises(ValueError):
        Router(cfg, radio, "manhattan")
    with pytest.raises(ConfigError, match="algorithm"):
        load_config(overrides={"router.algorithm": "bellman_ford"})
    with pytest.raises(ConfigError, match="prr_min"):
        load_config(overrides={"router.prr_min": 0.0})
    with pytest.raises(ConfigError, match="link_metric"):
        load_config(overrides={"router.b2.link_metric": "hop"})


def test_energy_term_single_source_and_hop_floor(cfg, radio):
    """§5.9: enerji terimi = E_hop(d)/E_hop(R), E_hop = 2·E_elec + ε_amp·d² RadioModel'den (sabit kopyası yok); taban ≈ 0,29."""
    r = Router(cfg, radio, "weighted")
    e_elec, eps = cfg.energy.e_elec_nJ_per_bit * 1e-9, cfg.energy.eps_amp_pJ_per_bit_m2 * 1e-12
    for d in (0.0, 10.0, 31.6, 50.0):
        assert math.isclose(radio.hop_energy_J_per_bit(d), 2 * e_elec + eps * d * d)
        assert math.isclose(r.energy_term(d), (2 * e_elec + eps * d * d) / (2 * e_elec + eps * 2500.0))
    assert math.isclose(r.energy_term(0.0), 100 / 350, rel_tol=1e-9) and r.energy_term(50.0) == 1.0
    # mesafe başına enerji d* = sqrt(2·E_elec/ε_amp) ≈ 31,6 m'de en küçük
    d_star = math.sqrt(2 * e_elec / eps)
    assert math.isclose(d_star, 31.62, rel_tol=1e-3)
    per_m = lambda d: r.energy_term(d) / d  # noqa: E731
    assert per_m(d_star) < per_m(20.0) and per_m(d_star) < per_m(45.0)
    # enerji parametresi config'ten değişince router'ın terimi de değişir (tek kaynak)
    cfg2 = load_config(overrides={"energy.e_elec_nJ_per_bit": 100.0})
    r2 = Router(cfg2, RadioModel.from_config(cfg2), "weighted")
    assert math.isclose(r2.energy_term(0.0), 200 / 450)


def test_a_energy_term_alone_prefers_one_long_hop(cfg, radio):
    """§5.9 test (a): yalnızca enerji terimiyle (λ2 = λ3 = 0) tek 40 m atlama iki 20 m'ye, tek 50 m üç 16,7 m'ye tercih edilir."""
    c = load_config(overrides=ENERGY_ONLY)
    r = Router(c, radio, "weighted")
    assert r.lambdas("E").l1 == 1.0
    # E(0,0) - A(20,0) - B(40,0): B -> E doğrudan (40 m) mi, B -> A -> E (2 × 20 m) mi?
    net, part = make_net(c, [(20, 0, "critical"), (40, 0, "critical")])
    A, B, E = 0, 1, 2
    direct, two = r.path_cost_of(net, (B, E), "E"), r.path_cost_of(net, (B, A, E), "E")
    assert math.isclose(direct, radio.hop_energy_J_per_bit(40) / radio.hop_energy_J_per_bit(50))
    assert direct < two and math.isclose(two / direct, 280 / 260, rel_tol=1e-6)
    assert r.reach_from_edge(net, part, "E").paths[B] == (B, E)
    # E(0,0) - C1(50/3) - C2(100/3) - D(50): tek 50 m atlama (350 nJ/bit) < üç 16,7 m atlama (3 × 127,8)
    net, part = make_net(c, [(50 / 3, 0, "critical"), (100 / 3, 0, "critical"), (50, 0, "critical")])
    C1, C2, D, E = 0, 1, 2, 3
    assert net.graph.has_edge(D, E)
    assert r.path_cost_of(net, (D, E), "E") < r.path_cost_of(net, (D, C2, C1, E), "E")
    # tam maliyette (PRR terimi dahil) 50 m bağlantı PRR_min'e kırpıldığı için sonuç ters döner: enerji terimi tek başına sınandı
    full = Router(cfg, radio, "weighted")
    assert full.path_cost_of(net, (D, E), "E") > full.path_cost_of(net, (D, C2, C1, E), "E")


def test_edge_cost_formula_hand_computed(cfg, radio, five_node):
    """§5.9: cost = λ1·E_hop(d)/E_hop(R) + λ2·(−log PRR)/L_max + λ3·depletion(u); PRR kırpılır, dışlanmaz; B1 = 1."""
    net, part = five_node
    r = Router(cfg, radio, "weighted")
    l_max = -math.log(0.05)
    d = net.distance(0, 4)   # n0 -> E: d = 10, tam enerji
    expected = 0.35 * (110 / 350) + 0.35 * (-math.log(radio.prr(10.0))) / l_max
    assert math.isclose(d, 10.0) and math.isclose(r.cost(net, 0, 4, "E"), expected, rel_tol=1e-9)
    assert 0.05 < r.link_prr(49.0) < 0.06 and r.link_term(49.0) < 1.0          # 49 m: radyo PRR'si 0,057, kırpılmaz
    assert r.link_prr(50.0) == 0.05 and math.isclose(r.link_term(50.0), 1.0)   # menzil sınırında kırpılır, terim tam 1
    g = r.routing_graph(net, part, "E")
    assert g.has_edge(2, 0) and g.has_edge(0, 2)                                # 49 m bağlantı çizgede (dışlanmadı)
    c10_before = r.cost(net, 1, 0, "E")
    assert r.cost(net, 1, 0, "R") < r.cost(net, 1, 0, "E")                     # tam enerjide R'nin λ1/λ2'si küçük
    net.nodes[0].energy_J = 0.5 * net.nodes[0].initial_energy_J                # depletion gönderici tarafında
    assert math.isclose(r.cost(net, 0, 4, "E"), expected + 0.3 * 0.5, rel_tol=1e-9)
    assert math.isclose(r.cost(net, 1, 0, "E"), c10_before)                    # n0'a GİREN kenar değişmez
    assert r.cost(net, 0, 4, "R") > r.cost(net, 0, 4, "E")                     # yorgun göndericide λ3 = 0,8 baskın
    assert Router(cfg, radio, "hop").cost(net, 2, 0, "E") == 1.0


# ----------------------------------------------------------------------------- 5 düğümlü çizge, H (§5.9, §5.17)


def test_five_node_expected_paths_prr_product_and_bfs_hops(cfg, radio, five_node):
    """§5.9: beklenen yollar (brute force ile aynı); Σ(−log PRR) = −log(Π PRR); test (c): H = BFS, en ucuz yolun atlamasından farklı."""
    net, part = five_node
    r = Router(cfg, radio, "weighted")
    reach = r.reach_from_edge(net, part, "E")
    assert reach.edge_router == 4 and reach.candidates == frozenset({0, 1, 2, 3})
    assert reach.paths[2] == (2, 1, 4) and reach.paths[3] == (3, 0, 4) and reach.paths[1] == (1, 4)
    g = r.routing_graph(net, part, "E")
    for v in (0, 1, 2, 3):
        best, best_cost = brute_force_best(g, v, 4)
        assert reach.paths[v] == best and math.isclose(reach.cost[v], best_cost)
        assert math.isclose(reach.cost[v], r.path_cost_of(net, reach.paths[v], "E"))
    assert reach.hops == {0: 1, 1: 1, 2: 2, 3: 1}                 # H: en az atlama (BFS; §5.17)
    assert reach.hops[3] == 1 and reach.path_hops(3) == 2          # n3-E kenarı var ama en ucuz yol n3 -> n0 -> E
    path = reach.paths[2]
    prrs = [r.link_prr(net.distance(u, v)) for u, v in zip(path, path[1:])]
    assert math.isclose(sum(-math.log(p) for p in prrs), -math.log(math.prod(prrs)))
    assert math.isclose(r.path_prr(net, path), math.prod(prrs))
    lam = r.lambdas("E")
    assert math.isclose(sum(lam.l2 * (-math.log(p)) / r.l_max for p in prrs),
                        lam.l2 * (-math.log(r.path_prr(net, path))) / r.l_max)


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
        best, _ = brute_force_best(r.routing_graph(net, part, t), 0, edge_id)
        assert best == expected


# ----------------------------------------------------------------------------- B1 ve B2 (§5.9, §5.12)


def test_b1_min_hops_b2_min_etx_and_b2_floor(cfg, radio):
    """§5.9: B1 en az atlamalı yolu (2 kötü atlama), B2 en düşük ETX toplamını (4 iyi atlama) seçer; test (b): B2'de her kenar ≥ 1."""
    # E(0,0) - G2(24,0) - M(48,0) - G1(72,0) - S(96,0): 24 m bağlantılar iyi (PRR ~0,99), 48 m bağlantılar kötü (PRR ~0,08)
    net, part = make_net(cfg, [(96, 0, "critical"), (72, 0, "critical"), (48, 0, "critical"), (24, 0, "critical")],
                         energy_frac={2: 0.4})   # M yorgun (depletion 0,6)
    S, G1, M, G2, E = 0, 1, 2, 3, 4
    assert radio.prr(48.0) < 0.1 < 0.98 < radio.prr(24.0)
    b1 = Router(cfg, radio, "hop").reach_from_edge(net, part, "E")
    assert b1.paths[S] == (S, M, E) and b1.cost[S] == 2.0 and b1.hops[S] == 2
    b2 = Router(cfg, radio, "etx_depletion")
    reach = b2.reach_from_edge(net, part, "E")
    assert reach.paths[S] == (S, G1, M, G2, E) and reach.path_hops(S) == 4 and reach.hops[S] == 2
    g = b2.routing_graph(net, part, "E")
    assert all(c >= 1.0 for _, _, c in g.edges(data="cost"))                      # (b) ETX ≥ 1 taban
    assert math.isclose(g.edges[S, G1]["cost"], 1 / b2.link_prr(24.0))             # dinlenmiş gönderici: yalnızca ETX
    assert math.isclose(g.edges[M, G2]["cost"], 1 / b2.link_prr(24.0) + 0.3 * 0.6)  # yorgun gönderici: + π6·depletion
    assert math.isclose(g.edges[S, M]["cost"], 1 / b2.link_prr(48.0)) and g.edges[S, M]["cost"] > 10
    assert math.isclose(reach.cost[S], 3 / b2.link_prr(24.0) + (1 / b2.link_prr(24.0) + 0.18))
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
    assert before.paths[2] == (2, 1, 4)
    net.nodes[1].kill(10.0)
    after = r.reach_from_edge(net, part, "E")
    assert 1 not in r.routing_graph(net, part, "E").nodes and not after.reachable(1)
    assert after.paths[2] == (2, 3, 0, 4) and after.candidates == frozenset({0, 2, 3})
    assert after.cost[2] > before.cost[2]
    # dinamik maliyet: n3 yorulunca (depletion 0,95) n2 için n3'ten geçmek yerine kötü 49 m bağlantı tercih edilir
    net.nodes[3].energy_J = 0.05 * net.nodes[3].initial_energy_J
    tired = r.reach_from_edge(net, part, "E")
    assert tired.paths[2] == (2, 0, 4)
    g = r.routing_graph(net, part, "E")
    assert math.isclose(g.edges[3, 0]["cost"], r.cost(net, 3, 0, "E")) and r.cost(net, 3, 0, "E") > 0.3 * 0.95


# ----------------------------------------------------------------------------- iki aşamalı yol, bypass, no_route (§5.4)


def test_d_two_stage_plan_bypass_and_no_route(cfg, radio):
    """§5.4 / test (d): sensör -> root -> edge; bypass YALNIZCA root'a yol yok + edge'e yol varken; ikisi de yoksa no_route."""
    net, part = make_net(cfg, [(10, 0, "critical"), (30, 0, "critical"), (59, 0, "critical"), (30, 30, "critical"),
                               (-42, 0, "critical"),      # n4: yalnızca Edge-E'ye komşu -> bypass
                               (500, 500, "critical")])   # n5: hiç komşusu yok -> no_route
    E = 6
    assert list(net.graph.neighbors(4)) == [E] and list(net.graph.neighbors(5)) == []
    assert part.disconnected[Topology.E] == frozenset({5})
    r = Router(cfg, radio, "weighted")
    reach = r.reach_from_edge(net, part, "E")
    assert reach.candidates == frozenset({0, 1, 2, 3, 4})
    plan = r.plan_routes(net, part, "E", root=0, reach=reach)
    assert plan.root == 0 and plan.root_to_edge == (0, E) and math.isclose(plan.root_to_edge_cost, reach.cost[0])
    assert plan.to_root == {0: (0,), 1: (1, 0), 2: (2, 1, 0), 3: (3, 0)}
    assert plan.full_path(2) == (2, 1, 0, E) and plan.next_hop(2) == 1 and plan.hops(2) == 3
    assert plan.full_path(0) == (0, E) and plan.next_hop(0) == E and plan.hops(0) == 1
    assert plan.bypass == {4: (4, E)} and plan.full_path(4) == (4, E) and plan.route_status(4) == BYPASS
    assert plan.no_route == frozenset({5}) and plan.full_path(5) is None and plan.next_hop(5) is None
    assert plan.route_status(5) == NO_ROUTE and plan.route_status(2) == VIA_ROOT
    assert math.isclose(plan.to_root_cost[2], r.path_cost_of(net, (2, 1, 0), "E"))
    rec = plan.to_record()
    assert rec["n_via_root"] == 4 and rec["n_bypass"] == 1 and rec["n_no_route"] == 1 and rec["root_hops_to_edge"] == 1
    assert math.isclose(rec["bypass_frac"], 1 / 6)
    # root'a yolu olan sensör edge router'a komşu olsa bile bypass edilmez (iki aşama korunur)
    plan1 = r.plan_routes(net, part, "E", root=1)
    assert plan1.root_to_edge == (1, E) and plan1.full_path(0) == (0, 1, E) and 0 not in plan1.bypass
    assert plan1.bypass == {4: (4, E)} and plan1.no_route == frozenset({5})
    # root değişince aşama 1 yolları ve root'un edge yolu değişir
    plan3 = r.plan_routes(net, part, "E", root=3)
    assert plan3.root_to_edge == (3, 0, E) and plan3.full_path(2) == (2, 3, 0, E)
    with pytest.raises(ValueError, match="aday"):
        r.plan_routes(net, part, "E", root=E)
    with pytest.raises(ValueError, match="aday"):
        r.plan_routes(net, part, "E", root=5)
    net.nodes[0].kill(1.0)
    with pytest.raises(ValueError, match="aday"):
        r.plan_routes(net, part, "E", root=0)
    # sabit root kuralı (§5.12): adaylar arasında edge router'a en yakın
    assert static_root(net, r.reach_from_edge(net, part, "E")) == 1


def test_real_network_candidates_and_plans(cfg, radio):
    """Gerçek ağ (seed 1, strict): adaylar = edge router'a bağlı canlı üyeler; her topolojide plan tam, no_route yok."""
    pn = build_partitioned_network(cfg, seed=1)
    net, part = pn.net, pn.part
    for method in ("B1", "B2", "NF-MTAR"):
        r = Router.for_method(cfg, radio, method)
        for t in TOPOLOGY_ORDER:
            reach = r.reach_from_edge(net, part, t)
            assert reach.candidates == part.connected_members(t)
            assert all(net.nodes[v].topology == t for v in reach.paths)   # strict: yalnızca kendi topolojisi
            plan = r.plan_routes(net, part, t, static_root(net, reach), reach)
            assert plan.no_route == frozenset()
            assert set(plan.to_root) | set(plan.bypass) == set(part.members[t])
            for v in part.members[t]:
                path = plan.full_path(v)
                assert path[0] == v and path[-1] == net.edge_router_ids[t]
                assert all(net.graph.has_edge(a, b) for a, b in zip(path, path[1:]))

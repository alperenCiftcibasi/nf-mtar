"""Testler: simulation.py + metrics.py — korunum, determinizm, ortak rastgele sayılar, tek atlama bileşenleri, enerji,
kalibrasyon köprüsü, kayıp nedenleri, ölüm/root yeniden seçimi, B1-plain, runs.csv ve paket logu
(CLAUDE.md §5.5, §5.10a, §5.12, §5.13, §5.19)."""

from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd
import pytest

from nfmtar.config import PROJECT_ROOT, load_config
from nfmtar.metrics import (
    append_runs_row,
    class_metrics,
    network_metrics,
    packet_log_frame,
    per_class_table,
    rows_equal_ignoring_runtime,
    runs_row,
    write_packet_log,
)
from nfmtar.network.graph import Topology
from nfmtar.network.partition import PartitionedNetwork
from nfmtar.network.radio import RadioModel
from nfmtar.simulation import (
    DELIVERED,
    DROP_IN_FLIGHT_AT_END,
    DROP_NO_ROUTE,
    DROP_NODE_DEAD,
    DROP_QUEUE_OVERFLOW,
    DROP_REASONS,
    IN_FLIGHT,
    Simulation,
    StaticRootPolicy,
    run_simulation,
)
from nfmtar.traffic import generate_trace
from tests.test_router import make_net

CLASSES = ("critical", "regular", "noncritical")
PRR_ONE = {"radio.prr.prr_max": 1.0, "radio.prr.d_mid_m": 1e6}   # teslim kesin (yeniden deneme yok)
SHORT = {"simulation.duration_s": 20.0, "run.warm_up_s": 2.0}      # gerçek ağda kısa koşu


def pn_of(cfg, net, part) -> PartitionedNetwork:
    return PartitionedNetwork(net, part, cfg.placement.random.r_placement, cfg.router.relay_mode,
                              cfg.network.enforce_partition_connectivity)


def run_handmade(cfg, sensors, sources, duration_s, seed=1, method="B1", edge_pos=None, energy_frac=None):
    """Elle kurulmuş ağ + o ağın sensörleri için iz + koşu."""
    net, part = make_net(cfg, sensors, edge_pos=edge_pos, energy_frac=energy_frac)
    trace = generate_trace(cfg, sources, seed, duration_s)
    return Simulation(cfg, pn_of(cfg, net, part), trace, method).run(), net


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def short_b1():
    """Gerçek ağ (seed 1, N = 150), B1, 20 s (+5 s drain), ısınma 2 s."""
    return run_simulation(load_config(overrides=SHORT), "B1", seed=1)


# ----------------------------------------------------------------------------- (a) korunum, bileşen toplamı (§5.10a)


def test_a_conservation_per_class_and_latency_is_sum_of_components(short_b1):
    """§5.10a: her sınıfta üretilen = teslim + Σ düşen (nedene göre) + in_flight_at_end; teslimde gecikme = Σ bileşen."""
    res = short_b1
    assert all(ps.status != IN_FLIGHT for ps in res.packets)
    for cls in CLASSES:
        pk = [ps for ps in res.packets if ps.packet.traffic_class == cls]
        n_del = sum(1 for ps in pk if ps.status == DELIVERED)
        by_reason = {r: sum(1 for ps in pk if ps.status == r) for r in DROP_REASONS}
        assert len(pk) == n_del + sum(by_reason.values())
        assert n_del > 0
        m = class_metrics(res, cls)   # ısınma sonrası alt küme için de aynı korunum (metrics.py)
        assert m["generated"] == m["delivered"] + sum(m[f"drop_{r}"] for r in DROP_REASONS) == m["delivered"] + m["drop_total"]
    for ps in res.packets:
        if ps.delivered:
            assert ps.hops >= 1 and ps.end_node == res.pn.net.edge_router_ids[ps.topology]
            assert math.isclose(ps.latency_s, sum(ps.components_s().values()) - ps.aifs_s, abs_tol=1e-9)  # aifs backoff'un parçası
            assert ps.end_s <= res.duration_s + res.drain_s
        else:
            assert ps.status in DROP_REASONS and not math.isnan(ps.end_s)
    assert res.ttl_drops == 0
    nm = network_metrics(res)
    assert nm["generated_total"] == sum(class_metrics(res, c)["generated"] for c in CLASSES)
    assert nm["dead_nodes"] == 0 and math.isnan(nm["fnd_s"]) and nm["root_changes_total"] == 0


# ----------------------------------------------------------------------------- (b) determinizm (§3)


def test_b_same_seed_same_runs_row():
    """§3: aynı seed iki kez -> runs.csv satırı aynı (zaman damgası, duvar saati ve karar süreleri hariç)."""
    c = load_config(overrides={**SHORT, "network.n_nodes": 60})
    r1, r2 = run_simulation(c, "B2", seed=4), run_simulation(c, "B2", seed=4)
    row1, row2 = runs_row(r1, "t"), runs_row(r2, "t")
    assert rows_equal_ignoring_runtime(row1, row2)
    assert [ps.status for ps in r1.packets] == [ps.status for ps in r2.packets]
    assert [ps.end_s for ps in r1.packets] == [ps.end_s for ps in r2.packets]
    r3 = run_simulation(c, "B2", seed=5)
    assert not rows_equal_ignoring_runtime(row1, runs_row(r3, "t"))


# ----------------------------------------------------------------------------- (c) ortak rastgele sayılar (§5.6)


def test_c_common_random_numbers_b1_b2_same_trace_and_network():
    """§5.6: aynı seed'de B1 ve B2 birebir aynı paket izini (ve aynı ağı) görür; yalnızca iletim sonucu değişir."""
    c = load_config(overrides={**SHORT, "network.n_nodes": 60})
    r1, r2 = run_simulation(c, "B1", seed=2), run_simulation(c, "B2", seed=2)
    assert [p.key() for p in r1.trace.packets] == [p.key() for p in r2.trace.packets]
    assert [p.packet_id for p in r1.trace.packets] == [p.packet_id for p in r2.trace.packets]
    assert r1.pn.net.positions().tolist() == r2.pn.net.positions().tolist()
    assert r1.pn.part.members == r2.pn.part.members
    for cls in CLASSES:
        assert class_metrics(r1, cls)["generated"] == class_metrics(r2, cls)["generated"]
    assert r1.roots == r2.roots   # sabit root kuralı yöntemden bağımsız (§5.12)


# ----------------------------------------------------------------------------- (d) tek atlama, çekişmesiz (§5.5)


def test_d_single_hop_no_contention_components():
    """§5.5: iletim = parça × çerçeve_bit/250000; geri çekilme parça başına [0, (2^BE_min − 1)·320 µs]; AIFS sınıf sabiti × parça."""
    c = load_config(overrides={**PRR_ONE, "run.warm_up_s": 0.0, "traffic.critical_episodes.enabled": False})
    radio = RadioModel.from_config(c)
    slot = c.radio.csma.backoff_slot_s
    # düzenli sensör Edge-L'nin 10 m yanında, kritik sensör Edge-E'nin 10 m yanında; aralarında 1000 m (çekişme yok)
    res, net = run_handmade(c, [(1010, 0, "regular"), (10, 0, "critical")], {0: "regular", 1: "critical"}, duration_s=20.0)
    assert all(ps.delivered for ps in res.packets)
    for ps in res.packets:
        cls = ps.packet.traffic_class
        n_frames = radio.n_fragments(ps.packet.size_bytes)
        assert ps.hops == 1 and ps.frame_attempts == n_frames
        assert math.isclose(ps.transmission_s, radio.transmission_time_s(ps.packet.size_bytes), rel_tol=1e-12)
        assert math.isclose(ps.transmission_s, n_frames * radio.frame_bytes(90) * 8 / 250000, rel_tol=0.35)  # son parça kısa olabilir
        assert math.isclose(ps.aifs_s, n_frames * c.radio.csma.aifs_slots[cls] * slot, rel_tol=1e-9)
        random_backoff = ps.backoff_s - ps.aifs_s
        assert -1e-12 <= random_backoff <= n_frames * (2 ** c.radio.csma.mac_min_be[cls] - 1) * slot + 1e-12
        assert math.isclose(ps.processing_s, c.radio.processing_delay_s)
        assert math.isclose(ps.latency_s, ps.queue_s + ps.backoff_s + ps.transmission_s + ps.processing_s, abs_tol=1e-9)
    regular = [ps for ps in res.packets if ps.packet.traffic_class == "regular"]
    assert regular and all(ps.queue_s == 0.0 for ps in regular)            # 1 s aralıkla gelen paketler kuyrukta beklemez
    assert all(ps.aifs_s > 0 for ps in regular) and c.radio.csma.aifs_slots["regular"] == 5
    critical = [ps for ps in res.packets if ps.packet.traffic_class == "critical"]
    assert critical and all(ps.aifs_s == 0.0 for ps in critical)           # kritik AIFS 0
    m = class_metrics(res, "regular")
    assert m["pdr"] == 1.0 and m["hops_mean"] == 1.0 and math.isclose(m["processing_ms"], 1.0)
    assert math.isclose(m["aifs_ms"], np.mean([ps.aifs_s for ps in regular]) * 1000)


# ----------------------------------------------------------------------------- (e) enerji (§5.5, §5.10a)


def test_e_energy_equals_sum_of_frame_attempts(short_b1):
    """§5.10a: Σ sensör tüketimi = Σ parça denemelerinin gönderici + (sensör) alıcı enerjisi; edge router alımı ayrı."""
    res = short_b1
    assert res.hops_active_at_end == 0 and not res.deaths   # önkoşul: sayaçlar bütün denemeleri kapsıyor, ölüm kırpması yok
    consumed = sum(res.energy_consumed_J().values())
    assert consumed > 0 and math.isclose(consumed, res.energy_tx_J + res.energy_rx_J, rel_tol=1e-9)
    assert res.energy_rx_edge_J > 0
    nm = network_metrics(res)
    assert math.isclose(nm["energy_total_J"], consumed) and math.isclose(
        sum(nm[f"energy_J_{t.value}"] for t in Topology), consumed, rel_tol=1e-9)
    net = res.pn.net
    for v in net.sensor_ids:
        assert net.nodes[v].energy_J <= net.nodes[v].initial_energy_J


# ----------------------------------------------------------------------------- (f) kalibrasyon köprüsü (§5.5)


def test_f_calibration_bridge_saturated_shares_match_table(cfg):
    """§5.5: üç doygun gönderici (sınıf başına bir, hepsi cs menzilinde) simulation.py'de koşunca airtime payları
    kalibrasyon CSV'sinin seçili (0/5/7) satırıyla ±0,02 içinde tutmalı (aynı zamanlama kuralı ve MAC mekanizması)."""
    table = PROJECT_ROOT / cfg.experiments["mac_aifs_calibration"]["table"]
    if not table.exists():
        pytest.skip("kalibrasyon tablosu yok: python experiments/calibration/mac_aifs.py")
    with open(table, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scenario"] == "grid"
                and all(int(r[f"aifs_{k}"]) == cfg.radio.csma.aifs_slots[k] for k in CLASSES)]
    assert len(rows) == 1, "config'teki aifs_slots ızgarada tek satırla eşleşmeli"
    target = {k: float(rows[0][f"share_{k}"]) for k in CLASSES}

    c = load_config(overrides={"traffic.load_multiplier": 2000.0, "traffic.critical_episodes.enabled": False,
                               "run.warm_up_s": 0.0, "simulation.drain_s": 0.0})
    sensors = [(10, 0, "critical"), (10, 10, "regular"), (10, 20, "noncritical")]   # kalibrasyon yerleşimi
    edge_pos = {"E": (0.0, 0.0), "L": (0.0, 10.0), "R": (0.0, 20.0)}
    shares = []
    for seed in (1, 2, 3):
        res, net = run_handmade(c, sensors, {0: "critical", 1: "regular", 2: "noncritical"}, duration_s=20.0,
                                seed=seed, edge_pos=edge_pos)
        assert all(res.roots[t] == i for i, t in enumerate(Topology))           # her sensör kendi topolojisinin root'u
        for cls in CLASSES:
            assert class_metrics(res, cls)["drop_queue_overflow"] > 0            # gerçekten doygun
        air = res.airtime_after_warm_up_s()
        total = sum(air.values())
        shares.append({k: air[k] / total for k in CLASSES})
    mean = {k: float(np.mean([s[k] for s in shares])) for k in CLASSES}
    for k in CLASSES:
        assert abs(mean[k] - target[k]) <= 0.02, (k, mean, target)


# ----------------------------------------------------------------------------- kayıp nedenleri, ölüm, root yeniden seçimi (§5.10a, §5.12)


def test_no_route_and_in_flight_at_end():
    """§5.10a: edge router'a yolu olmayan sensörün bütün paketleri no_route; drain 0 ile yoldakiler in_flight_at_end."""
    c = load_config(overrides={"run.warm_up_s": 0.0, "simulation.drain_s": 0.0, "traffic.critical_episodes.enabled": False})
    res, _ = run_handmade(c, [(10, 0, "critical"), (500, 500, "critical")], {0: "critical", 1: "critical"}, duration_s=10.0)
    isolated = [ps for ps in res.packets if ps.packet.source == 1]
    assert isolated and all(ps.status == DROP_NO_ROUTE and ps.hops == 0 for ps in isolated)
    connected = [ps for ps in res.packets if ps.packet.source == 0]
    assert all(ps.status in (DELIVERED, DROP_IN_FLIGHT_AT_END) for ps in connected)
    assert class_metrics(res, "critical")["drop_no_route"] == len(isolated)


def test_node_death_drops_queue_and_reselects_static_root():
    """§5.10a / §5.12: aktarıcı/root ölünce kuyruğu node_dead, FND kaydedilir, yollar hemen yenilenir, root aynı kuralla yeniden seçilir."""
    c = load_config(overrides={**PRR_ONE, "run.warm_up_s": 0.0, "traffic.critical_episodes.enabled": False,
                               "traffic.load_multiplier": 20.0})
    # Edge-E (0,0); n0 (20,0) root ve n1/n2'nin aktarıcısı, enerjisi az; n1 (40,0); n2 (60,0): ölümden sonra n2 -> n1 -> E
    res, net = run_handmade(c, [(20, 0, "critical"), (40, 0, "critical"), (60, 0, "critical")],
                            {0: "critical", 1: "critical", 2: "critical"}, duration_s=10.0, energy_frac={0: 2e-5})
    assert len(res.deaths) == 1 and res.deaths[0][1] == 0
    t_death = res.deaths[0][0]
    assert 0 < t_death < 10.0 and not net.nodes[0].alive
    changes = [rc for rc in res.root_changes if rc["topology"] == "E"]
    assert len(changes) == 1 and changes[0]["old"] == 0 and changes[0]["new"] == 1 and changes[0]["reason"] == "death:0"
    assert changes[0]["trigger"] == "root_health" and changes[0]["rule"] == "static_reselect"   # §5.18 log
    assert res.roots[Topology.E] == 1
    after = [ps for ps in res.packets if ps.packet.source == 2 and ps.delivered and ps.packet.created_s > t_death]
    assert after and all(ps.hops == 2 for ps in after)                      # n2 -> n1 -> Edge-E
    nm = network_metrics(res)
    assert nm["dead_nodes"] == 1 and math.isclose(nm["fnd_s"], t_death) and math.isnan(nm["hnd_s"])  # ⌈3/2⌉ = 2. ölüm yok
    assert nm["root_changes_E"] == 1
    dead_drops = [ps for ps in res.packets if ps.status == DROP_NODE_DEAD]
    assert dead_drops and all(ps.end_node == 0 for ps in dead_drops)
    assert any(ps.packet.source == 0 for ps in res.packets if ps.status == DROP_NODE_DEAD)   # ölü kaynağın sonraki paketleri


def test_static_root_policy_keeps_current_candidate(cfg):
    """§5.12: mevcut root aday kaldığı sürece korunur; aday değilse static_root kuralı."""
    from nfmtar.router import Router, static_root

    net, part = make_net(cfg, [(10, 0, "critical"), (30, 0, "critical"), (45, 0, "critical")])
    reach = Router(cfg, RadioModel.from_config(cfg), "hop").reach_from_edge(net, part, "E")
    pol = StaticRootPolicy()
    assert pol.select(Topology.E, net, part, reach, None, 0.0) == static_root(net, reach) == 0
    assert pol.select(Topology.E, net, part, reach, 2, 0.0) == 2          # aday olan mevcut root korunur
    assert pol.select(Topology.E, net, part, reach, 99, 0.0) == 0         # aday değil -> kural


# ----------------------------------------------------------------------------- B1-plain (§5.12)


def test_b1_plain_parameters_and_config_defaults(cfg):
    """§5.12 B1-plain: tek FIFO (8), π10 = 3, macMinBE 3 / AIFS 0 bütün sınıflarda; §5.10a drain_s config'te."""
    assert cfg.simulation.drain_s == 5.0
    assert cfg.policy.plain.buffer_packets == 8 and cfg.policy.plain.retransmit_limit == 3
    net, part = make_net(cfg, [(10, 0, "critical"), (1010, 0, "regular"), (2010, 0, "noncritical")])
    trace = generate_trace(cfg, {0: "critical", 1: "regular", 2: "noncritical"}, 1, 5.0)
    plain = Simulation(cfg, pn_of(cfg, net, part), trace, "B1-plain")
    assert plain.vsdn_policy is False and not plain.backoff.vsdn_policy
    assert all(plain.retransmit_limit[k] == 3 for k in CLASSES)
    assert all(plain.backoff.min_be(k) == 3 and plain.backoff.aifs_slots(k) == 0 for k in CLASSES)
    assert all(q.single_fifo and q.node.queue.capacity == 8 for q in plain.queues.values())
    net2, part2 = make_net(cfg, [(10, 0, "critical"), (1010, 0, "regular"), (2010, 0, "noncritical")])
    vsdn = Simulation(cfg, pn_of(cfg, net2, part2), trace, "B1")
    assert vsdn.vsdn_policy and vsdn.retransmit_limit == {"critical": 5, "regular": 3, "noncritical": 1}
    assert {q.node.traffic_class: q.node.queue.capacity for q in vsdn.queues.values()} == {"critical": 16, "regular": 8, "noncritical": 4}
    assert vsdn.queues[0].order == ["critical", "regular", "noncritical"]    # π2: 7 > 4 > 1
    with pytest.raises(KeyError, match="bilinmeyen"):
        Simulation(cfg, pn_of(cfg, net2, part2), trace, "RPL")


# ----------------------------------------------------------------------------- runs.csv ve paket logu (§5.19)


def test_runs_csv_append_and_packet_log(short_b1, tmp_path):
    """§5.19: her koşu bir satır (sütun birleşimiyle ekleme); paket logu bir satır bir paket, parquet/csv.gz."""
    row = runs_row(short_b1, "unit")
    assert row["run_id"] == "unit-B1-s1-l1-strict" and row["method"] == "B1" and row["seed"] == 1
    assert {"pdr_critical", "lat_p95_ms_regular", "drop_link_loss_noncritical", "energy_total_J", "fnd_s",
            "airtime_share_critical", "deadline_ratio_critical", "window_ratio_regular", "regen_attempts"} <= set(row)
    path = tmp_path / "runs.csv"
    append_runs_row(row, path)
    append_runs_row({**row, "extra_col": 1.0}, path)
    df = pd.read_csv(path)
    assert len(df) == 2 and "extra_col" in df.columns and df["run_id"].tolist() == [row["run_id"]] * 2
    table = per_class_table(short_b1)
    assert list(table.index) == list(CLASSES) and "pdr" in table.columns
    frame = packet_log_frame(short_b1)
    assert len(frame) == len(short_b1.packets) and frame["delivered"].sum() == sum(1 for ps in short_b1.packets if ps.delivered)
    out = write_packet_log(short_b1, tmp_path / "log", "parquet")
    assert out.suffix in (".parquet", ".gz") and out.exists()
    back = pd.read_parquet(out) if out.suffix == ".parquet" else pd.read_csv(out)
    assert len(back) == len(frame)


# ----------------------------------------------------------------------------- Φ (§5.13)


def test_phi_formula_hand_computed():
    """§5.13: Φ_t = α·D̄/π3 + β·(1 − PDR) + γ·U, γ = π6, α = β = (1 − π6)/2; elle iki örnek."""
    from nfmtar.metrics import phi_terms

    # E (π6 = 0,3, π3 = 25 ms): D̄ = 20 ms, PDR 0,9, U 0,1 -> 0,35·0,8 + 0,35·0,1 + 0,3·0,1 = 0,345
    t = phi_terms(20.0, 25.0, 0.9, 0.1, 0.3)
    assert math.isclose(t["phi"], 0.345) and math.isclose(t["phi_delay_norm"], 0.8) and math.isclose(t["phi_loss"], 0.1)
    # R (π6 = 0,8, π3 = 500 ms): D̄ = 500 ms (sınırda), PDR 1, U 0,5 -> 0,1·1 + 0 + 0,8·0,5 = 0,5
    assert math.isclose(phi_terms(500.0, 500.0, 1.0, 0.5, 0.8)["phi"], 0.5)
    assert math.isnan(phi_terms(math.nan, 25.0, 0.0, 0.1, 0.3)["phi"])        # teslim yoksa tanımsız


def test_phi_in_class_metrics_and_runs_row(short_b1):
    """§5.13: sınıf Φ_t'si sınıf metriklerinden ve ısınma sonrası topoloji enerjisinden; runs.csv Φ = sınıf ortalaması."""
    res = short_b1
    cfg = res.cfg
    net, part = res.pn.net, res.pn.part
    phis = []
    for cls in CLASSES:
        m = class_metrics(res, cls)
        topo = res.class_topology[cls]
        pol = cfg.policy.for_topology(topo)
        members = part.members[topo]
        used = sum(res.energy_warm_up_J[v] - net.nodes[v].energy_J for v in members)
        u = used / sum(net.nodes[v].initial_energy_J for v in members)
        a = (1 - pol.energy_weight) / 2
        expected = a * m["lat_mean_ms"] / pol.max_delay_ms + a * (1 - m["pdr"]) + pol.energy_weight * u
        assert 0 < u < 1 and math.isclose(m["phi_U"], u) and math.isclose(m["phi"], expected, rel_tol=1e-12)
        phis.append(m["phi"])
    # ısınma sonrası enerji, bütün koşunun enerjisinden küçük (ısınmada da tüketim var)
    assert sum(res.energy_consumed_after_warm_up_J().values()) < sum(res.energy_consumed_J().values())
    assert math.isclose(runs_row(res, "t")["phi"], float(np.mean(phis)))


def test_x2_radio_busy_ratio_from_simulation(cfg):
    """§5.14 x2: simülasyon her atlamanın çerçeve aralıklarını gönderici ve sensör alıcıya ekler. Kısa B3 koşusunda
    oranlar [0,1]; trafik taşıyan düğümlerin oranı > 0; hiç iletmeyen/almayan düğüm 0; controller x2'yi bu
    izleyiciden okur (aynı nesne) ve extract_features x2'si izleyicinin değerine eşit."""
    from nfmtar.engine.fuzzy import FuzzySpec, extract_features, nominal_rates_pps
    from nfmtar.network.partition import build_partitioned_network
    from nfmtar.traffic import build_traffic_trace

    c = load_config(overrides={"simulation.duration_s": 20.0, "run.warm_up_s": 2.0, "network.n_nodes": 60,
                               "traffic.load_multiplier": 0.5})
    pn = build_partitioned_network(c, 3)
    sim = Simulation(c, pn, build_traffic_trace(c, pn.net, 3), "B3")
    assert sim.root_policy.busy_tracker is sim.busy
    sim.run()
    t = sim.now
    ratios = sim.busy.busy_ratios(pn.net.sensor_ids, t)
    assert all(0.0 <= r <= 1.0 for r in ratios.values())
    assert max(ratios.values()) > 0.0
    roots = [r for r in sim.routes.values() if r.root is not None]
    assert all(ratios[rs.root] > 0 for rs in roots)                         # root bütün trafiği taşır
    spec = FuzzySpec.from_config(c)
    feats = extract_features(spec, pn.net, pn.part, "E", nominal_rates_pps(c, pn.net), sim.radio.prr, ratios)
    assert all(f.x2 == pytest.approx(ratios[v]) for v, f in feats.items())

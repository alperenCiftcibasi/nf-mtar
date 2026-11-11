"""Testler: radio.py — çerçeveleme, PRR, enerji, sınıf bazlı CSMA/CA (geri çekilme + AIFS), yerel taşıyıcı
algılama, π10 (CLAUDE.md §5.5, §5.7)."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from nfmtar.config import PROJECT_ROOT, ConfigError, load_config
from nfmtar.network.graph import NodeState
from nfmtar.network.radio import (
    LOSS_CSMA_FAILURE,
    LOSS_RETRY_LIMIT,
    LOSS_SENDER_DEAD,
    BackoffPolicy,
    Channel,
    RadioModel,
    run_mac_sessions,
    saturated_sender,
    send_packet,
)

CLASSES = ("critical", "regular", "noncritical")

PRR_ONE = {"radio.prr.prr_max": 1.0, "radio.prr.d_mid_m": 1e6}      # teslim kesin
PRR_ZERO = {"radio.prr.prr_max": 0.0, "radio.prr.prr_min": 0.0}       # teslim imkânsız


def node(node_id: int, x: float, y: float, energy: float = 2.0, cls: str = "critical") -> NodeState:
    return NodeState(node_id, (x, y), cls, False, energy, energy, 0.0)


def channel_for(*nodes: NodeState, cs_range: float = 50.0) -> Channel:
    return Channel({n.node_id: n.pos for n in nodes}, cs_range)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def radio(cfg):
    return RadioModel.from_config(cfg)


@pytest.fixture(scope="module")
def backoff(cfg):
    return BackoffPolicy(cfg.radio.csma)


# ----------------------------------------------------------------------------- config (§5.5)


def test_radio_config_defaults(cfg):
    """§5.5: sınıf bazlı macMinBE 2/3/5, macMaxBE 5, cs_range 50 m, B1-plain BE 3."""
    c = cfg.radio.csma
    assert c.mac_min_be == {"critical": 2, "regular": 3, "noncritical": 5}
    assert c.mac_max_be == 5 and c.plain_be == 3 and c.max_csma_backoffs == 4
    assert cfg.radio.cs_range_m == 50.0 and cfg.radio.channel_model == "local_contention"
    assert cfg.radio.phy_header_bytes == 6
    with pytest.raises(ConfigError, match="mac_min_be"):
        load_config(overrides={"radio.csma.mac_min_be": {"critical": 2, "regular": 3}})
    with pytest.raises(ConfigError, match="mac_max_be"):
        load_config(overrides={"radio.csma.mac_min_be": {"critical": 2, "regular": 3, "noncritical": 9}})


# ----------------------------------------------------------------------------- PHY (§5.5)


def test_fragmentation(radio):
    """§5.5: 90 B veri/çerçeve; 1024 B -> 12 parça; parça kaybı paketi düşürür (send_packet testlerinde)."""
    assert radio.fragment_payloads(64) == [64]
    assert radio.fragment_payloads(90) == [90]
    assert radio.fragment_payloads(91) == [90, 1]
    assert radio.n_fragments(1024) == 12 and radio.fragment_payloads(1024)[-1] == 1024 - 11 * 90
    assert radio.frame_overhead_bytes == 127 - 90
    with pytest.raises(ValueError):
        radio.fragment_payloads(0)


def test_transmission_time_1024B_single_hop_at_least_32_8_ms(radio):
    """§5.5: 1024 B tek atlama iletim süresi >= 1024·8/250000 = 32,8 ms (başlıklar süreyi uzatır)."""
    t = radio.transmission_time_s(1024)
    assert t >= 1024 * 8 / 250_000
    expected = sum((6 + 37 + p) * 8 / 250_000 for p in radio.fragment_payloads(1024))
    assert math.isclose(t, expected)
    assert math.isclose(radio.frame_airtime_s(90), 133 * 8 / 250_000)


def test_prr_logistic_monotone_and_clipped(radio, cfg):
    """§5.5: PRR mesafeyle azalır; d_mid'de prr_max/2; [prr_min, prr_max] içinde; menzilde tabana iner."""
    c = cfg.radio.prr
    assert math.isclose(radio.prr(0.0), c.prr_max, rel_tol=1e-6)
    assert math.isclose(radio.prr(c.d_mid_m), c.prr_max / 2)
    ds = np.linspace(0, 60, 61)
    values = [radio.prr(d) for d in ds]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(c.prr_min <= v <= c.prr_max for v in values)
    assert radio.prr(50.0) == c.prr_min and radio.prr(1e9) == c.prr_min


def test_first_order_energy_model(radio):
    """§5.5: E_tx = 50 nJ/bit · k + 100 pJ/bit/m² · k · d², E_rx = 50 nJ/bit · k."""
    bits = 133 * 8
    assert math.isclose(radio.tx_energy_J(bits, 50.0), 50e-9 * bits + 100e-12 * bits * 2500.0)
    assert math.isclose(radio.rx_energy_J(bits), 50e-9 * bits)
    assert radio.tx_energy_J(bits, 0.0) == radio.rx_energy_J(bits)


# ----------------------------------------------------------------------------- MAC: geri çekilme (§5.5, §5.7 π1/π2)


def test_class_based_backoff_critical_shorter_than_noncritical(backoff, cfg):
    """§5.5: kritik sınıfın ortalama geri çekilmesi kritik olmayandan kısa (BE 2 < 3 < 5); B1-plain'de eşit."""
    assert backoff.expected_backoff_s("critical") < backoff.expected_backoff_s("regular") < backoff.expected_backoff_s("noncritical")
    rng = np.random.default_rng(1)
    means = {c: np.mean([backoff.draw_backoff_s(rng, c) for _ in range(4000)]) for c in ("critical", "regular", "noncritical")}
    assert means["critical"] < means["regular"] < means["noncritical"]
    assert math.isclose(means["critical"], backoff.expected_backoff_s("critical"), rel_tol=0.15)
    slots = {backoff.draw_slots(rng, 2) for _ in range(500)}
    assert slots == {0, 1, 2, 3}
    plain = BackoffPolicy(cfg.radio.csma, vsdn_policy=False)
    assert plain.min_be("critical") == plain.min_be("noncritical") == 3


def test_contention_critical_backoff_shorter_on_average(cfg):
    """§5.5 bütünleşik: aynı anda başlayan kritik ve kritik olmayan göndericilerde kritik daha az bekler."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    crit_wait, non_wait = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        a, ra, b, rb = node(0, 0, 0), node(1, 10, 0), node(2, 20, 0, cls="noncritical"), node(3, 30, 0)
        ch = channel_for(a, ra, b, rb)
        res = run_mac_sessions({
            "crit": send_packet(radio, backoff, ch, rng, a, ra, 64, "critical", 5, now=0.0),
            "non": send_packet(radio, backoff, ch, rng, b, rb, 64, "noncritical", 1, now=0.0),
        })
        crit_wait.append(res["crit"].backoff_s)
        non_wait.append(res["non"].backoff_s)
    assert np.mean(crit_wait) < np.mean(non_wait)


# ----------------------------------------------------------------------------- MAC: taşıyıcı algılama (§5.5)


def _overlaps(intervals_a, intervals_b) -> bool:
    return any(a0 < b1 and b0 < a1 for a0, a1 in intervals_a for b0, b1 in intervals_b)


def test_nodes_within_cs_range_never_transmit_simultaneously():
    """§5.5: algılama menzilindeki iki düğüm aynı anda iletim yapamaz (12'şer çerçeve, aynı anda başlar)."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        a, ra, b, rb = node(0, 0, 0), node(1, 10, 0), node(2, 20, 0), node(3, 30, 0)
        ch = channel_for(a, ra, b, rb, cs_range=50.0)
        assert 2 in ch.cs_neighbors[0]
        res = run_mac_sessions({
            "a": send_packet(radio, backoff, ch, rng, a, ra, 1024, "regular", 3, now=0.0),
            "b": send_packet(radio, backoff, ch, rng, b, rb, 1024, "regular", 3, now=0.0),
        })
        assert res["a"].success and res["b"].success
        assert len(res["a"].tx_intervals) == len(res["b"].tx_intervals) == 12
        assert not _overlaps(res["a"].tx_intervals, res["b"].tx_intervals)
        assert res["a"].csma_busy_events + res["b"].csma_busy_events > 0  # çekişme gerçekten yaşandı
    assert math.isclose(ch.total_airtime_s, 2 * radio.transmission_time_s(1024))


def test_nodes_outside_cs_range_transmit_simultaneously():
    """§5.5 mekânsal yeniden kullanım: algılama menzili dışındaki düğümler aynı anda iletebilir."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(3)
    a, ra, b, rb = node(0, 0, 0), node(1, 10, 0), node(2, 200, 0), node(3, 210, 0)
    ch = channel_for(a, ra, b, rb, cs_range=50.0)
    assert 2 not in ch.cs_neighbors[0] and 1 in ch.cs_neighbors[0]
    res = run_mac_sessions({
        "a": send_packet(radio, backoff, ch, rng, a, ra, 1024, "regular", 3, now=0.0),
        "b": send_packet(radio, backoff, ch, rng, b, rb, 1024, "regular", 3, now=0.0),
    })
    assert res["a"].success and res["b"].success
    assert _overlaps(res["a"].tx_intervals, res["b"].tx_intervals)
    assert res["a"].csma_busy_events == res["b"].csma_busy_events == 0


def test_channel_reserve_rejects_when_busy():
    """§5.5 Channel: menzil içindeki rezerv meşgulken reddedilir, bitince kabul edilir."""
    a, b, far = node(0, 0, 0), node(1, 30, 0), node(2, 500, 0)
    ch = channel_for(a, b, far)
    assert ch.reserve(0, 0.0, 1.0, "critical")
    assert not ch.is_idle(1, 0.5) and not ch.reserve(1, 0.5, 1.5, "regular")
    assert ch.is_idle(2, 0.5) and ch.reserve(2, 0.5, 1.5, "regular")
    assert ch.busy_until(1, 0.5) == 1.0
    assert ch.idle_since(0) == ch.idle_since(1) == 1.0 and ch.idle_since(2) == 1.5  # duyulan son iletimin bitişi (AIFS)
    assert ch.is_idle(1, 1.0) and ch.reserve(1, 1.0, 2.0, "noncritical")
    share = ch.airtime_share()
    assert math.isclose(sum(share.values()), 1.0) and set(share) == {"critical", "regular", "noncritical"}


# ----------------------------------------------------------------------------- MAC: AIFS (§5.5 (b))


def test_aifs_config_and_plain_policy(cfg):
    """§5.5 (b): aifs_slots sınıf bazlı, kritik 0, negatif olamaz; B1-plain'de bütün sınıflar AIFS 0."""
    csma = cfg.radio.csma
    assert set(csma.aifs_slots) == set(CLASSES) and csma.aifs_slots["critical"] == 0
    policy, plain = BackoffPolicy(csma), BackoffPolicy(csma, vsdn_policy=False)
    for k in CLASSES:
        assert policy.aifs_slots(k) == csma.aifs_slots[k]
        assert math.isclose(policy.aifs_s(k), csma.aifs_slots[k] * csma.backoff_slot_s)
        assert plain.aifs_slots(k) == 0 and plain.aifs_s(k) == 0.0
    with pytest.raises(ConfigError, match="aifs_slots"):
        load_config(overrides={"radio.csma.aifs_slots": {"critical": 0, "regular": 1}})
    with pytest.raises(ConfigError, match="aifs_slots"):
        load_config(overrides={"radio.csma.aifs_slots": {"critical": 0, "regular": -1, "noncritical": 2}})


def test_aifs_config_matches_calibration_table(cfg):
    """§5.5 kalibrasyon: config'deki aifs_slots, mac_aifs_calibration.csv'de seçilen (selected) satırla aynı olmalı."""
    table = PROJECT_ROOT / cfg.experiments["mac_aifs_calibration"]["table"]
    if not table.exists():
        pytest.skip("kalibrasyon tablosu yok: python experiments/calibration/mac_aifs.py")
    with open(table, encoding="utf-8", newline="") as fh:
        selected = [row for row in csv.DictReader(fh) if row["selected"] == "True"]
    assert len(selected) == 1
    assert {k: int(selected[0][f"aifs_{k}"]) for k in CLASSES} == cfg.radio.csma.aifs_slots


def test_aifs_adds_fixed_wait_per_frame_on_idle_channel():
    """§5.5 (b): çekişme yokken AIFS her çerçeveye aifs_slots × slot ekler; AIFS 0 sınıfa ve B1-plain'e eklemez."""
    c = load_config(overrides={**PRR_ONE, "radio.csma.aifs_slots": {"critical": 0, "regular": 0, "noncritical": 5}})
    radio = RadioModel.from_config(c)

    def one_packet(backoff: BackoffPolicy, traffic_class: str, packet_bytes: int):
        s, r = node(0, 0, 0), node(1, 10, 0)
        gen = send_packet(radio, backoff, channel_for(s, r), np.random.default_rng(0), s, r, packet_bytes, traffic_class, 1, now=0.0)
        return run_mac_sessions({"p": gen})["p"]

    non = one_packet(BackoffPolicy(c.radio.csma), "noncritical", 1024)
    assert non.success and non.n_frames == 12
    assert math.isclose(non.aifs_s, 12 * 5 * c.radio.csma.backoff_slot_s) and non.backoff_s >= non.aifs_s
    assert math.isclose(non.t_end - non.t_start, non.backoff_s + non.transmission_s + non.processing_s)
    assert one_packet(BackoffPolicy(c.radio.csma), "critical", 64).aifs_s == 0.0
    assert one_packet(BackoffPolicy(c.radio.csma, vsdn_policy=False), "noncritical", 1024).aifs_s == 0.0


def _jammer(ch: Channel, node_id: int, intervals: list[tuple[float, float]]):
    """Yardımcı oturum: verilen [başlangıç, bitiş] aralıklarında kanalı meşgul eder."""
    for t0, t1 in intervals:
        t = yield t0
        assert ch.reserve(node_id, t, t1, "critical")
        yield t1
        ch.release(node_id)


def test_aifs_counts_from_idle_channel_and_restarts_when_interrupted():
    """§5.5 (b): AIFS kanal boşaldıktan sonra sayılır; AIFS dolmadan araya iletim girerse o iletimin bitişinden yeniden başlar."""
    c = load_config(overrides={
        **PRR_ONE,
        "radio.csma.mac_min_be": {"critical": 0, "regular": 0, "noncritical": 0},   # geri çekilme 0: zamanlama belirli
        "radio.csma.aifs_slots": {"critical": 0, "regular": 0, "noncritical": 5},
    })
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    aifs = 5 * c.radio.csma.backoff_slot_s  # 1,6 ms
    s, r, jam = node(0, 0, 0), node(1, 10, 0), node(2, 20, 0)
    ch = channel_for(s, r, jam)
    # kanal 0–10 ms meşgul; boşaldıktan 1 ms sonra (AIFS dolmadan) 11–15 ms yine meşgul
    res = run_mac_sessions({
        "jam": _jammer(ch, 2, [(0.0, 0.010), (0.011, 0.015)]),
        "p": send_packet(radio, backoff, ch, np.random.default_rng(0), s, r, 64, "noncritical", 1, now=0.0),
    })["p"]
    assert res.success and res.csma_busy_events == 0
    assert math.isclose(res.tx_intervals[0][0], 0.015 + aifs)  # son boşalma + AIFS
    assert math.isclose(res.aifs_s, 0.015 + aifs) and math.isclose(res.backoff_s, res.aifs_s)


def test_next_packet_ready_follows_single_processing_delay(cfg):
    """§5.5 zamanlama kuralı: sıradaki paket = son MAC etkinliği + radio.processing_delay_s (config); başarı/kayıp aynı.

    Doygun gönderici ardışık paketlere tam bu anda başlar; işlem süresi için ikinci bir sabit yoktur
    (processing_delay_s değişince hem bileşen hem kural değişir).
    """
    for overrides in (PRR_ONE, PRR_ZERO):
        for proc in (0.001, 0.004):
            c = load_config(overrides={**overrides, "radio.processing_delay_s": proc})
            radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
            s, r = node(0, 0, 0, energy=math.inf), node(1, 10, 0, energy=math.inf)
            log: list = []
            run_mac_sessions(
                {"sat": saturated_sender(radio, backoff, channel_for(s, r), np.random.default_rng(0), s, r, "regular", 1, lambda: 64, log)},
                until_s=0.2,
            )
            assert len(log) >= 5 and all(p.success for p in log) == (overrides is PRR_ONE)
            for prev, nxt in zip(log, log[1:]):
                assert math.isclose(radio.next_packet_ready_s(prev), prev.mac_end_s + proc)
                assert math.isclose(nxt.t_start, radio.next_packet_ready_s(prev))
                if prev.success:  # başarıda t_end işlem süresini zaten içerir: kural t_end'i verir
                    assert math.isclose(prev.processing_s, proc) and math.isclose(radio.next_packet_ready_s(prev), prev.t_end)
                else:             # kayıpta t_end kaybın anıdır: kural işlem süresini ekler
                    assert prev.processing_s == 0.0 and math.isclose(radio.next_packet_ready_s(prev), prev.t_end + proc)


def _saturated_shares(aifs_slots: dict[str, int], seed: int, duration_s: float = 5.0) -> dict[str, float]:
    """Sınıf başına bir doygun gönderici, hepsi cs menzilinde; yalnızca AIFS farklı (aynı BE 3, 90 B paket, π10 = 3)."""
    c = load_config(overrides={
        **PRR_ONE,
        "radio.csma.mac_min_be": {k: 3 for k in CLASSES},
        "radio.csma.aifs_slots": aifs_slots,
    })
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(seed)
    senders = [node(2 * i, 0, 10 * i, energy=math.inf, cls=k) for i, k in enumerate(CLASSES)]
    receivers = [node(2 * i + 1, 10, 10 * i, energy=math.inf, cls=k) for i, k in enumerate(CLASSES)]
    ch = channel_for(*senders, *receivers)
    assert all({o.node_id for o in senders if o is not s} <= ch.cs_neighbors[s.node_id] for s in senders)
    run_mac_sessions(
        {k: saturated_sender(radio, backoff, ch, rng, s, r, k, 3, lambda: 90, []) for k, s, r in zip(CLASSES, senders, receivers)},
        until_s=duration_s,
    )
    share = ch.airtime_share()
    assert math.isclose(sum(share.values()), 1.0)
    return share


def test_larger_aifs_gets_smaller_saturated_airtime_share():
    """§5.5 (b): doygunlukta AIFS'si büyük olan sınıfın airtime payı, AIFS'si küçük olanınkinden düşüktür.

    AIFS'nin etkisi yalıtılır (BE, paket boyutu, π10 eşit). AIFS hepsi 0 iken paylar eşit (≈ 1/3);
    AIFS 0 / 4 / 8 ile pay sırası AIFS sırasının tersi; AIFS'ler sınıflar arasında ters çevrilince sıra da döner.
    """
    for seed in range(3):
        equal = _saturated_shares({"critical": 0, "regular": 0, "noncritical": 0}, seed)
        assert all(abs(v - 1 / 3) < 0.06 for v in equal.values())
        share = _saturated_shares({"critical": 0, "regular": 4, "noncritical": 8}, seed)
        assert share["critical"] > share["regular"] + 0.15 and share["regular"] > share["noncritical"] + 0.10
        flipped = _saturated_shares({"critical": 8, "regular": 4, "noncritical": 0}, seed)
        assert flipped["noncritical"] > flipped["regular"] + 0.15 and flipped["regular"] > flipped["critical"] + 0.10


# ----------------------------------------------------------------------------- bütünleşik tek atlama (§5.5, §5.7 π10)


def test_successful_hop_components_and_energy():
    """§5.5: gecikme bileşenleri tutarlı; enerji gönderici/alıcıdan düşer; kuyruk bekleme now − enqueued_at."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(0)
    s, r = node(0, 0, 0), node(1, 40, 0)
    ch = channel_for(s, r)
    res = run_mac_sessions({"p": send_packet(radio, backoff, ch, rng, s, r, 1024, "noncritical", 1, now=5.0, enqueued_at=4.9)})["p"]
    assert res.success and res.loss_reason is None and res.frame_attempts == 12
    assert math.isclose(res.queue_s, 0.1)
    assert math.isclose(res.transmission_s, radio.transmission_time_s(1024)) and res.transmission_s >= 0.0328
    assert math.isclose(res.processing_s, c.radio.processing_delay_s)
    assert math.isclose(res.t_end - res.t_start, res.backoff_s + res.transmission_s + res.processing_s)
    assert math.isclose(res.hop_delay_s, res.queue_s + res.t_end - res.t_start)
    assert set(res.components()) == {"queue", "backoff", "transmission", "processing"}
    bits = [radio.frame_bytes(p) * 8 for p in radio.fragment_payloads(1024)]
    assert math.isclose(res.energy_tx_J, sum(radio.tx_energy_J(b, 40.0) for b in bits))
    assert math.isclose(res.energy_rx_J, sum(radio.rx_energy_J(b) for b in bits))
    assert math.isclose(s.energy_J, 2.0 - res.energy_tx_J) and math.isclose(r.energy_J, 2.0 - res.energy_rx_J)


def test_retry_limit_drops_whole_packet():
    """§5.7 π10: parça limit+1 denemede teslim edilemezse paket kaybolur; her deneme enerji harcar."""
    c = load_config(overrides=PRR_ZERO)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(0)
    s, r = node(0, 0, 0), node(1, 10, 0)
    res = run_mac_sessions({"p": send_packet(radio, backoff, channel_for(s, r), rng, s, r, 1024, "regular", 3, now=0.0)})["p"]
    assert not res.success and res.loss_reason == LOSS_RETRY_LIMIT
    assert res.frame_attempts == 3 + 1 and res.n_frames == 12
    assert res.energy_tx_J > 0 and s.energy_J < 2.0


def test_csma_failure_counts_as_retry():
    """§5.5: kanal sürekli meşgulse macMaxCSMABackoffs aşılır; π10 = 0 ile paket kanal erişim hatasıyla düşer."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(0)
    s, r, jammer = node(0, 0, 0), node(1, 10, 0), node(2, 20, 0)
    ch = channel_for(s, r, jammer)
    assert ch.reserve(2, 0.0, 1e6, "noncritical")
    res = run_mac_sessions({"p": send_packet(radio, backoff, ch, rng, s, r, 64, "critical", 0, now=0.0)})["p"]
    assert not res.success and res.loss_reason == LOSS_CSMA_FAILURE
    assert res.csma_busy_events == c.radio.csma.max_csma_backoffs + 1 and res.frame_attempts == 0


def test_sender_dies_mid_packet():
    """§5.5: enerjisi biten gönderici ölür; kalan parçalar iletilemez, paket 'sender_dead' ile kaybolur."""
    c = load_config(overrides=PRR_ONE)
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)
    rng = np.random.default_rng(0)
    per_frame = radio.tx_energy_J(radio.frame_bytes(90) * 8, 10.0)
    s, r = node(0, 0, 0, energy=2.5 * per_frame), node(1, 10, 0)
    res = run_mac_sessions({"p": send_packet(radio, backoff, channel_for(s, r), rng, s, r, 1024, "regular", 3, now=0.0)})["p"]
    assert not res.success and res.loss_reason == LOSS_SENDER_DEAD
    assert res.frame_attempts == 3 and not s.alive and s.energy_J == 0.0


def test_run_mac_sessions_is_reproducible():
    """§3: aynı seed aynı zamanlama."""
    c = load_config()
    radio, backoff = RadioModel.from_config(c), BackoffPolicy(c.radio.csma)

    def run(seed):
        rng = np.random.default_rng(seed)
        a, ra, b, rb = node(0, 0, 0), node(1, 10, 0), node(2, 20, 0), node(3, 30, 0)
        ch = channel_for(a, ra, b, rb)
        res = run_mac_sessions({
            "a": send_packet(radio, backoff, ch, rng, a, ra, 256, "regular", 3, now=0.0),
            "b": send_packet(radio, backoff, ch, rng, b, rb, 64, "critical", 5, now=0.0),
        })
        return [(k, v.success, v.t_end, v.backoff_s) for k, v in sorted(res.items())]

    assert run(11) == run(11) and run(11) != run(12)

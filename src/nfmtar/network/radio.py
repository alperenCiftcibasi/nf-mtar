"""IEEE 802.15.4 iletim, yerel kanal çekişmesi, sınıf bazlı CSMA/CA, PRR ve enerji modeli (CLAUDE.md §5.5, §5.7).

Uygulanan kararlar:
- §5.5 çerçeveleme: 250 kbit/s; çerçeve <= 127 B PSDU + PHY başlığı; çerçeve başına 90 B veri. Büyük paketler
  parçalanır; bir parça kaybolursa paket kaybolur (fragment_loss_drops_packet).
- §5.5 gecikme: atlama başına gecikme = kuyruk bekleme + geri çekilme + iletim + işlem; bileşenler `HopResult`'ta
  ayrı tutulur (§5.13 gecikme bileşenleri).
- §5.5 paylaşılan kanal, yerel çekişme: kanal ağ genelinde tek bir havuz değildir. Taşıyıcı algılama menzilindeki
  (`cs_range_m`) başka bir düğüm iletim yaparken bir düğüm iletime başlayamaz (CCA meşgul -> yeni geri çekilme).
  Gizli terminal çarpışmaları modellenmez; kayıplar PRR ile temsil edilir (COOJA UDGM benzeri basitleştirme).
- §5.5 / §5.7 π1, π2 MAC'te: 802.11e EDCA benzeri iki mekanizma.
  (a) Sınıf bazlı geri çekilme: macMinBE kritik 2, düzenli 3, kritik olmayan 5; macMaxBE 5.
  (b) Sınıf bazlı AIFS (`aifs_slots`, slot cinsinden; kritik 0): AIFS'si > 0 olan sınıf, her geri çekilme
      sayımından önce kanalın boşalmasını bekler ve kanal AIFS boyunca kesintisiz boş kaldıktan sonra sayıma
      başlar; araya iletim girerse bekleme o iletimin bitişinden yeniden başlar. AIFS = 0 ise bu adım hiç
      yoktur: düğüm standart 802.15.4 unslotted CSMA/CA gibi kanalı dinlemeden geri çekilir, sonra CCA yapar.
      Değerler experiments/calibration/mac_aifs.py ile kalibre edilir (π1 hedefine L1 uzaklığı en küçük çift).
  B1-plain'de bütün sınıflar BE 3, AIFS 0. Sınıf bazında ulaşılan airtime payı `Channel` tarafından ölçülür
  (§5.13, π1 hedefiyle karşılaştırma).
- §5.7 π10: parça başına yeniden iletim limiti; aşılınca parça ve dolayısıyla paket kaybolur. Kanal erişim
  hatası (macMaxCSMABackoffs aşıldı) bir yeniden deneme olarak sayılır.
- §5.5 link kalitesi: lojistik PRR(d) = prr_max / (1 + exp(k·(d − d_mid))), [prr_min, prr_max]'a kırpılır.
- §5.5 enerji: birinci dereceden radyo modeli E_tx(k, d) = E_elec·k + ε_amp·k·d², E_rx(k) = E_elec·k;
  her çerçeve denemesinde gönderici ve alıcı `NodeState.consume` ile düşer, enerjisi biten düğüm ölür.
  Sınırlılıklar (raporda): ACK çerçeveleri, CCA süresi ve boşta dinleme enerjisi modellenmez.
- §5.7 π9: kuyruk kapasitesi `PacketQueue`'dadır (graph.py); bu modül yalnızca MAC/PHY'yi modeller.
- §5.5 zamanlama kuralı (tek kaynak): bir düğüm sıradaki pakete, önceki paketin son MAC etkinliğinden
  `radio.processing_delay_s` (config) sonra başlar — `RadioModel.next_packet_ready_s`. Aynı değer atlama
  gecikmesinin işlem bileşenidir. `saturated_sender` (AIFS kalibrasyonu) ve simulation.py'deki düğüm servis
  döngüsü yalnızca bu fonksiyonu kullanır; kural değişirse kalibrasyon yeniden koşturulmalıdır.

Olay akışı: `send_packet` bir üreteçtir. Her `t_wake = yield t_wake` "beni t_wake anında uyandır" demektir;
simulation.py ve buradaki `run_mac_sessions` mini sürücüsü üreteçleri zaman sırasıyla ilerletir. Bu sayede
farklı düğümlerin geri çekilme / iletim adımları birbirine karışık (interleaved) işlenir ve taşıyıcı algılama
doğru çalışır. `saturated_sender` kuyruğu hiç boşalmayan bir kaynaktır (§5.5 AIFS kalibrasyonu ve testler).
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Mapping

import numpy as np

from nfmtar.config import Config, CsmaConfig, PrrConfig
from nfmtar.network.graph import NodeState, PhysicalNetwork

LOSS_RETRY_LIMIT = "retry_limit"        # π10 aşıldı (parça ACK alamadı)
LOSS_CSMA_FAILURE = "csma_failure"      # kanal erişim hatası π10 kadar tekrarlandı
LOSS_SENDER_DEAD = "sender_dead"        # §5.5 gönderici iletim sırasında öldü
LOSS_RECEIVER_DEAD = "receiver_dead"    # §5.5 alıcı ölü


# --------------------------------------------------------------------------- #
# PHY: çerçeveleme, PRR, enerji (§5.5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RadioModel:
    """Fiziksel katman sabitleri ve formülleri (§5.5): çerçeve süreleri, PRR(d), E_tx / E_rx."""

    bitrate_bps: int
    max_frame_bytes: int
    payload_per_frame_bytes: int
    phy_header_bytes: int
    prr_cfg: PrrConfig
    e_elec_J_per_bit: float
    eps_amp_J_per_bit_m2: float
    processing_delay_s: float

    @classmethod
    def from_config(cls, cfg: Config) -> "RadioModel":
        """configs/default.yaml radio + energy bölümlerinden kurar (§5.5; birimler nJ/pJ -> J)."""
        r = cfg.radio
        return cls(
            bitrate_bps=r.bitrate_bps,
            max_frame_bytes=r.max_frame_bytes,
            payload_per_frame_bytes=r.payload_per_frame_bytes,
            phy_header_bytes=r.phy_header_bytes,
            prr_cfg=r.prr,
            e_elec_J_per_bit=cfg.energy.e_elec_nJ_per_bit * 1e-9,
            eps_amp_J_per_bit_m2=cfg.energy.eps_amp_pJ_per_bit_m2 * 1e-12,
            processing_delay_s=r.processing_delay_s,
        )

    @property
    def frame_overhead_bytes(self) -> int:
        """MAC başlığı + FCS: 127 B PSDU − 90 B veri = 37 B (§5.5)."""
        return self.max_frame_bytes - self.payload_per_frame_bytes

    def fragment_payloads(self, packet_bytes: int) -> list[int]:
        """Paketi çerçeve başına veri sınırına göre parçalara böler (§5.5); son parça kalan baytları taşır."""
        if packet_bytes <= 0:
            raise ValueError("paket boyutu pozitif olmalı")
        full, rem = divmod(packet_bytes, self.payload_per_frame_bytes)
        return [self.payload_per_frame_bytes] * full + ([rem] if rem else [])

    def n_fragments(self, packet_bytes: int) -> int:
        """Parça sayısı = ceil(paket / çerçeve_veri) (§5.5)."""
        return len(self.fragment_payloads(packet_bytes))

    def frame_bytes(self, payload_bytes: int) -> int:
        """Havada giden bayt: PHY başlığı + MAC yükü (başlık + veri + FCS) (§5.5)."""
        return self.phy_header_bytes + self.frame_overhead_bytes + payload_bytes

    def frame_airtime_s(self, payload_bytes: int) -> float:
        """Bir çerçevenin iletim süresi = çerçeve_bit / 250000 (§5.5)."""
        return self.frame_bytes(payload_bytes) * 8 / self.bitrate_bps

    def transmission_time_s(self, packet_bytes: int) -> float:
        """Paketin tek atlamada saf iletim süresi = Σ parça çerçeve süreleri (§5.5 'parça_sayısı × çerçeve_bit/250000')."""
        return sum(self.frame_airtime_s(p) for p in self.fragment_payloads(packet_bytes))

    def prr(self, distance_m: float) -> float:
        """Mesafeye bağlı lojistik paket alma oranı, [prr_min, prr_max] (§5.5; §5.9 PRR tabanı)."""
        c = self.prr_cfg
        if c.model != "logistic":
            raise ValueError(f"desteklenmeyen PRR modeli {c.model!r} (§5.5: logistic)")
        z = c.steepness_per_m * (distance_m - c.d_mid_m)
        value = c.prr_max / (1.0 + math.exp(min(z, 700.0)))
        return min(c.prr_max, max(c.prr_min, value))

    def tx_energy_J(self, bits: int, distance_m: float) -> float:
        """Birinci dereceden radyo modeli, gönderim: E_elec·k + ε_amp·k·d² (§5.5)."""
        return self.e_elec_J_per_bit * bits + self.eps_amp_J_per_bit_m2 * bits * distance_m**2

    def rx_energy_J(self, bits: int) -> float:
        """Birinci dereceden radyo modeli, alım: E_elec·k (§5.5)."""
        return self.e_elec_J_per_bit * bits

    def next_packet_ready_s(self, result: "HopResult") -> float:
        """§5.5 zamanlama kuralı: düğüm sıradaki pakete, önceki paketin son MAC etkinliğinden processing_delay_s sonra başlar.

        Başarıda bu an HopResult.t_end'dir (send_packet işlem süresini zaten eklemiştir); kayıpta t_end kaybın anıdır
        ve işlem süresi burada eklenir. Kural başarı/kayıp ayrımı yapmaz: her paket, bitişinden sonra düğümü
        processing_delay_s kadar meşgul eder. Tek kaynak: doygun gönderici (kalibrasyon) ve simulation.py düğüm
        servis döngüsü yalnızca bu fonksiyonu kullanır.
        """
        return result.mac_end_s + self.processing_delay_s


# --------------------------------------------------------------------------- #
# MAC: sınıf bazlı geri çekilme (§5.5, §5.7 π1/π2) ve yerel taşıyıcı algılama (§5.5)
# --------------------------------------------------------------------------- #


class BackoffPolicy:
    """Sınıf bazlı CSMA/CA parametreleri (§5.5): (a) geri çekilme üssü ve (b) AIFS.

    (a) BE_min sınıfa göre, BE her meşgul CCA'da +1 (macMaxBE'ye kadar); süre = U{0, …, 2^BE − 1} × slot (320 µs).
    (b) AIFS: sınıfa göre sabit ek bekleme (slot); kritik 0, diğerleri kalibrasyonla (config `aifs_slots`).
    vsdn_policy=False (B1-plain, §5.12): bütün sınıflar plain_be (3) ve AIFS 0 kullanır, öncelik yoktur.
    """

    def __init__(self, csma: CsmaConfig, vsdn_policy: bool = True) -> None:
        self.csma = csma
        self.vsdn_policy = vsdn_policy

    @property
    def max_be(self) -> int:
        return self.csma.mac_max_be

    @property
    def max_csma_backoffs(self) -> int:
        return self.csma.max_csma_backoffs

    @property
    def slot_s(self) -> float:
        return self.csma.backoff_slot_s

    def min_be(self, traffic_class: str) -> int:
        """macMinBE: sınıfa göre (π2 sırası) veya B1-plain'de sabit (§5.5)."""
        if not self.vsdn_policy:
            return self.csma.plain_be
        return self.csma.mac_min_be[traffic_class]

    def aifs_slots(self, traffic_class: str) -> int:
        """AIFS (slot): sınıfa göre (kritik 0) veya B1-plain'de bütün sınıflar için 0 (§5.5 (b))."""
        if not self.vsdn_policy:
            return 0
        return self.csma.aifs_slots[traffic_class]

    def aifs_s(self, traffic_class: str) -> float:
        """AIFS süresi (s) = aifs_slots × slot (§5.5 (b))."""
        return self.aifs_slots(traffic_class) * self.slot_s

    def draw_slots(self, rng: np.random.Generator, be: int) -> int:
        """U{0, …, 2^BE − 1} slot (§5.5)."""
        return int(rng.integers(0, 2**be))

    def draw_backoff_s(self, rng: np.random.Generator, traffic_class: str, be: int | None = None) -> float:
        """Bir geri çekilme süresi (s); be verilmezse sınıfın macMinBE'si kullanılır (§5.5)."""
        return self.draw_slots(rng, self.min_be(traffic_class) if be is None else be) * self.slot_s

    def expected_backoff_s(self, traffic_class: str) -> float:
        """İlk tur beklenen geri çekilme = (2^BE_min − 1)/2 × slot; kritik < düzenli < kritik olmayan (§5.5)."""
        return (2 ** self.min_be(traffic_class) - 1) / 2.0 * self.slot_s


class Channel:
    """Yerel çekişmeli paylaşılan kanal (§5.5): taşıyıcı algılama menzilindeki iletimler kanalı meşgul eder.

    `reserve` yalnızca kanal boşsa kabul edilir; böylece algılama menzilindeki iki düğümün iletimleri hiçbir
    zaman örtüşmez. Menzil dışındaki düğümler aynı anda iletebilir (mekânsal yeniden kullanım).
    Sınıf bazında toplam airtime ölçülür (§5.13 airtime payı; π1 hedefiyle karşılaştırılır).
    """

    def __init__(self, positions: Mapping[int, tuple[float, float]], cs_range_m: float) -> None:
        self.cs_range_m = cs_range_m
        ids = sorted(positions)
        pos = np.array([positions[i] for i in ids], dtype=float)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        self.cs_neighbors: dict[int, frozenset[int]] = {}
        for a, u in enumerate(ids):
            self.cs_neighbors[u] = frozenset(ids[b] for b in np.flatnonzero(dist[a] <= cs_range_m) if ids[b] != u)
        self._active: dict[int, tuple[float, float, str]] = {}
        # AIFS için (§5.5 (b)): her düğümün algılama menzilinde duyduğu son iletimin bitiş anı
        self._heard_until: dict[int, float] = {u: -math.inf for u in ids}
        self.airtime_s_by_class: dict[str, float] = defaultdict(float)
        self.busy_cca_count = 0
        self.reservation_count = 0

    @classmethod
    def from_network(cls, net: PhysicalNetwork, cs_range_m: float) -> "Channel":
        """Fiziksel ağın düğüm konumlarından kurar (§5.5 cs_range varsayılanı = radyo menzili)."""
        return cls({i: st.pos for i, st in net.nodes.items()}, cs_range_m)

    def _prune(self, t: float) -> None:
        for node, (_, t_end, _) in list(self._active.items()):
            if t_end <= t:
                del self._active[node]

    def is_idle(self, node_id: int, t: float) -> bool:
        """CCA: düğümün kendisi veya algılama menzilindeki bir komşusu t anında iletimde mi (§5.5)."""
        self._prune(t)
        if node_id in self._active:
            return False
        return not any(n in self._active for n in self.cs_neighbors[node_id])

    def busy_until(self, node_id: int, t: float) -> float:
        """Algılama menzilindeki iletimlerin en geç bitiş zamanı (boşsa t)."""
        self._prune(t)
        ends = [self._active[n][1] for n in self.cs_neighbors[node_id] | {node_id} if n in self._active]
        return max(ends, default=t)

    def idle_since(self, node_id: int) -> float:
        """Düğümün (kendi iletimi dahil) duyduğu son iletimin bitiş anı; hiç iletim duymadıysa −inf (§5.5 (b) AIFS).

        Kanal bu andan beri boştur. Süren bir iletim varsa değer gelecektedir: kanal o anda boşalacaktır.
        """
        return self._heard_until[node_id]

    def reserve(self, node_id: int, t_start: float, t_end: float, traffic_class: str) -> bool:
        """İletimi kaydeder; kanal meşgulse False (çağıran yeniden geri çekilir). Airtime sınıfa yazılır (§5.13)."""
        if not self.is_idle(node_id, t_start):
            self.busy_cca_count += 1
            return False
        self._active[node_id] = (t_start, t_end, traffic_class)
        for n in (node_id, *self.cs_neighbors[node_id]):  # bu iletimi duyan düğümler (AIFS için, §5.5 (b))
            self._heard_until[n] = max(self._heard_until[n], t_end)
        self.airtime_s_by_class[traffic_class] += t_end - t_start
        self.reservation_count += 1
        return True

    def release(self, node_id: int) -> None:
        """İletim bitti; kaydı kaldırır (zaman bazlı budama da aynı sonucu verir)."""
        self._active.pop(node_id, None)

    @property
    def total_airtime_s(self) -> float:
        return sum(self.airtime_s_by_class.values())

    def airtime_share(self) -> dict[str, float]:
        """Sınıf bazında ulaşılan airtime payı (toplam = 1); π1 hedefi 0.80 / 0.15 / 0.05 ile karşılaştırılır (§5.13)."""
        total = self.total_airtime_s
        if total <= 0:
            return {k: 0.0 for k in self.airtime_s_by_class}
        return {k: v / total for k, v in self.airtime_s_by_class.items()}


# --------------------------------------------------------------------------- #
# Tek atlama paket iletimi (§5.5 gecikme bileşenleri, §5.7 π10)
# --------------------------------------------------------------------------- #


@dataclass
class HopResult:
    """Bir paketin tek atlama iletim sonucu: başarı/kayıp nedeni, gecikme bileşenleri, enerji (§5.5, §5.13).

    backoff_s kanal erişim beklemesinin tamamıdır: rastgele geri çekilme + AIFS. aifs_s bunun AIFS'den gelen
    kısmıdır (kanalın boşalmasını bekleme + AIFS, §5.5 (b)); AIFS'si 0 olan sınıfta 0'dır.
    """

    sender: int
    receiver: int
    traffic_class: str
    packet_bytes: int
    n_frames: int
    t_start: float
    success: bool = False
    loss_reason: str | None = None
    frame_attempts: int = 0
    csma_busy_events: int = 0
    t_end: float = 0.0
    queue_s: float = 0.0
    backoff_s: float = 0.0
    aifs_s: float = 0.0
    transmission_s: float = 0.0
    processing_s: float = 0.0
    energy_tx_J: float = 0.0
    energy_rx_J: float = 0.0
    tx_intervals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def hop_delay_s(self) -> float:
        """Atlama gecikmesi = kuyruk + geri çekilme + iletim + işlem (§5.5)."""
        return self.queue_s + self.backoff_s + self.transmission_s + self.processing_s

    @property
    def mac_end_s(self) -> float:
        """Son MAC etkinliğinin (çerçeve veya geri çekilme) bitiş anı = t_end − işlem süresi; kayıpta t_end (§5.5 zamanlama kuralı)."""
        return self.t_end - self.processing_s

    def components(self) -> dict[str, float]:
        """Paket logu için gecikme bileşenleri (§5.13)."""
        return {
            "queue": self.queue_s,
            "backoff": self.backoff_s,
            "transmission": self.transmission_s,
            "processing": self.processing_s,
        }


def send_packet(
    radio: RadioModel,
    backoff: BackoffPolicy,
    channel: Channel,
    rng: np.random.Generator,
    sender: NodeState,
    receiver: NodeState,
    packet_bytes: int,
    traffic_class: str,
    retransmit_limit: int,
    now: float,
    enqueued_at: float | None = None,
) -> Generator[float, float, HopResult]:
    """Bir paketi tek atlamada iletir: parçalama, sınıf bazlı CSMA/CA, PRR kaybı, π10 yeniden deneme, enerji (§5.5, §5.7).

    Üreteç protokolü: `yield t` -> çağıran, simülasyon saati t olduğunda `send(t)` ile devam ettirir.
    Her parça için: [AIFS > 0 ise kanal AIFS boyunca boş kalana kadar bekle] -> geri çekilme -> CCA (kanal meşgulse
    NB += 1, BE += 1, baştan; NB > macMaxCSMABackoffs ise kanal erişim hatası) -> iletim (kanal rezervi, enerji)
    -> PRR ile teslim; teslim yoksa π10'a kadar tekrar.
    Dönüş: HopResult (gecikme bileşenleri ayrı; kuyruk bekleme = now − enqueued_at).
    """
    payloads = radio.fragment_payloads(packet_bytes)
    distance = math.dist(sender.pos, receiver.pos)
    aifs = backoff.aifs_s(traffic_class)
    result = HopResult(
        sender=sender.node_id,
        receiver=receiver.node_id,
        traffic_class=traffic_class,
        packet_bytes=packet_bytes,
        n_frames=len(payloads),
        t_start=now,
        queue_s=0.0 if enqueued_at is None else max(0.0, now - enqueued_at),
    )
    t = now

    def fail(reason: str) -> HopResult:
        result.success = False
        result.loss_reason = reason
        result.t_end = t
        return result

    for payload in payloads:
        retries = 0
        while True:
            if not sender.alive:
                return fail(LOSS_SENDER_DEAD)
            if not receiver.alive:
                return fail(LOSS_RECEIVER_DEAD)

            # --- CSMA/CA kanal erişimi (§5.5): [AIFS] + geri çekilme + CCA ---
            nb = 0
            be = backoff.min_be(traffic_class)
            got_channel = False
            while True:
                # (b) AIFS: kanal boşaldıktan (t_free) sonra AIFS boyunca kesintisiz boş kalmalı. Uyanınca araya
                # iletim girdiği görülürse (idle_since ilerlemiş) bekleme o iletimin bitişinden yeniden başlar.
                while aifs > 0:
                    t_free = max(t, channel.idle_since(sender.node_id))
                    t_wake = yield t_free + aifs
                    result.aifs_s += t_wake - t
                    result.backoff_s += t_wake - t
                    t = t_wake
                    if channel.idle_since(sender.node_id) <= t_free:
                        break
                # (a) rastgele geri çekilme, ardından CCA
                delay = backoff.draw_slots(rng, be) * backoff.slot_s
                result.backoff_s += delay
                t = yield t + delay
                if channel.is_idle(sender.node_id, t):
                    got_channel = True
                    break
                result.csma_busy_events += 1
                channel.busy_cca_count += 1
                nb += 1
                be = min(be + 1, backoff.max_be)
                if nb > backoff.max_csma_backoffs:
                    break

            if got_channel:
                # --- iletim (§5.5): kanal rezervi, enerji, PRR ---
                airtime = radio.frame_airtime_s(payload)
                bits = radio.frame_bytes(payload) * 8
                if not channel.reserve(sender.node_id, t, t + airtime, traffic_class):
                    raise RuntimeError("CCA boş dedi ama rezerv reddedildi; kanal durumu tutarsız")
                result.frame_attempts += 1
                result.tx_intervals.append((t, t + airtime))
                e_tx = radio.tx_energy_J(bits, distance)
                sender.consume(e_tx, t)
                result.energy_tx_J += e_tx
                e_rx = radio.rx_energy_J(bits)
                receiver.consume(e_rx, t)
                result.energy_rx_J += e_rx
                result.transmission_s += airtime
                t = yield t + airtime
                channel.release(sender.node_id)
                if rng.random() < radio.prr(distance):
                    break  # parça teslim edildi (ACK); sıradaki parçaya geç
                reason = LOSS_RETRY_LIMIT
            else:
                reason = LOSS_CSMA_FAILURE

            retries += 1
            if retries > retransmit_limit:  # π10: toplam deneme = limit + 1
                return fail(reason)

    # --- alıcıda işlem süresi (§5.5) ---
    result.processing_s = radio.processing_delay_s
    t += result.processing_s
    result.success = True
    result.t_end = t
    return result


def saturated_sender(
    radio: RadioModel,
    backoff: BackoffPolicy,
    channel: Channel,
    rng: np.random.Generator,
    sender: NodeState,
    receiver: NodeState,
    traffic_class: str,
    retransmit_limit: int,
    next_packet_bytes: Callable[[], int],
    log: list[HopResult],
) -> Generator[float, float, None]:
    """Doygun kaynak: kuyruğu hiç boşalmayan gönderici (§5.5 AIFS kalibrasyonu ve testler).

    Sıradaki paket `radio.next_packet_ready_s` anında (§5.5 zamanlama kuralı: önceki paketin son MAC etkinliği +
    config `radio.processing_delay_s`) çekişmeye girer. Yani paketler arasında işlem süresi kadar boşluk vardır;
    aynı paketin parçaları arasında yoktur. Ölçülen paylar bu kurala duyarlıdır: boşluksuz modelde yüksek AIFS'li
    sınıflar tamamen aç kalır (bkz. mac_aifs.py). simulation.py'deki düğüm servis döngüsü aynı fonksiyonu kullanır.
    Her paketin HopResult'ı `log`'a eklenir. Üreteç kendiliğinden bitmez; `run_mac_sessions(until_s=...)` keser.
    Gönderici veya alıcı ölürse durur.
    """
    t = 0.0
    while sender.alive and receiver.alive:
        result = yield from send_packet(
            radio, backoff, channel, rng, sender, receiver, next_packet_bytes(), traffic_class, retransmit_limit, now=t
        )
        log.append(result)
        t = radio.next_packet_ready_s(result)


def run_mac_sessions(
    sessions: Mapping[str, Generator[float, float, Any]],
    until_s: float | None = None,
) -> dict[str, Any]:
    """Mini olay sürücüsü: üreteçleri zaman sırasıyla (eşitlikte başlatma sırasıyla) ilerletir.

    Testler ve bağımsız kullanım içindir; simulation.py aynı protokolü kendi olay kuyruğunda (heapq) uygular (§3).
    until_s verilirse bu andan sonraki olaylar işlenmez; bitmemiş üreteçler yarıda kalır ve sonuç sözlüğünde yer
    almaz (doygun kaynaklar, §5.5 kalibrasyon). Dönüş: oturum adı -> üretecin dönüş değeri (send_packet: HopResult).
    """
    heap: list[tuple[float, int, str]] = []
    gens = dict(sessions)
    results: dict[str, Any] = {}
    seq = 0
    for name, gen in gens.items():
        try:
            t_next = next(gen)
        except StopIteration as stop:
            results[name] = stop.value
            continue
        heapq.heappush(heap, (t_next, seq, name))
        seq += 1
    while heap:
        t, _, name = heapq.heappop(heap)
        if until_s is not None and t > until_s:
            break
        try:
            t_next = gens[name].send(t)
        except StopIteration as stop:
            results[name] = stop.value
            continue
        if t_next < t:
            raise RuntimeError(f"{name}: zaman geri gidemez ({t_next} < {t})")
        heapq.heappush(heap, (t_next, seq, name))
        seq += 1
    return results

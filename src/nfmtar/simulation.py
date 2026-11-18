"""Olay güdümlü simülasyon çekirdeği: trafik izi -> kuyruk -> atlama atlama MAC iletimi -> teslim / kayıp
(CLAUDE.md §5.10, §5.10a; dayandığı kararlar §5.4–§5.7, §5.9, §5.12).

Uygulanan kararlar:
- §5.10a olay güdümlü çekirdek: tek `heapq`; anahtar (zaman, olay türü önceliği, sıra no) -> eşzamanlı olaylarda
  deterministik sıra. Öncelik sırası: varış (izden kaynağa veya aktarıcıya giriş) < yol güncellemesi < MAC adımı
  < servis başlangıcı < kontrol (ısınma sonu, bitiş).
- §5.10a MAC nedenselliği: `radio.send_packet` üreteçleri `radio.run_mac_sessions` ile AYNI protokolle sürülür:
  her `yield t` bir uyanma olayıdır, üreteç yalnızca simülasyon saati t olduğunda devam ettirilir. Taşıyıcı algılama
  (CCA) ve kanal rezervi üretecin o anki uyanışında, o anki kanal durumuna göre yapılır; rezerv "şimdi başlayan
  çerçevenin havada kalma süresi"dir, gelecekte başlayacak bir iletim için önceden rezerv yoktur. Geri çekilmesi
  fiziksel olarak daha önce biten düğüm, oluşturulma/çağrı sırasından bağımsız olarak daha önce uyandırılır ve
  kanalı önce alır. AIFS kalibrasyonu (experiments/calibration/mac_aifs.py) aynı mekanizmayla ölçüldüğü için
  kalibre edilmiş `aifs_slots` simülasyona aynen taşınır.
- §5.5 zamanlama kuralı: gönderici sıradaki pakete `RadioModel.next_packet_ready_s(hop_result)` anında başlar
  (son MAC etkinliği + processing_delay_s; başarı/kayıp ayrımı yok). Kuyruk o anda boşsa düğüm boşa düşer; sonraki
  varış, düğümü max(varış, hazır olma) anında başlatır.
- §5.10a kuyruk: sensör başına sınıf kuyruğu (`NodeQueues`), kapasite π9 (sınıfın topolojisi); dolu kuyrukta yeni
  paket düşer (`queue_overflow`). Servis katı öncelikli: π2'si yüksek sınıfın kuyruğu önce (§5.5 shared mod; strict
  modda bir düğümde tek sınıf vardır). Düğümün kendi sınıfının kuyruğu `NodeState.queue`'dur (doluluk = §5.14 x2).
  B1-plain (vsdn_policy=false): tek FIFO, kapasite policy.plain.buffer_packets, π10 = policy.plain.retransmit_limit,
  macMinBE 3 ve AIFS 0 (`BackoffPolicy(vsdn_policy=False)`).
- §5.10a servis: düğüm boşta ve kuyruğu doluysa baş paketin sonraki atlaması O ANKİ plandan alınır
  (`RoutePlan.forward`): aşama 1 root'a, root'ta aşama 2 edge router'a (§5.4 iki aşamalı yol). Root'a yolu kalmayan
  düğümdeki paket root'u atlar (§5.4 bypass; `PacketState.bypass`), edge router'a da yol yoksa `no_route`.
  Yol değişikliği kuyruktaki paketlere böylece hemen uygulanır.
- §5.10a alıcı: başarılı iletimde paket, MAC bitişi + işlem süresinde (`HopResult.t_end`) sonraki düğümün kuyruğuna
  girer (işlem bileşeni atlama başına bir kez). Edge router havuzdur: teslim zamanı t_end'dir.
- §5.10a kayıp nedenleri: queue_overflow; channel_access_failure (radio LOSS_CSMA_FAILURE); link_loss
  (LOSS_RETRY_LIMIT: parça kaybı, denemeler tükendi); no_route; node_dead (ölen düğümün kuyruğundakiler ve iletim
  sırasında gönderici/alıcısı ölen paketler); in_flight_at_end. Ek koruma: atlama sayısı çizgedeki düğüm sayısını
  aşan paket (ardışık plan değişimlerinden doğabilecek döngü; IPv6 hop limit benzeri) no_route ile düşer ve
  `ttl_drops` ile ayrıca sayılır (beklenen 0).
- §5.10a enerji: `radio.send_packet` her parça denemesinde gönderici (E_elec·b + ε_amp·b·d²) ve alıcı (E_elec·b)
  enerjisini düşürür; edge router sınırsız. Enerjisi biten düğüm ölür (`NodeState.consume`): kuyruğu node_dead ile
  boşaltılır, olay kaydedilir ve BÜTÜN topolojilerin yolları hemen yeniden hesaplanır (strict modda diğer topolojilerin
  planı değişmez; shared modda ölü düğüm yabancı aktarıcı olabilir).
- §5.10a yol güncelleme: topoloji başına π11'de bir (`route_update_s`) ve ölümde hemen; her güncellemenin süresi
  kaydedilir (§5.13 karar turu çalışma süresi). Root seçimi `RootPolicy` arayüzünün arkasındadır: M5'te
  `StaticRootPolicy` (§5.12: canlı ve edge router'a yolu olan root korunur; değilse `router.static_root` ile yenisi).
  M9'da controller + engine aynı arayüze bağlanır (dinamik root, histerezis). Root değişim logu (§5.18): reason =
  güncellemeyi başlatan olay (initial / periodic / traffic:<sınıf> / death:<düğüm>), trigger = §5.18 tetikleyici
  türü (periodic / traffic / root_health / node_death), rule = karar kuralı (hysteresis, dead_or_no_edge_path,
  depletion, dead_root, static_reselect, no_candidate), delta_R.
- §5.18 tetikleyici 2 (trafik): dinamik root politikasında (`traffic_trigger`) kaynakta oluşan her paket
  `controller.TrafficMonitor`'a bildirilir; her check_interval_s'de kontrol edilir, tetiklenen sınıfın topolojisi
  hemen yeniden değerlendirilir (reason "traffic:<sınıf>"). Sabit root'lu yöntemlerde izlenmez (§5.10a).
- §5.10a zaman: üretim duration_s'de durur (iz [0, duration) aralığındadır), drain_s boyunca yoldaki paketler
  tamamlanır, kalanlar in_flight_at_end. Isınma ayrımı metrics.py'de (paket oluşturma zamanı >= warm_up_s);
  kanalın sınıf bazlı airtime'ı ısınma sonunda ayrıca kaydedilir ki ısınma sonrası pay hesaplanabilsin.
- §5.6 ortak rastgele sayılar: trafik izi yöntemden bağımsızdır ve aynen verilir. MAC rastgeleliği (geri çekilme,
  PRR çekilişi) düğüm başına ayrı akıştır: `SeedSequence([seed, MAC_STREAM]).spawn(N)`; böylece bir düğümün
  çekilişleri diğer düğümlerin olay sırasına bağlı değildir ve izin SeedSequence(seed) akışıyla çakışmaz.
- §5.12: sabit root'lu bütün yöntemler aynı kuralı (static_root) kullanır; aralarındaki fark yalnızca yol maliyeti
  (router) ve politika bayrağıdır (engine.methods[*].vsdn_policy).

Kullanım: `run_simulation(cfg, method, seed)` ağı, bölümlemeyi ve izi kurup koşturur; `Simulation(cfg, pn, trace,
method)` hazır nesnelerle çalışır (testler). Ağ nesnesi koşu sırasında DEĞİŞTİRİLİR (enerji, canlılık, kuyruk); aynı
seed için farklı yöntemler ayrı ağ nesneleriyle koşturulmalıdır (`build_partitioned_network` belirlenimcidir).
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Protocol

import numpy as np

from nfmtar.config import Config
from nfmtar.network.graph import TOPOLOGY_ORDER, NodeState, PacketQueue, PhysicalNetwork, Topology
from nfmtar.network.partition import Partition, PartitionedNetwork, build_partitioned_network
from nfmtar.network.radio import (
    LOSS_CSMA_FAILURE,
    LOSS_RECEIVER_DEAD,
    LOSS_RETRY_LIMIT,
    LOSS_SENDER_DEAD,
    BackoffPolicy,
    Channel,
    HopResult,
    RadioModel,
    send_packet,
)
from nfmtar.router import EdgeReach, RoutePlan, Router, static_root
from nfmtar.traffic import Packet, TrafficTrace, build_traffic_trace

log = logging.getLogger(__name__)

# --- paket sonuç durumları (§5.10a) ---
IN_FLIGHT = "in_flight"
DELIVERED = "delivered"
DROP_QUEUE_OVERFLOW = "queue_overflow"
DROP_CHANNEL_ACCESS = "channel_access_failure"
DROP_LINK_LOSS = "link_loss"
DROP_NO_ROUTE = "no_route"
DROP_NODE_DEAD = "node_dead"
DROP_IN_FLIGHT_AT_END = "in_flight_at_end"
DROP_REASONS: tuple[str, ...] = (
    DROP_QUEUE_OVERFLOW, DROP_CHANNEL_ACCESS, DROP_LINK_LOSS, DROP_NO_ROUTE, DROP_NODE_DEAD, DROP_IN_FLIGHT_AT_END,
)
_RADIO_LOSS_TO_DROP = {
    LOSS_CSMA_FAILURE: DROP_CHANNEL_ACCESS,
    LOSS_RETRY_LIMIT: DROP_LINK_LOSS,
    LOSS_SENDER_DEAD: DROP_NODE_DEAD,
    LOSS_RECEIVER_DEAD: DROP_NODE_DEAD,
}

# --- olay türü öncelikleri: eş zamanda küçük olan önce (§5.10a deterministik sıra) ---
EV_ARRIVAL, EV_ROUTE_UPDATE, EV_MAC, EV_SERVICE, EV_CONTROL = 0, 1, 2, 3, 4

MAC_STREAM = 0x4D4143   # "MAC": MAC RNG akışlarının entropisi; trafik izi SeedSequence(seed) kullanır (§5.6)

# --- düğüm servis durumu ---
IDLE, SCHEDULED, BUSY, DEAD = "idle", "scheduled", "busy", "dead"


# --------------------------------------------------------------------------- #
# Paket durumu ve kuyruklar
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PacketState:
    """Bir paketin simülasyon boyunca durumu: konum, aşama, sonuç ve biriken gecikme bileşenleri (§5.10a, §5.13).

    status: IN_FLIGHT, DELIVERED veya bir kayıp nedeni. end_s: teslim/kayıp anı; end_node: teslim eden edge router
    veya kaybın yaşandığı düğüm. Bileşenler atlamalar üzerinden toplanır; teslimde Σ bileşen = end_s − created_s.
    """

    packet: Packet
    topology: Topology
    status: str = IN_FLIGHT
    node: int = -1
    to_root: bool = True
    bypass: bool = False
    enqueued_at: float = 0.0
    hops: int = 0
    end_s: float = math.nan
    end_node: int = -1
    queue_s: float = 0.0
    backoff_s: float = 0.0
    aifs_s: float = 0.0
    transmission_s: float = 0.0
    processing_s: float = 0.0
    frame_attempts: int = 0
    csma_busy_events: int = 0

    @property
    def delivered(self) -> bool:
        return self.status == DELIVERED

    @property
    def latency_s(self) -> float:
        """Uçtan uca gecikme (yalnızca teslimde; aksi halde NaN)."""
        return self.end_s - self.packet.created_s if self.delivered else math.nan

    @property
    def deadline_met(self) -> bool:
        """π3 içinde teslim (§5.7, §5.13 süre sınırı oranı)."""
        return self.delivered and self.end_s <= self.packet.deadline_s

    def components_s(self) -> dict[str, float]:
        """Gecikme bileşenleri (§5.13): kuyruk, geri çekilme (AIFS dahil), AIFS payı, iletim, işlem."""
        return {
            "queue": self.queue_s,
            "backoff": self.backoff_s,
            "aifs": self.aifs_s,
            "transmission": self.transmission_s,
            "processing": self.processing_s,
        }


class NodeQueues:
    """Bir sensörün kuyrukları (§5.10a): sınıf -> PacketQueue, π2'ye göre katı öncelikli servis; B1-plain tek FIFO.

    Düğümün kendi sınıfının kuyruğu `NodeState.queue`'dur (kapasite partition.py'de π9 ile atanmış; B1-plain'de
    policy.plain.buffer_packets). Yabancı sınıf kuyrukları (shared mod) ilk pakette, o sınıfın π9'u ile açılır.
    """

    def __init__(self, node: NodeState, single_fifo: bool, capacity_of: Callable[[str], int], order: list[str]) -> None:
        self.node = node
        self.single_fifo = single_fifo
        self.capacity_of = capacity_of
        self.order = order                      # servis sırası: π2 büyükten küçüğe (vsdn) / tek anahtar (plain)
        key = None if single_fifo else node.traffic_class
        self.queues: dict[str | None, PacketQueue] = {key: node.queue}

    def _queue_for(self, traffic_class: str) -> PacketQueue:
        key = None if self.single_fifo else traffic_class
        q = self.queues.get(key)
        if q is None:
            q = self.queues[key] = PacketQueue(self.capacity_of(traffic_class))
        return q

    def push(self, item: PacketState) -> bool:
        """Kuyruğa ekler; dolu ise False (π9 taşması, §5.10a queue_overflow)."""
        return self._queue_for(item.packet.traffic_class).push(item)

    def pop(self) -> PacketState | None:
        """Katı öncelik: sırası önde olan ilk boş olmayan kuyruğun başı; hepsi boşsa None."""
        if self.single_fifo:
            q = self.queues[None]
            return q.pop() if len(q) else None
        for cls in self.order:
            q = self.queues.get(cls)
            if q is not None and len(q):
                return q.pop()
        return None

    def drain(self) -> list[PacketState]:
        """Bütün kuyrukları boşaltır (ölüm, §5.10a node_dead)."""
        items: list[PacketState] = []
        for q in self.queues.values():
            while len(q):
                items.append(q.pop())
        return items

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values())


# --------------------------------------------------------------------------- #
# Root seçimi arayüzü (§5.10a; M9'da controller/engine buraya bağlanır)
# --------------------------------------------------------------------------- #


class RootPolicy(Protocol):
    """Bir yol güncellemesinde topolojinin root'unu seçer; aday yoksa None (§5.10a, §5.12, §5.18)."""

    name: str

    def select(self, topology: Topology, net: PhysicalNetwork, part: Partition, reach: EdgeReach,
               current: int | None, now: float, trigger: str = "periodic") -> int | None: ...


class StaticRootPolicy:
    """§5.12 sabit root: mevcut root canlı ve edge router'a yolu olan bir aday olduğu sürece korunur; değilse
    (ölüm, edge yolunun kopması) aynı kuralla (`router.static_root`: edge router'a Öklid en yakın aday) yenisi seçilir."""

    name = "static"

    def __init__(self) -> None:
        self.last_decision: dict[str, Any] = {}

    def select(self, topology: Topology, net: PhysicalNetwork, part: Partition, reach: EdgeReach,
               current: int | None, now: float, trigger: str = "periodic") -> int | None:
        if not reach.candidates:
            return None
        if current is not None and current in reach.candidates:
            return current
        self.last_decision = {"trigger": "initial" if current is None else "root_health", "rule": "static_reselect",
                              "delta_R": math.nan}
        return static_root(net, reach)


@dataclass
class RouteState:
    """Bir topolojinin güncel yönlendirme durumu."""

    reach: EdgeReach | None = None
    plan: RoutePlan | None = None
    root: int | None = None


# --------------------------------------------------------------------------- #
# Sonuç
# --------------------------------------------------------------------------- #


@dataclass
class SimulationResult:
    """Bir koşunun ham çıktısı; metrikler metrics.py'de hesaplanır (§5.13)."""

    cfg: Config
    method: str
    seed: int
    relay_mode: str
    vsdn_policy: bool
    duration_s: float
    warm_up_s: float
    drain_s: float
    pn: PartitionedNetwork
    trace: TrafficTrace
    packets: list[PacketState]
    class_topology: dict[str, Topology]
    deaths: list[tuple[float, int]]                 # (zaman, düğüm)
    root_changes: list[dict[str, Any]]
    traffic_triggers: list[dict[str, Any]]           # §5.18 tetikleyici 2 olayları (yalnızca dinamik root)
    roots: dict[Topology, int | None]
    route_update_ms: dict[Topology, list[float]]
    airtime_total_s: dict[str, float]              # bütün koşu
    airtime_warm_up_s: dict[str, float]            # ısınma sonunda birikmiş
    energy_warm_up_J: dict[int, float]             # ısınma sonunda sensörlerin kalan enerjisi (§5.13 Φ'deki U_t)
    energy_tx_J: float                             # tamamlanan atlamaların parça denemelerinin gönderici enerjisi
    energy_rx_J: float                             # tamamlanan atlamaların sensör alıcı enerjisi
    energy_rx_edge_J: float                        # edge router alıcıların enerjisi (sensör bütçesinde değil)
    n_hops_completed: int
    hops_active_at_end: int                        # bitişte yarıda kalan atlamalar (enerjileri düğümlerden düşmüş ama
                                                   # energy_tx_J / energy_rx_J sayaçlarına girmemiştir; drain ile beklenen 0)
    ttl_drops: int
    wall_clock_s: float

    def airtime_after_warm_up_s(self) -> dict[str, float]:
        """Sınıf bazında ısınma sonrası airtime (s) (§5.13 airtime payı)."""
        return {c: self.airtime_total_s.get(c, 0.0) - self.airtime_warm_up_s.get(c, 0.0)
                for c in set(self.airtime_total_s) | set(self.airtime_warm_up_s)}

    def energy_consumed_J(self) -> dict[int, float]:
        """Sensör başına tüketilen enerji = başlangıç − kalan (§5.13)."""
        net = self.pn.net
        return {v: net.nodes[v].initial_energy_J - net.nodes[v].energy_J for v in net.sensor_ids}

    def energy_consumed_after_warm_up_J(self) -> dict[int, float]:
        """Sensör başına ısınma sonrası tüketim = ısınma sonundaki kalan − son kalan (§5.13 Φ'deki U_t)."""
        net = self.pn.net
        start = self.energy_warm_up_J or {v: net.nodes[v].initial_energy_J for v in net.sensor_ids}
        return {v: start[v] - net.nodes[v].energy_J for v in net.sensor_ids}

    def packets_after_warm_up(self) -> list[PacketState]:
        """Metriklere giren paketler: oluşturulma zamanı >= warm_up_s (§5.19)."""
        return [ps for ps in self.packets if ps.packet.created_s >= self.warm_up_s]

    def to_record(self) -> dict[str, Any]:
        """Koşu tanımlayıcıları (runs.csv başı; metrikler metrics.py'de eklenir)."""
        return {
            "method": self.method,
            "seed": self.seed,
            "relay_mode": self.relay_mode,
            "vsdn_policy": self.vsdn_policy,
            "load_multiplier": self.trace.load_multiplier,
            "duration_s": self.duration_s,
            "warm_up_s": self.warm_up_s,
            "drain_s": self.drain_s,
            "wall_clock_s": self.wall_clock_s,
            **self.pn.to_record(),
        }


# --------------------------------------------------------------------------- #
# Simülasyon
# --------------------------------------------------------------------------- #


class Simulation:
    """§5.10a çekirdeği: olay kuyruğu, kuyruklar, atlama atlama iletim, yol güncellemeleri, kayıt."""

    def __init__(
        self,
        cfg: Config,
        pn: PartitionedNetwork,
        trace: TrafficTrace,
        method: str,
        relay_mode: str | None = None,
        root_policy: RootPolicy | None = None,
    ) -> None:
        try:
            spec = cfg.engine["methods"][method]
        except KeyError as exc:
            raise KeyError(f"bilinmeyen yöntem {method!r}; engine.methods: {sorted(cfg.engine['methods'])}") from exc
        self.cfg = cfg
        self.pn = pn
        self.net: PhysicalNetwork = pn.net
        self.part: Partition = pn.part
        self.trace = trace
        self.method = method
        self.vsdn_policy = bool(spec["vsdn_policy"])
        self.radio = RadioModel.from_config(cfg)
        self.backoff = BackoffPolicy(cfg.radio.csma, vsdn_policy=self.vsdn_policy)
        self.router = Router.for_method(cfg, self.radio, method, relay_mode)
        self.relay_mode = self.router.relay_mode
        self.channel = Channel.from_network(self.net, cfg.radio.cs_range_m)
        if root_policy is None and spec.get("root") == "dynamic":   # §5.12 B3 / NF-MTAR: controller + engine (M9)
            from nfmtar.controller import DynamicRootPolicy

            root_policy = DynamicRootPolicy.for_method(cfg, method, self.radio, self.net)
        self.root_policy: RootPolicy = root_policy or StaticRootPolicy()
        self.traffic_monitor = None                            # §5.18 tetikleyici 2 (yalnızca dinamik root)
        if getattr(self.root_policy, "traffic_trigger", False):
            from nfmtar.controller import TrafficMonitor

            self.traffic_monitor = TrafficMonitor.from_config(cfg, list(cfg.traffic.classes))
            self.traffic_check_s = float(cfg.controller["triggers"]["traffic"]["check_interval_s"])
        self.kill_roots_at_s: float | None = None   # E5 kök arıza enjeksiyonu (§5.19); run_simulation ayarlar
        self.root_failures: list[tuple[float, str, int]] = []
        self.snapshot_interval_s: float | None = None   # §5.16 eğitim verisi: her K saniyede ağ durumu kaydı
        self.snapshot_hook: Callable[["Simulation"], None] | None = None
        self.seed = trace.seed
        self.duration_s = trace.duration_s
        self.drain_s = cfg.simulation.drain_s
        self.warm_up_s = cfg.run.warm_up_s
        self.end_s = self.duration_s + self.drain_s
        self.hop_limit = len(self.net.nodes)                 # §5.10a döngü koruması

        classes = list(cfg.traffic.classes)
        self.class_topology: dict[str, Topology] = {c: Topology(cfg.traffic.classes[c].topology) for c in classes}
        if self.vsdn_policy:
            policy = {c: cfg.policy.for_topology(self.class_topology[c]) for c in classes}
            self.capacity_of: Callable[[str], int] = lambda c: policy[c].buffer_packets          # π9
            self.retransmit_limit = {c: policy[c].retransmit_limit for c in classes}              # π10
            order = sorted(classes, key=lambda c: (-policy[c].priority, c))                       # π2
        else:                                                                                     # §5.12 B1-plain
            plain = cfg.policy.plain
            self.capacity_of = lambda c: plain.buffer_packets
            self.retransmit_limit = {c: plain.retransmit_limit for c in classes}
            order = classes
            for v in self.net.sensor_ids:
                self.net.nodes[v].queue.capacity = plain.buffer_packets
        self.queues: dict[int, NodeQueues] = {
            v: NodeQueues(self.net.nodes[v], not self.vsdn_policy, self.capacity_of, order) for v in self.net.sensor_ids
        }
        children = np.random.SeedSequence([self.seed, MAC_STREAM]).spawn(len(self.net.sensor_ids))
        self.rng: dict[int, np.random.Generator] = {v: np.random.default_rng(s) for v, s in zip(self.net.sensor_ids, children)}

        self.now = 0.0
        self._heap: list[tuple[float, int, int, str, Any]] = []
        self._seq = 0
        self.node_state: dict[int, str] = {v: IDLE for v in self.net.sensor_ids}
        self.ready_at: dict[int, float] = {v: 0.0 for v in self.net.sensor_ids}
        self.active: dict[int, tuple[Generator[float, float, HopResult], PacketState]] = {}
        self.routes: dict[Topology, RouteState] = {t: RouteState() for t in TOPOLOGY_ORDER}
        self.packets: list[PacketState] = [PacketState(p, self.class_topology[p.traffic_class]) for p in trace.packets]
        self.deaths: list[tuple[float, int]] = []
        self._dead_recorded: set[int] = set()
        self.root_changes: list[dict[str, Any]] = []
        self.route_update_ms: dict[Topology, list[float]] = {t: [] for t in TOPOLOGY_ORDER}
        self.energy_tx_J = 0.0
        self.energy_rx_J = 0.0          # sensör alıcılar
        self.energy_rx_edge_J = 0.0     # edge router alıcılar (sensör bütçesine girmez)
        self.n_hops_completed = 0
        self.ttl_drops = 0
        self.airtime_warm_up_s: dict[str, float] = {}
        self.energy_warm_up_J: dict[int, float] = {}
        self._finished = False

    # ------------------------------------------------------------------ olay kuyruğu

    def _push(self, t: float, priority: int, kind: str, payload: Any = None) -> None:
        heapq.heappush(self._heap, (t, priority, self._seq, kind, payload))
        self._seq += 1

    def run(self) -> SimulationResult:
        """Koşuyu sonuna (duration_s + drain_s) kadar sürer ve ham sonucu döndürür."""
        if self._finished:
            raise RuntimeError("Simulation.run yalnızca bir kez çağrılabilir (ağ nesnesi değişti)")
        t_wall = time.perf_counter()
        for t in TOPOLOGY_ORDER:                                   # başlangıç yolları ve π11 periyotları
            self._update_routes(t, reason="initial")
            self._push(self._route_period_s(t), EV_ROUTE_UPDATE, "route", t)
        if self.packets:
            self._push(self.packets[0].packet.created_s, EV_ARRIVAL, "trace", 0)
        self._push(self.warm_up_s, EV_CONTROL, "warm_up_end")
        if self.kill_roots_at_s is not None:
            self._push(self.kill_roots_at_s, EV_CONTROL, "kill_roots")
        if self.traffic_monitor is not None:
            self._push(self.traffic_check_s, EV_ROUTE_UPDATE, "traffic_check")
        if self.snapshot_interval_s and self.snapshot_hook is not None:
            self._push(self.snapshot_interval_s, EV_CONTROL, "snapshot")
        self._push(self.end_s, EV_CONTROL, "end")

        heap = self._heap
        while heap:
            t, _, _, kind, payload = heapq.heappop(heap)
            if t < self.now:
                raise RuntimeError(f"zaman geri gidemez ({t} < {self.now}; olay {kind})")
            self.now = t
            if kind == "mac":
                self._on_mac(payload)
            elif kind == "service":
                self._on_service(payload)
            elif kind == "trace":
                self._on_trace_arrival(payload)
            elif kind == "relay":
                ps, v = payload
                self._enqueue(ps, v)
            elif kind == "route":
                self._update_routes(payload, reason="periodic")
                self._push(self.now + self._route_period_s(payload), EV_ROUTE_UPDATE, "route", payload)
            elif kind == "traffic_check":
                for c in self.traffic_monitor.check(self.now):
                    self._update_routes(self.class_topology[c], reason=f"traffic:{c}")
                if self.now + self.traffic_check_s < self.duration_s:
                    self._push(self.now + self.traffic_check_s, EV_ROUTE_UPDATE, "traffic_check")
            elif kind == "warm_up_end":
                self.airtime_warm_up_s = dict(self.channel.airtime_s_by_class)
                self.energy_warm_up_J = {v: self.net.nodes[v].energy_J for v in self.net.sensor_ids}
            elif kind == "snapshot":
                self.snapshot_hook(self)
                if self.now + self.snapshot_interval_s < self.duration_s:
                    self._push(self.now + self.snapshot_interval_s, EV_CONTROL, "snapshot")
            elif kind == "kill_roots":
                self._kill_roots()
            elif kind == "end":
                break
            else:  # pragma: no cover
                raise RuntimeError(f"bilinmeyen olay {kind!r}")
        self._finalize()
        self._finished = True
        return SimulationResult(
            cfg=self.cfg, method=self.method, seed=self.seed, relay_mode=self.relay_mode, vsdn_policy=self.vsdn_policy,
            duration_s=self.duration_s, warm_up_s=self.warm_up_s, drain_s=self.drain_s, pn=self.pn, trace=self.trace,
            packets=self.packets, class_topology=self.class_topology, deaths=list(self.deaths),
            root_changes=list(self.root_changes),
            traffic_triggers=list(self.traffic_monitor.events) if self.traffic_monitor is not None else [], roots={t: rs.root for t, rs in self.routes.items()},
            route_update_ms={t: list(v) for t, v in self.route_update_ms.items()},
            airtime_total_s=dict(self.channel.airtime_s_by_class), airtime_warm_up_s=dict(self.airtime_warm_up_s),
            energy_warm_up_J=dict(self.energy_warm_up_J),
            energy_tx_J=self.energy_tx_J, energy_rx_J=self.energy_rx_J, energy_rx_edge_J=self.energy_rx_edge_J,
            n_hops_completed=self.n_hops_completed, hops_active_at_end=len(self.active), ttl_drops=self.ttl_drops, wall_clock_s=time.perf_counter() - t_wall,
        )

    def _route_period_s(self, topology: Topology) -> float:
        """π11 (§5.7, §5.10a): topolojinin periyodik yol güncelleme aralığı."""
        return float(self.cfg.policy.for_topology(topology).route_update_s)

    # ------------------------------------------------------------------ varış ve kuyruk

    def _on_trace_arrival(self, idx: int) -> None:
        """İzdeki idx. paket kaynağında oluşur; sıradaki varış olayı tembelce kuyruğa eklenir (§5.6)."""
        ps = self.packets[idx]
        if self.traffic_monitor is not None:
            self.traffic_monitor.observe(ps.packet.traffic_class, ps.packet.created_s)
        self._enqueue(ps, ps.packet.source)
        if idx + 1 < len(self.packets):
            self._push(self.packets[idx + 1].packet.created_s, EV_ARRIVAL, "trace", idx + 1)

    def _enqueue(self, ps: PacketState, v: int) -> None:
        """Paket v düğümünün kuyruğuna girer (§5.10a); boşta düğüm max(şimdi, hazır olma) anında servise alınır."""
        node = self.net.nodes[v]
        if not node.alive:
            self._drop(ps, DROP_NODE_DEAD, v)
            return
        ps.node = v
        ps.enqueued_at = self.now
        if not self.queues[v].push(ps):
            self._drop(ps, DROP_QUEUE_OVERFLOW, v)
            return
        if self.node_state[v] == IDLE:
            self.node_state[v] = SCHEDULED
            self._push(max(self.now, self.ready_at[v]), EV_SERVICE, "service", v)

    # ------------------------------------------------------------------ servis ve MAC

    def _on_service(self, v: int) -> None:
        """Düğüm boşa çıktı: kuyruk başındaki ilk yönlendirilebilir paketin atlamasını başlat (§5.10a)."""
        if not self.net.nodes[v].alive:
            self.node_state[v] = DEAD
            return
        self.node_state[v] = IDLE
        q = self.queues[v]
        while True:
            ps = q.pop()
            if ps is None:
                return
            nxt = self._next_hop(ps, v)
            if nxt is None:
                self._drop(ps, DROP_NO_ROUTE, v)
                continue
            self._start_hop(v, nxt, ps)
            return

    def _next_hop(self, ps: PacketState, v: int) -> int | None:
        """O anki plandan sonraki atlama (§5.10a): aşama 1 root'a, root'ta aşama 2 edge'e; root'a yol yoksa bypass (§5.4)."""
        plan = self.routes[ps.topology].plan
        if plan is None:
            return None
        if ps.hops >= self.hop_limit:
            self.ttl_drops += 1
            log.warning("paket %d atlama sınırını aştı (%d); no_route (§5.10a döngü koruması)", ps.packet.packet_id, ps.hops)
            return None
        if ps.to_root:
            if v == plan.root:
                ps.to_root = False
            else:
                nxt = plan.forward(v, to_root=True)
                if nxt is not None:
                    return nxt
                ps.to_root = False
                ps.bypass = True
        return plan.forward(v, to_root=False)

    def _start_hop(self, v: int, nxt: int, ps: PacketState) -> None:
        cls = ps.packet.traffic_class
        gen = send_packet(
            self.radio, self.backoff, self.channel, self.rng[v], self.net.nodes[v], self.net.nodes[nxt],
            ps.packet.size_bytes, cls, self.retransmit_limit[cls], now=self.now, enqueued_at=ps.enqueued_at,
        )
        self.node_state[v] = BUSY
        self.active[v] = (gen, ps)
        self._resume(v, gen, ps, first=True)

    def _on_mac(self, v: int) -> None:
        gen, ps = self.active[v]
        self._resume(v, gen, ps)

    def _resume(self, v: int, gen: Generator[float, float, HopResult], ps: PacketState, first: bool = False) -> None:
        """Üreteci şimdiki zamanda devam ettirir (`run_mac_sessions` protokolü); bitince atlama sonucu işlenir."""
        try:
            t_next = next(gen) if first else gen.send(self.now)
        except StopIteration as stop:
            self._complete_hop(v, ps, stop.value)
            return
        if t_next < self.now:
            raise RuntimeError(f"düğüm {v}: MAC zamanı geri gidemez ({t_next} < {self.now})")
        self._push(t_next, EV_MAC, "mac", v)

    def _complete_hop(self, v: int, ps: PacketState, result: HopResult) -> None:
        """Atlama bitti: bileşenler/enerji toplanır, paket teslim edilir / aktarıcıya girer / düşer; gönderici hazırlanır."""
        del self.active[v]
        ps.queue_s += result.queue_s
        ps.backoff_s += result.backoff_s
        ps.aifs_s += result.aifs_s
        ps.transmission_s += result.transmission_s
        ps.processing_s += result.processing_s
        ps.frame_attempts += result.frame_attempts
        ps.csma_busy_events += result.csma_busy_events
        self.energy_tx_J += result.energy_tx_J
        recv = result.receiver
        if self.net.nodes[recv].is_edge_router:       # edge router sınırsız: sensör bütçesinden düşmez (§5.4)
            self.energy_rx_edge_J += result.energy_rx_J
        else:
            self.energy_rx_J += result.energy_rx_J
        self.n_hops_completed += 1
        if result.success:
            ps.hops += 1
            if self.net.nodes[recv].is_edge_router:
                ps.status = DELIVERED
                ps.end_s = result.t_end
                ps.end_node = recv
                ps.node = recv
            else:
                self._push(result.t_end, EV_ARRIVAL, "relay", (ps, recv))   # §5.10a MAC bitişi + işlem süresi
        else:
            self._drop(ps, _RADIO_LOSS_TO_DROP[result.loss_reason], v, result.t_end)
        # §5.5 zamanlama kuralı: sıradaki paket next_packet_ready_s'de
        self.ready_at[v] = self.radio.next_packet_ready_s(result)
        node = self.net.nodes[v]
        if node.alive:
            if len(self.queues[v]):
                self.node_state[v] = SCHEDULED
                self._push(self.ready_at[v], EV_SERVICE, "service", v)
            else:
                self.node_state[v] = IDLE
        for u in (v, recv):                                        # §5.10a enerji ≤ 0 -> ölüm
            n = self.net.nodes[u]
            if not n.alive and not n.is_edge_router and u not in self._dead_recorded:
                self._on_death(u)

    # ------------------------------------------------------------------ kayıp, ölüm, yol güncellemesi

    def _drop(self, ps: PacketState, reason: str, node: int, t: float | None = None) -> None:
        ps.status = reason
        ps.end_s = self.now if t is None else t
        ps.end_node = node

    def _on_death(self, u: int) -> None:
        """Düğüm öldü (§5.10a): kuyruğu node_dead, olay kaydı, bütün topolojilerin yolları hemen yeniden."""
        self._dead_recorded.add(u)
        node = self.net.nodes[u]
        t_death = node.death_time_s if node.death_time_s is not None else self.now
        self.deaths.append((t_death, u))
        self.node_state[u] = DEAD
        for ps in self.queues[u].drain():
            self._drop(ps, DROP_NODE_DEAD, u)
        log.info("t=%.3f: düğüm %d (%s) öldü; %d. ölüm (§5.10a)", self.now, u, node.topology, len(self.deaths))
        for t in TOPOLOGY_ORDER:
            self._update_routes(t, reason=f"death:{u}")

    def _update_routes(self, topology: Topology, reason: str) -> None:
        """Topolojinin reach/plan/root'unu yeniden hesaplar (§5.10a); süresi ve root değişimi kaydedilir (§5.13)."""
        t0 = time.perf_counter()
        rs = self.routes[topology]
        reach = self.router.reach_from_edge(self.net, self.part, topology)
        event = reason.split(":")[0]
        event = {"death": "node_death"}.get(event, event)           # initial | periodic | traffic | node_death
        root = self.root_policy.select(topology, self.net, self.part, reach, rs.root, self.now, trigger=event)
        plan = self.router.plan_routes(self.net, self.part, topology, root, reach) if root is not None else None
        if root != rs.root and reason != "initial":
            decision = getattr(self.root_policy, "last_decision", None) or {}
            if root is None:                                         # aday kalmadı
                decision = {"trigger": "root_health", "rule": "no_candidate", "delta_R": math.nan}
            self.root_changes.append({"time_s": self.now, "topology": topology.value, "old": rs.root, "new": root,
                                      "reason": reason, "trigger": decision.get("trigger", event),
                                      "rule": decision.get("rule", ""), "delta_R": decision.get("delta_R", math.nan)})
            log.info("t=%.3f topoloji %s: root %s -> %s (%s; %s/%s)", self.now, topology.value, rs.root, root, reason,
                     decision.get("trigger", event), decision.get("rule", ""))
        rs.reach, rs.plan, rs.root = reach, plan, root
        self.route_update_ms[topology].append((time.perf_counter() - t0) * 1000.0)

    def _kill_roots(self) -> None:
        """E5 (§5.19): her topolojinin o anki root'u zorla öldürülür (enerji = 0); ölüm işlemi ve yol güncellemesi hemen."""
        for t in TOPOLOGY_ORDER:
            u = self.routes[t].root
            if u is None or not self.net.nodes[u].alive:
                continue
            self.net.nodes[u].kill(self.now)
            self.root_failures.append((self.now, t.value, u))
            if u not in self._dead_recorded:
                self._on_death(u)

    def _finalize(self) -> None:
        """Bitiş: yolda (kuyrukta veya iletimde) kalan paketler in_flight_at_end (§5.10a)."""
        for ps in self.packets:
            if ps.status == IN_FLIGHT:
                self._drop(ps, DROP_IN_FLIGHT_AT_END, ps.node, self.end_s)


# --------------------------------------------------------------------------- #
# Kanonik giriş noktası
# --------------------------------------------------------------------------- #


def run_simulation(
    cfg: Config,
    method: str,
    seed: int,
    relay_mode: str | None = None,
    duration_s: float | None = None,
    root_policy: RootPolicy | None = None,
    kill_roots_at_s: float | None = None,
) -> SimulationResult:
    """Ağ (§5.1), bölümleme (§5.3), iz (§5.6) ve koşu: aynı (cfg, seed) için bütün yöntemler aynı ağı ve izi görür (§5.19).

    relay_mode yalnızca yönlendirme çizgesini seçer; §5.1 yeniden üretim ölçütü cfg.router.relay_mode'a bağlıdır,
    bu yüzden shared koşular için config'te de relay_mode=shared verilmelidir (run.py bunu yapar).
    """
    pn = build_partitioned_network(cfg, seed)
    trace = build_traffic_trace(cfg, pn.net, seed, duration_s)
    sim = Simulation(cfg, pn, trace, method, relay_mode=relay_mode, root_policy=root_policy)
    sim.kill_roots_at_s = kill_roots_at_s
    return sim.run()

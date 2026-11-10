"""Fiziksel ağ: unit disk graph, yerleşim ve düğüm durumu (CLAUDE.md §5.1, §5.2, §5.4).

Uygulanan kararlar:
- §5.1: Alan (varsayılan 300×300 m), N düğüm, menzil R = 50 m. Mesafesi <= R olan her düğüm
  çifti arasında kenar (unit disk graph; COOJA'nın UDGM radyo ortamına benzer). Yeniden üretim
  `default_rng([seed, deneme])` ile yapılır; kabul ölçütü (her topoloji kendi edge router'ına bağlı,
  shared modda ayrıca fiziksel ağ bağlı) bölümleme gerektirdiği için partition.py'deki
  `build_partitioned_network` tarafından `accept` geri çağrısıyla verilir. Deneme sayısı
  `PhysicalNetwork.regen_attempts` alanında tutulur ve runs.csv'ye yazılır.
- §5.2: Kritik düğümler elips bölgede sıkı küme, düzenli düğümler koridor şeridinde. Kritik
  olmayanlar `r_placement: region` (varsayılan) ile Edge-R'ye bitişik, alanı
  A_R = n_R · π · R² / r_target_degree olan dikdörtgende düzgün rastgele; `uniform` (bütün alan)
  yalnızca E0 analizi için. Sınıf oranları config'den.
- §5.4: Her topolojinin alan kenarına yakın SABİT bir edge router'ı vardır. Edge router çizgede
  bir düğümdür ama sensör değildir: trafik üretmez, enerjisi sonsuzdur.
- §5.5 (yalnızca durum alanları): başlangıç enerjisi, kalan enerji, ölüm eşiği; enerji tüketim
  hesabı radio.py'de yapılır. Enerjisi biten düğüm ölür.
- §5.7 π9: düğüm kuyruğu sınırlı FIFO'dur; kapasite partition.py'de topolojiye göre atanır.
  Her düğüm tek trafik sınıfı ürettiği için (§5.3) düğüm içi öncelik gerekmez; sınıflar arası
  öncelik (π2) kanal erişiminde uygulanır (radio.py / controller.py).

Topoloji ataması (V_E, V_L, V_R) bu modülde DEĞİL, partition.py'de yapılır (§5.3): fiziksel
katman yalnızca sensörün ürettiği trafik sınıfını bilir.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import networkx as nx
import numpy as np

from nfmtar.config import Config, EllipticPlacement, LinearPlacement, RandomPlacement

log = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]  # (x0, y0, genişlik, yükseklik)


class Topology(str, Enum):
    """Üç sanal topoloji (§2, §5.2): E eliptik/kritik, L doğrusal/düzenli, R rastgele/kritik olmayan."""

    E = "E"
    L = "L"
    R = "R"


TOPOLOGY_ORDER: tuple[Topology, ...] = (Topology.E, Topology.L, Topology.R)


# --------------------------------------------------------------------------- #
# Düğüm durumu
# --------------------------------------------------------------------------- #


class PacketQueue:
    """Sınırlı FIFO kuyruk (§5.7 π9 tampon; doluluk oranı §5.14'teki x2 kullanım girdisidir).

    capacity None ise sınırsızdır (politika henüz uygulanmamış demektir; partition.py π9 atar).
    Dolu kuyruğa gelen paket düşer ve `dropped` sayacı artar (kayıp nedeni: tampon taşması).
    """

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = capacity
        self._items: deque[Any] = deque()
        self.dropped = 0

    def push(self, item: Any) -> bool:
        """Kuyruğa ekler; tampon doluysa (π9) paketi düşürür ve False döndürür."""
        if self.is_full:
            self.dropped += 1
            return False
        self._items.append(item)
        return True

    def pop(self) -> Any:
        """En eski paketi çıkarır (FIFO)."""
        return self._items.popleft()

    def peek(self) -> Any:
        """En eski paketi çıkarmadan döndürür."""
        return self._items[0]

    def clear(self) -> None:
        """Kuyruğu boşaltır (ör. düğüm öldüğünde)."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return not self._items

    @property
    def is_full(self) -> bool:
        return self.capacity is not None and len(self._items) >= self.capacity

    @property
    def occupancy(self) -> float:
        """Kuyruk doluluk oranı [0,1] (§5.14 x2 kullanım); sınırsız kuyrukta 0."""
        if not self.capacity:
            return 0.0
        return min(1.0, len(self._items) / self.capacity)


@dataclass
class NodeState:
    """Bir düğümün durumu: konum, enerji, kuyruk, canlılık (§5.1, §5.4, §5.5, §5.7).

    - traffic_class: sensörün ürettiği tek trafik sınıfı (§5.3); edge router için None.
    - topology: partition.py'nin atadığı sanal topoloji (§5.3); edge router için sabit (§5.4).
    - energy_J / initial_energy_J: §5.5; edge router'da sonsuz.
    - queue: sınırlı FIFO (§5.7 π9).
    """

    node_id: int
    pos: tuple[float, float]
    traffic_class: str | None
    is_edge_router: bool
    initial_energy_J: float
    energy_J: float
    dead_threshold_J: float
    queue: PacketQueue = field(default_factory=PacketQueue)
    topology: Topology | None = None
    alive: bool = True
    death_time_s: float | None = None

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    @property
    def depletion(self) -> float:
        """Tükenme = 1 − kalan/başlangıç enerji, [0,1] (§5.9 depletion(u), §5.14 x3); edge router'da 0."""
        if math.isinf(self.initial_energy_J) or self.initial_energy_J <= 0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - self.energy_J / self.initial_energy_J))

    def consume(self, joules: float, now_s: float | None = None) -> bool:
        """Enerji düşer; kalan enerji ölüm eşiğine inince düğüm ölür (§5.5). Canlı mı döndürür.

        Edge router'ın enerjisi sonsuzdur (§5.4), tüketim onu etkilemez.
        """
        if joules < 0:
            raise ValueError("tüketim negatif olamaz")
        if self.is_edge_router or not self.alive:
            return self.alive
        self.energy_J = max(0.0, self.energy_J - joules)
        if self.energy_J <= self.dead_threshold_J:
            self._die(now_s)
        return self.alive

    def kill(self, now_s: float | None = None) -> None:
        """Düğümü zorla öldürür: enerji = 0 (E5 root arıza enjeksiyonu, §5.19; canlı demo)."""
        if self.is_edge_router:
            raise ValueError("edge router öldürülemez (§5.4 sabit altyapı)")
        self.energy_J = 0.0
        self._die(now_s)

    def _die(self, now_s: float | None) -> None:
        if self.alive:
            self.alive = False
            self.death_time_s = now_s
            self.queue.clear()


# --------------------------------------------------------------------------- #
# Yerleşim (§5.2)
# --------------------------------------------------------------------------- #


def split_counts(n: int, ratio: Mapping[str, float]) -> dict[str, int]:
    """N düğümü sınıf oranlarına göre tam sayılara böler; en büyük kalan yöntemi, toplam = N (§5.2)."""
    keys = list(ratio)
    raw = [n * ratio[k] for k in keys]
    counts = [int(math.floor(r)) for r in raw]
    remainder = n - sum(counts)
    by_fraction = sorted(range(len(keys)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in by_fraction[:remainder]:
        counts[i] += 1
    return dict(zip(keys, counts))


def _clip_to_area(points: np.ndarray, area_m: tuple[float, float]) -> np.ndarray:
    """Noktaları alan sınırları içine kırpar (yerleşim bölgeleri alanı taşmasın)."""
    return np.column_stack(
        [np.clip(points[:, 0], 0.0, area_m[0]), np.clip(points[:, 1], 0.0, area_m[1])]
    )


def place_elliptic(
    rng: np.random.Generator, n: int, area_m: tuple[float, float], cfg: EllipticPlacement
) -> np.ndarray:
    """Kritik düğümler: elips içinde düzgün dağılım (§5.2 eliptik / yoğun bakım kümesi).

    Elips içinde düzgün örnekleme: r = sqrt(U), θ ~ U(0, 2π); (cx + a·r·cosθ, cy + b·r·sinθ).
    """
    cx, cy = cfg.center_frac[0] * area_m[0], cfg.center_frac[1] * area_m[1]
    a, b = cfg.semi_axes_frac[0] * area_m[0], cfg.semi_axes_frac[1] * area_m[1]
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    pts = np.column_stack([cx + a * r * np.cos(theta), cy + b * r * np.sin(theta)])
    return _clip_to_area(pts, area_m)


def place_linear(
    rng: np.random.Generator, n: int, area_m: tuple[float, float], cfg: LinearPlacement
) -> np.ndarray:
    """Düzenli düğümler: alanı boydan boya geçen dar bir koridor şeridinde düzgün dağılım (§5.2)."""
    along_axis, across_axis = (0, 1) if cfg.axis == "x" else (1, 0)
    center = cfg.center_frac * area_m[across_axis]
    along = rng.uniform(0.0, area_m[along_axis], n)
    across = rng.uniform(center - cfg.width_m / 2.0, center + cfg.width_m / 2.0, n)
    pts = np.empty((n, 2))
    pts[:, along_axis] = along
    pts[:, across_axis] = across
    return _clip_to_area(pts, area_m)


def r_region_rect(
    n_r: int,
    radius_m: float,
    area_m: tuple[float, float],
    edge_pos: tuple[float, float],
    cfg: RandomPlacement,
) -> Rect:
    """R bölgesi dikdörtgeni (§5.2 region): A_R = n_R · π · R² / r_target_degree, Edge-R'ye bitişik.

    Dikdörtgen, Edge-R'den alanın içine doğru uzanır: alan merkezinden Edge-R'ye giden yönün baskın
    ekseninde Edge-R dikdörtgenin dış kenarının orta noktasıdır. Kenarlar region_aspect ile
    (genişlik/yükseklik) belirlenir, alanı aşan kenar kırpılır ve dikdörtgen alan içine kaydırılır.
    Varsayılanla (45 düğüm, R = 50 m, hedef derece 10) A_R ≈ 35.343 m², ~188 × 188 m.
    Yoğunluk sabit ölçeklemede (E2) n_R ile birlikte kendiliğinden ölçeklenir.
    """
    area_w, area_h = area_m
    area_r = n_r * math.pi * radius_m**2 / cfg.r_target_degree
    w = math.sqrt(area_r * cfg.region_aspect)
    h = math.sqrt(area_r / cfg.region_aspect)
    if w > area_w or h > area_h:
        log.warning(
            "R bölgesi (%.0f × %.0f m) alanı aşıyor (%.0f × %.0f m); kırpılıyor (§5.2)", w, h, area_w, area_h
        )
        w, h = min(w, area_w), min(h, area_h)
    ex, ey = edge_pos
    dx, dy = ex - area_w / 2.0, ey - area_h / 2.0
    if abs(dx) >= abs(dy):
        x0 = ex - w if dx >= 0 else ex
        y0 = ey - h / 2.0
    else:
        y0 = ey - h if dy >= 0 else ey
        x0 = ex - w / 2.0
    x0 = min(max(x0, 0.0), area_w - w)
    y0 = min(max(y0, 0.0), area_h - h)
    return (x0, y0, w, h)


def place_random(
    rng: np.random.Generator,
    n: int,
    area_m: tuple[float, float],
    cfg: RandomPlacement,
    region: Rect | None,
) -> np.ndarray:
    """Kritik olmayan düğümler (§5.2): region modunda dikdörtgen içinde, uniform modunda tüm alanda düzgün."""
    if cfg.r_placement == "region":
        if region is None:
            raise ValueError("region modunda dikdörtgen gerekli (r_region_rect)")
        x0, y0, w, h = region
        pts = np.column_stack([rng.uniform(x0, x0 + w, n), rng.uniform(y0, y0 + h, n)])
        return _clip_to_area(pts, area_m)
    return np.column_stack([rng.uniform(0.0, area_m[0], n), rng.uniform(0.0, area_m[1], n)])


def place_sensors(
    rng: np.random.Generator, counts: Mapping[Topology, int], cfg: Config
) -> tuple[dict[Topology, np.ndarray], Rect | None]:
    """Her topoloji bölgesi için sensör konumları (§5.2); sıra E, L, R (tekrarlanabilirlik, §3).

    Döndürür: (topoloji -> (n, 2) konumlar, R bölgesi dikdörtgeni veya uniform modda None).
    """
    area = cfg.network.area_m
    region: Rect | None = None
    if cfg.placement.random.r_placement == "region":
        region = r_region_rect(
            counts[Topology.R],
            cfg.network.radio_range_m,
            area,
            cfg.edge_routers.positions_m[Topology.R.value],
            cfg.placement.random,
        )
    placed = {
        Topology.E: place_elliptic(rng, counts[Topology.E], area, cfg.placement.elliptic),
        Topology.L: place_linear(rng, counts[Topology.L], area, cfg.placement.linear),
        Topology.R: place_random(rng, counts[Topology.R], area, cfg.placement.random, region),
    }
    return placed, region


# --------------------------------------------------------------------------- #
# Fiziksel ağ (§5.1)
# --------------------------------------------------------------------------- #


def unit_disk_edges(positions: np.ndarray, radius_m: float) -> list[tuple[int, int, float]]:
    """Mesafesi <= radius olan tüm düğüm çiftleri (u < v, mesafe) — unit disk graph (§5.1)."""
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    iu, ju = np.triu_indices(len(positions), k=1)
    mask = dist[iu, ju] <= radius_m
    return [(int(i), int(j), float(dist[i, j])) for i, j in zip(iu[mask], ju[mask])]


@dataclass
class PhysicalNetwork:
    """Fiziksel ağ: unit disk çizgesi + düğüm durumları + sabit edge router'lar (§5.1, §5.4).

    Düğüm kimlikleri: sensörler 0..N−1, edge router'lar N, N+1, N+2 (E, L, R sırasıyla).
    regen_attempts: kabul edilen ağdan önce reddedilen üretim sayısı (0 = ilk deneme kabul; §5.1).
    r_region: R bölgesi dikdörtgeni (§5.2 region), uniform modda None.
    """

    graph: nx.Graph
    nodes: dict[int, NodeState]
    sensor_ids: tuple[int, ...]
    edge_router_ids: dict[Topology, int]
    area_m: tuple[float, float]
    radio_range_m: float
    seed: int
    regen_attempts: int
    r_region: Rect | None = None

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_ids)

    def positions(self) -> np.ndarray:
        """(N+3, 2) konum dizisi; satır indeksi = düğüm kimliği."""
        return np.array([self.nodes[i].pos for i in range(len(self.nodes))])

    def distance(self, u: int, v: int) -> float:
        """Öklid mesafesi (m); kenar varsa saklanan değer, yoksa hesaplanır."""
        if self.graph.has_edge(u, v):
            return float(self.graph.edges[u, v]["dist"])
        return math.dist(self.nodes[u].pos, self.nodes[v].pos)

    def neighbors(self, v: int) -> list[int]:
        """Menzil içindeki (fiziksel) komşular (§5.1)."""
        return list(self.graph.neighbors(v))

    def is_edge_router(self, v: int) -> bool:
        return self.nodes[v].is_edge_router

    def topology_of_edge_router(self, v: int) -> Topology:
        """Edge router kimliğinden topolojisi (§5.4)."""
        for t, eid in self.edge_router_ids.items():
            if eid == v:
                return t
        raise KeyError(f"{v} bir edge router değil")

    def sensors_of_class(self, traffic_class: str) -> list[int]:
        """Verilen trafik sınıfını üreten sensörler (§5.3 tek sınıf / düğüm)."""
        return [i for i in self.sensor_ids if self.nodes[i].traffic_class == traffic_class]

    def alive_sensors(self) -> list[int]:
        """Canlı sensörler (§5.5: enerjisi biten düğüm ölür)."""
        return [i for i in self.sensor_ids if self.nodes[i].alive]

    def is_connected(self) -> bool:
        """Fiziksel çizge (edge router'lar dahil) bağlı mı (§5.1: shared modda aranır; E0'da ölçülür)."""
        return nx.is_connected(self.graph)

    def component_count(self) -> int:
        return nx.number_connected_components(self.graph)


def generate_network(cfg: Config, seed: int, attempt: int = 0) -> PhysicalNetwork:
    """Tek bir üretim: `default_rng([seed, attempt])` ile yerleşim (§5.2), edge router'lar (§5.4), kenarlar (§5.1).

    Kabul ölçütü uygulanmaz; yeniden üretim döngüsü için `build_network` kullanılır.
    """
    rng = np.random.default_rng([seed, attempt])
    net_cfg = cfg.network
    counts_by_name = split_counts(net_cfg.n_nodes, cfg.placement.class_ratio)
    counts = {t: counts_by_name[t.value] for t in TOPOLOGY_ORDER}
    placed, region = place_sensors(rng, counts, cfg)

    nodes: dict[int, NodeState] = {}
    node_id = 0
    for t in TOPOLOGY_ORDER:
        traffic_class = cfg.traffic.class_for_topology(t.value)
        for x, y in placed[t]:
            nodes[node_id] = NodeState(
                node_id=node_id,
                pos=(float(x), float(y)),
                traffic_class=traffic_class,
                is_edge_router=False,
                initial_energy_J=cfg.energy.initial_J,
                energy_J=cfg.energy.initial_J,
                dead_threshold_J=cfg.energy.dead_threshold_J,
            )
            node_id += 1
    sensor_ids = tuple(range(node_id))

    edge_router_ids: dict[Topology, int] = {}
    for t in TOPOLOGY_ORDER:
        x, y = cfg.edge_routers.positions_m[t.value]
        nodes[node_id] = NodeState(
            node_id=node_id,
            pos=(float(x), float(y)),
            traffic_class=None,
            is_edge_router=True,
            initial_energy_J=math.inf,
            energy_J=math.inf,
            dead_threshold_J=cfg.energy.dead_threshold_J,
            topology=t,
        )
        edge_router_ids[t] = node_id
        node_id += 1

    graph = nx.Graph()
    for i, st in nodes.items():
        graph.add_node(i, pos=st.pos, is_edge_router=st.is_edge_router, traffic_class=st.traffic_class)
    positions = np.array([nodes[i].pos for i in range(node_id)])
    for u, v, d in unit_disk_edges(positions, net_cfg.radio_range_m):
        graph.add_edge(u, v, dist=d)

    return PhysicalNetwork(
        graph=graph,
        nodes=nodes,
        sensor_ids=sensor_ids,
        edge_router_ids=edge_router_ids,
        area_m=net_cfg.area_m,
        radio_range_m=net_cfg.radio_range_m,
        seed=seed,
        regen_attempts=attempt,
        r_region=region,
    )


def build_network(
    cfg: Config,
    seed: int,
    accept: Callable[[PhysicalNetwork], bool] | None = None,
) -> PhysicalNetwork:
    """Seed'den tekrarlanabilir fiziksel ağ; `accept` reddettikçe yeni denemeyle yeniden üretir (§5.1, §3).

    Deneme k için üreteç `default_rng([seed, k])`: aynı seed aynı ağı verir, her yeniden üretim farklı
    ama belirlenimci bir akış kullanır. `accept` None ise ilk üretim kabul edilir (E0: yalnızca ölç).
    §5.1'deki kabul ölçütü (topoloji bağlantısallığı) bölümleme gerektirdiğinden kanonik giriş noktası
    `nfmtar.network.partition.build_partitioned_network`'tür. Yeniden üretimler log'a yazılır;
    `max_regen_attempts` aşılırsa RuntimeError.
    """
    if accept is None:
        return generate_network(cfg, seed, 0)
    attempts = max(1, cfg.network.max_regen_attempts)
    for attempt in range(attempts):
        net = generate_network(cfg, seed, attempt)
        if accept(net):
            if attempt > 0:
                log.warning("seed %d: ağ %d yeniden üretimden sonra kabul edildi (§5.1)", seed, attempt)
            return net
        log.debug("seed %d deneme %d: kabul ölçütü sağlanmadı; yeniden üretiliyor (§5.1)", seed, attempt)
    raise RuntimeError(
        f"seed {seed}: {attempts} denemede kabul ölçütünü sağlayan ağ üretilemedi "
        f"(N={cfg.network.n_nodes}, alan={cfg.network.area_m}, R={cfg.network.radio_range_m}, "
        f"r_placement={cfg.placement.random.r_placement}, relay_mode={cfg.router.relay_mode})"
    )

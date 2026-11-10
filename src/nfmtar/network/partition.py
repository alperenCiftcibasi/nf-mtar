"""V-SDN bölümleme: V_E, V_L, V_R, edge router'a bağlantısallık ve §5.1 yeniden üretim ölçütü (CLAUDE.md §5.1, §5.3, §5.4).

Uygulanan kararlar:
- §5.3: Atama DETERMİNİSTİK — her sensör ürettiği trafik sınıfına göre tek bir topolojiye girer
  (makaledeki Denk. 2'nin olasılıkları tanımsız; one-hot kabul). Her düğüm tek trafik sınıfı
  üretir; bölümler ayrıktır. Veri yapısı örtüşen bölümlemeye genişletilebilir:
  node -> set(topology). Her topolojinin indüklenmiş alt çizgesinin kendi edge router'ına bağlı
  olduğu kontrol edilir; kopuk düğümler uyarı + log, sayısı metrik olarak kaydedilir (§5.13).
- §5.1: Yeniden üretim ölçütü (enforce_partition_connectivity = true): strict modda her topoloji
  kendi edge router'ına bağlı; shared modda (§5.9 relay_mode) ayrıca fiziksel ağ bağlı. Ölçüt
  sağlanana kadar `default_rng([seed, deneme])` ile yeniden üretilir; deneme sayısı loglanır ve
  runs.csv'ye yazılır (`PartitionedNetwork.to_record`). false ise (E0) yeniden üretim yapılmaz.
- §5.4: Her topolojinin alt çizgesi, o topolojinin sabit edge router'ını içerir.
- §5.7 π9: Topolojiye özgü tampon kapasitesi, bölümleme sırasında düğüm kuyruklarına uygulanır
  (B1-plain için controller bu politikayı geçersiz kılabilir, §5.12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from nfmtar.config import Config
from nfmtar.network.graph import TOPOLOGY_ORDER, PhysicalNetwork, Topology, build_network

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Partition:
    """Bölümleme sonucu: üyelikler, düğüm -> topoloji kümesi, kopuk düğümler (§5.3)."""

    members: dict[Topology, frozenset[int]]
    node_topologies: dict[int, frozenset[Topology]]
    disconnected: dict[Topology, frozenset[int]]
    edge_router_ids: dict[Topology, int]

    def topology_of(self, node_id: int) -> Topology:
        """Düğümün tek topolojisi (§5.3 ayrık bölümleme); birden çoksa ValueError."""
        tops = self.node_topologies[node_id]
        if len(tops) != 1:
            raise ValueError(f"düğüm {node_id} {len(tops)} topolojide; tekil atama bekleniyordu (§5.3)")
        return next(iter(tops))

    def connected_members(self, topology: Topology) -> frozenset[int]:
        """Edge router'a yolu olan sensörler (§5.3 bağlantısallık; §5.17 aday kümesi için temel)."""
        return self.members[topology] - self.disconnected[topology]

    @property
    def disconnected_count(self) -> int:
        """Toplam kopuk düğüm sayısı — §5.13 metriği."""
        return sum(len(s) for s in self.disconnected.values())

    def subgraph(self, net: PhysicalNetwork, topology: Topology, alive_only: bool = False) -> nx.Graph:
        """Topolojinin indüklenmiş alt çizgesi: üyeler + kendi edge router'ı (§5.3, §5.4).

        alive_only=True ile ölü düğümler dışlanır (§5.5; yol/rank hesapları canlı düğümler üzerinde).
        """
        return induced_subgraph(net, self.members[topology], self.edge_router_ids[topology], alive_only)

    def subgraph_mean_degree(self, net: PhysicalNetwork, topology: Topology) -> float:
        """Üyelerin alt çizgedeki ortalama derecesi (E0 analizi, §5.19)."""
        members = self.members[topology]
        if not members:
            return 0.0
        sub = self.subgraph(net, topology)
        return sum(sub.degree(v) for v in members) / len(members)


def class_to_topology_map(cfg: Config) -> dict[str, Topology]:
    """Trafik sınıfı -> topoloji eşlemesi; config'de bire bir tanımlıdır (§5.3 one-hot, §5.6)."""
    return {name: Topology(c.topology) for name, c in cfg.traffic.classes.items()}


def assign_topologies(net: PhysicalNetwork, cfg: Config) -> dict[int, frozenset[Topology]]:
    """Deterministik atama: sensör -> {ürettiği sınıfın topolojisi}; edge router -> {kendi topolojisi} (§5.3, §5.4).

    Dönüş yapısı küme değerlidir ki örtüşen bölümleme ileride aynı arayüzle desteklenebilsin (§5.3).
    """
    mapping = class_to_topology_map(cfg)
    result: dict[int, frozenset[Topology]] = {}
    for node_id, st in net.nodes.items():
        if st.is_edge_router:
            result[node_id] = frozenset({net.topology_of_edge_router(node_id)})
        else:
            if st.traffic_class is None:
                raise ValueError(f"sensör {node_id} trafik sınıfı taşımıyor (§5.3)")
            result[node_id] = frozenset({mapping[st.traffic_class]})
    return result


def induced_subgraph(
    net: PhysicalNetwork, members: frozenset[int], edge_router_id: int, alive_only: bool = False
) -> nx.Graph:
    """Üyeler ∪ {edge router} üzerindeki indüklenmiş alt çizge görünümü (§5.3)."""
    keep = set(members) | {edge_router_id}
    if alive_only:
        keep = {v for v in keep if net.nodes[v].alive}
    return net.graph.subgraph(keep)


def check_edge_connectivity(
    net: PhysicalNetwork, members: frozenset[int], edge_router_id: int, alive_only: bool = False
) -> frozenset[int]:
    """Alt çizge içinde edge router'a yolu OLMAYAN üyeleri döndürür (§5.3 bağlantısallık kontrolü).

    Yol yalnızca aynı topolojinin düğümlerinden geçebilir (bölümler ayrık, §5.3; strict relay_mode'da
    yol seçimi topoloji alt çizgesinde, §5.9).
    """
    sub = induced_subgraph(net, members, edge_router_id, alive_only)
    reachable = nx.node_connected_component(sub, edge_router_id)
    considered = {v for v in members if not alive_only or net.nodes[v].alive}
    return frozenset(considered - reachable)


def partition_network(net: PhysicalNetwork, cfg: Config, warn: bool = True) -> Partition:
    """Fiziksel ağı V_E, V_L, V_R'ye böler, düğüm durumlarına yazar, bağlantısallığı kontrol eder (§5.3).

    Adımlar:
    1. Deterministik atama (sınıf -> topoloji, one-hot) ve NodeState.topology güncellemesi.
    2. Ayrıklık doğrulaması (cfg.partition.disjoint).
    3. π9 tampon kapasitesinin kuyruklara uygulanması (§5.7).
    4. Her topoloji için edge router'a bağlantısallık kontrolü; kopuk düğümler uyarı + log (§5.3, §5.13).
       warn=False ile uyarı bastırılır (yeniden üretim döngüsündeki ara denemeler için).
    """
    node_topologies = assign_topologies(net, cfg)
    members: dict[Topology, set[int]] = {t: set() for t in TOPOLOGY_ORDER}
    for node_id in net.sensor_ids:
        tops = node_topologies[node_id]
        if cfg.partition.disjoint and len(tops) != 1:
            raise ValueError(f"sensör {node_id} {len(tops)} topolojiye atanmış; ayrık bölümleme bekleniyor (§5.3)")
        st = net.nodes[node_id]
        if len(tops) == 1:
            st.topology = next(iter(tops))
            st.queue.capacity = cfg.policy.for_topology(st.topology).buffer_packets  # §5.7 π9
        for t in tops:
            members[t].add(node_id)

    disconnected: dict[Topology, frozenset[int]] = {}
    for t in TOPOLOGY_ORDER:
        edge_id = net.edge_router_ids[t]
        if cfg.partition.check_edge_connectivity:
            disconnected[t] = check_edge_connectivity(net, frozenset(members[t]), edge_id)
        else:
            disconnected[t] = frozenset()
        if disconnected[t] and warn and cfg.partition.log_disconnected_count:
            log.warning(
                "seed %d topoloji %s: %d/%d sensörün edge router'a yolu yok (§5.3): %s",
                net.seed,
                t.value,
                len(disconnected[t]),
                len(members[t]),
                sorted(disconnected[t]),
            )

    return Partition(
        members={t: frozenset(members[t]) for t in TOPOLOGY_ORDER},
        node_topologies=node_topologies,
        disconnected=disconnected,
        edge_router_ids=dict(net.edge_router_ids),
    )


# --------------------------------------------------------------------------- #
# §5.1 yeniden üretim ölçütü ve kanonik giriş noktası
# --------------------------------------------------------------------------- #


def connectivity_criterion(net: PhysicalNetwork, part: Partition, cfg: Config) -> bool:
    """§5.1 kabul ölçütü: her topoloji kendi edge router'ına bağlı; shared modda (§5.9) ayrıca fiziksel ağ bağlı."""
    if part.disconnected_count > 0:
        return False
    if cfg.router.relay_mode == "shared" and not net.is_connected():
        return False
    return True


@dataclass(frozen=True)
class PartitionedNetwork:
    """Kabul edilmiş fiziksel ağ + bölümlemesi; deneylerin paylaştığı ağ nesnesi (§5.1, §5.19)."""

    net: PhysicalNetwork
    part: Partition
    r_placement: str
    relay_mode: str
    enforced: bool

    def to_record(self) -> dict[str, Any]:
        """runs.csv'ye yazılacak ağ üretim alanları (§5.1 deneme sayısı, §5.13 kopuk düğüm sayısı)."""
        rec: dict[str, Any] = {
            "seed": self.net.seed,
            "n_nodes": self.net.n_sensors,
            "r_placement": self.r_placement,
            "relay_mode": self.relay_mode,
            "partition_connectivity_enforced": self.enforced,
            "regen_attempts": self.net.regen_attempts,
            "physical_connected": self.net.is_connected(),
            "disconnected_total": self.part.disconnected_count,
        }
        for t in TOPOLOGY_ORDER:
            rec[f"disconnected_{t.value}"] = len(self.part.disconnected[t])
        return rec


def build_partitioned_network(cfg: Config, seed: int) -> PartitionedNetwork:
    """Kanonik giriş noktası: §5.1 ölçütünü sağlayana kadar üret, bölümle, π9 uygula, kopukları logla.

    enforce_partition_connectivity = false ise (E0) ilk üretim alınır ve yalnızca ölçülür.
    """
    accept = None
    if cfg.network.enforce_partition_connectivity:

        def accept(candidate: PhysicalNetwork) -> bool:
            return connectivity_criterion(candidate, partition_network(candidate, cfg, warn=False), cfg)

    net = build_network(cfg, seed, accept=accept)
    part = partition_network(net, cfg, warn=True)
    return PartitionedNetwork(
        net=net,
        part=part,
        r_placement=cfg.placement.random.r_placement,
        relay_mode=cfg.router.relay_mode,
        enforced=cfg.network.enforce_partition_connectivity,
    )


def build_partitioned_networks(cfg: Config, seeds: Iterable[int]) -> dict[int, PartitionedNetwork]:
    """Birden çok seed için ağ üretir (deneylerde tüm yöntemler aynı ağları paylaşır, §5.19)."""
    return {s: build_partitioned_network(cfg, s) for s in seeds}

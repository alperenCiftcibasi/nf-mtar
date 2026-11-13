"""Yol seçimi: kenar maliyeti, ters yönlü tek kaynaklı Dijkstra, iki aşamalı yol ve relay_mode (CLAUDE.md §5.9, §5.4, §5.12, §5.17).

Uygulanan kararlar:
- §5.9 kenar maliyeti (u -> v), bütün terimler [0, 1]:
    cost(u, v) = λ1 · E_hop(d)/E_hop(R)  +  λ2 · (−log PRR(u,v)) / L_max  +  λ3 · depletion(u)
  E_hop(d) = 2·E_elec + ε_amp·d²: §5.5 birinci dereceden radyo modelinde bir atlamanın bit başına enerjisi
  (gönderim + alım); `RadioModel.hop_energy_J_per_bit` ile config'in energy: bölümünden okunur, burada sabit
  kopyası yoktur (tek kaynak). Sabit 2·E_elec payı her atlamaya taban bedel verir: tek 40 m atlama (≈260 nJ/bit)
  iki 20 m atlamadan (≈280 nJ/bit) ucuzdur; E_hop(d)/d en küçük değerini d* = sqrt(2·E_elec/ε_amp) ≈ 31,6 m'de
  alır, bu yüzden enerji terimi ~30 m'lik atlamaları tercih eder (ilk sürümdeki d²/R² vekili bu tabanı içermediği
  için çok sayıda kısa atlamayı ödüllendiriyordu).
  −log PRR: yol boyunca toplamı −log(Π PRR) olduğundan en düşük maliyet = en yüksek uçtan uca teslim olasılığı;
  L_max = −log(PRR_min). PRR < PRR_min olan bağlantılar DIŞLANMAZ, maliyette PRR_min'e kırpılır (bağlantısallık
  korunur, terim 1'i aşmaz). depletion(u) = 1 − kalan/başlangıç enerji (gönderici; edge router 0).
- §5.9 λ'lar: λ3 = π6 (E 0,3 / L 0,5 / R 0,8; `lambda3_from_policy`), λ1 ve λ2 config'deki oranla (1 − λ3)'ü
  paylaşır (`normalize_lambdas`; varsayılan eşit -> λ1 = λ2 = (1 − λ3)/2, toplam 1). lambda3_from_policy=false ->
  λ3 = 0 (enerji terimsiz ablasyon); normalize_lambdas=false -> ham değerler.
- §5.9 / §5.12 yöntem varyantları (engine.methods[*].path_cost):
    weighted      : yukarıdaki formül (B3, NF-MTAR)
    etx_depletion : B2 — cost(u,v) = ETX(u,v) + π6·depletion(u), ETX = 1/max(PRR, PRR_min); ETX NORMALİZE EDİLMEZ
                    (MRHOF / RFC 6719: yol maliyeti ETX toplamı, her atlama en az 1); enerji terimi yok
    hop           : B1 (ve B1-plain) — her kenar 1 (atlama sayısı)
- §5.9 ters yönlü tek kaynaklı Dijkstra: maliyet göndericiye (depletion(u)) bağlı olduğundan çizge YÖNLÜdür;
  hedeften (edge router veya root) ters çizgede tek kaynaklı Dijkstra bütün düğümlerin hedefe en ucuz yolunu tek
  seferde verir. Edge router'dan: adayların H ve C değerleri; root'tan: bütün sensörlerin root'a yolları.
- §5.17 H: edge router'a EN AZ atlama sayısı (yönlendirme çizgesinde BFS), en ucuz yolun atlama sayısı DEĞİL.
  Gerekçe: H topolojik yakınlığı, C maliyeti taşır; ikisi aynı yoldan türetilirse rank'taki w2 ve w3 terimleri aynı
  bilgiyi iki kez sayar. En ucuz yolun atlaması `EdgeReach.path_hops` ile ayrıca verilir (günlük/analiz).
- §5.4 iki aşamalı yol: sensör -> root -> edge router. **Bypass:** root'a (güncel relay_mode çizgesinde) ulaşamayan
  ama edge router'a yolu olan sensör root'u atlar, doğrudan edge router yolunu kullanır (`RoutePlan.bypass`); bütün
  yöntemlerde aynı kural, oranı kayda yazılır. Ne root'a ne edge router'a yolu olan canlı sensör `RoutePlan.no_route`
  kümesindedir ve yolu yoktur (M5'te paket kaybı sebebi `NO_ROUTE`).
- §5.9 relay_mode: strict (varsayılan, makaleye sadık) — çizge = topolojinin CANLI sensörleri + kendi edge router'ı.
  shared — çizge = fiziksel ağın bütün canlı sensörleri (diğer topolojilerin düğümleri aktarıcı olabilir) + kendi
  edge router'ı. Edge router'lar hiçbir modda aktarıcı değildir: başka topolojilerin edge router'ları çizgeye
  alınmaz, kendi edge router'ının çıkan kenarı yoktur (yalnızca hedef). Root seçimi ve politika her zaman kendi
  topolojisinindir; shared modda aday kümesi yine topolojinin üyeleriyle sınırlıdır.
- Ölü düğümler (§5.5) çizgeye alınmaz. Maliyetler dinamiktir (depletion değişir): çizge her çağrıda güncel düğüm
  durumundan yeniden kurulur; her karar turunda (§5.18) yeniden hesaplanır.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from nfmtar.config import Config
from nfmtar.network.graph import NodeState, PhysicalNetwork, Topology
from nfmtar.network.partition import Partition
from nfmtar.network.radio import RadioModel

log = logging.getLogger(__name__)

PATH_COSTS: tuple[str, ...] = ("weighted", "etx_depletion", "hop")   # §5.12 engine.methods[*].path_cost
RELAY_MODES: tuple[str, ...] = ("strict", "shared")                   # §5.9

# RoutePlan.route_status değerleri; NO_ROUTE M5'te paket kaybı sebebi olarak loglanır (§5.4)
VIA_ROOT = "via_root"
BYPASS = "bypass"
NO_ROUTE = "no_route"


@dataclass(frozen=True)
class Lambdas:
    """Maliyet ağırlıkları (§5.9): l1 enerji terimi, l2 bağlantı terimi, l3 = π6 depletion.

    weighted: normalize (toplam 1). etx_depletion (B2): l1 = 0, l2 = 1 (ETX normalize edilmez), l3 = π6. hop: hepsi 0.
    """

    l1: float
    l2: float
    l3: float

    @property
    def total(self) -> float:
        return self.l1 + self.l2 + self.l3


@dataclass(frozen=True)
class EdgeReach:
    """Edge router'dan ters Dijkstra + BFS sonucu (§5.9, §5.17): her ulaşabilen düğüm için H, C ve en ucuz yol.

    hops (H): edge router'a EN AZ atlama (BFS; maliyetten bağımsız, §5.17). cost (C): en ucuz yolun maliyeti.
    paths: v -> ... -> edge router. candidates: topolojinin canlı ve edge router'a yolu olan sensörleri
    (shared modda çizgedeki diğer topolojilerin sensörleri aday DEĞİLDİR).
    """

    topology: Topology
    edge_router: int
    relay_mode: str
    hops: dict[int, int]
    cost: dict[int, float]
    paths: dict[int, tuple[int, ...]]
    candidates: frozenset[int]

    def reachable(self, v: int) -> bool:
        return v in self.paths

    def path_hops(self, v: int) -> int:
        """En ucuz yolun atlama sayısı (H'den farklı olabilir; §5.17)."""
        return len(self.paths[v]) - 1


@dataclass(frozen=True)
class RoutePlan:
    """Bir topolojinin iki aşamalı yol planı (§5.4, §5.9): sensör -> root (aşama 1), root -> edge router (aşama 2).

    to_root: üye sensör -> root'a en ucuz yol (root için (root,)). bypass: root'a ulaşamayan ama edge router'a
    ulaşan üyeler -> doğrudan edge router yolu (§5.4). no_route: ikisine de ulaşamayan canlı üyeler (yol yok).
    """

    topology: Topology
    root: int
    edge_router: int
    relay_mode: str
    root_to_edge: tuple[int, ...]
    root_to_edge_cost: float
    to_root: dict[int, tuple[int, ...]]
    to_root_cost: dict[int, float]
    bypass: dict[int, tuple[int, ...]]
    no_route: frozenset[int]

    def route_status(self, v: int) -> str:
        """VIA_ROOT, BYPASS veya NO_ROUTE (§5.4; M5 paket logu)."""
        if v in self.to_root:
            return VIA_ROOT
        if v in self.bypass:
            return BYPASS
        return NO_ROUTE

    def full_path(self, v: int) -> tuple[int, ...] | None:
        """Sensörün uçtan uca yolu: v -> ... -> root -> ... -> edge router; bypass'ta v -> ... -> edge router; no_route'ta None."""
        if v in self.to_root:
            return self.to_root[v] + self.root_to_edge[1:]
        return self.bypass.get(v)

    def next_hop(self, v: int) -> int | None:
        """Sensörün ilk aktarıcısı (root veya edge router olabilir); yol yoksa None."""
        path = self.full_path(v)
        return path[1] if path and len(path) > 1 else None

    def hops(self, v: int) -> int | None:
        path = self.full_path(v)
        return None if path is None else len(path) - 1

    def via_root(self, v: int) -> bool:
        return v in self.to_root

    def to_record(self) -> dict[str, Any]:
        """Karar turu kaydı (§5.13/§5.19): root, root'un edge yolu, yol uzunlukları, bypass oranı, no_route sayısı."""
        hops = [self.hops(v) for v in self.to_root]
        n_members = len(self.to_root) + len(self.bypass) + len(self.no_route)
        return {
            "topology": self.topology.value,
            "root": self.root,
            "root_hops_to_edge": len(self.root_to_edge) - 1,
            "root_cost_to_edge": self.root_to_edge_cost,
            "n_via_root": len(self.to_root),
            "n_bypass": len(self.bypass),
            "bypass_frac": len(self.bypass) / n_members if n_members else 0.0,
            "n_no_route": len(self.no_route),
            "mean_hops_via_root": sum(hops) / len(hops) if hops else float("nan"),
        }


def _topology(value: Any) -> Topology:
    return Topology(getattr(value, "value", value))


class Router:
    """§5.9 yol seçicisi: kenar maliyeti + ters yönlü Dijkstra + iki aşamalı yol; path_cost ile yöntem varyantı (§5.12).

    Durumsuzdur: her çağrı ağın GÜNCEL durumundan (canlılık, enerji) çizgeyi yeniden kurar (dinamik maliyet).
    """

    def __init__(self, cfg: Config, radio: RadioModel, path_cost: str = "weighted", relay_mode: str | None = None) -> None:
        if path_cost not in PATH_COSTS:
            raise ValueError(f"path_cost {path_cost!r}; {PATH_COSTS} bekleniyor (§5.12)")
        self.relay_mode = cfg.router.relay_mode if relay_mode is None else relay_mode
        if self.relay_mode not in RELAY_MODES:
            raise ValueError(f"relay_mode {self.relay_mode!r}; {RELAY_MODES} bekleniyor (§5.9)")
        self.cfg = cfg
        self.radio = radio
        self.path_cost = path_cost
        self.range_m = cfg.network.radio_range_m
        self.prr_min = cfg.router.prr_min
        self.l_max = -math.log(self.prr_min)                        # §5.9 L_max = −log(PRR_min)
        self.e_hop_max = radio.hop_energy_J_per_bit(self.range_m)   # §5.9 E_hop(R): enerji terimi normalizasyonu

    @classmethod
    def for_method(cls, cfg: Config, radio: RadioModel, method: str, relay_mode: str | None = None) -> "Router":
        """engine.methods[method].path_cost'a göre yönlendirici (§5.12: B1/B1-plain hop, B2 etx_depletion, B3/NF-MTAR weighted)."""
        try:
            spec = cfg.engine["methods"][method]
        except KeyError as exc:
            raise KeyError(f"bilinmeyen yöntem {method!r}; engine.methods: {sorted(cfg.engine['methods'])}") from exc
        return cls(cfg, radio, spec["path_cost"], relay_mode)

    # ------------------------------------------------------------------ kenar maliyeti (§5.9)

    def lambdas(self, topology: Any) -> Lambdas:
        """weighted: λ3 = π6 (veya 0), λ1/λ2 config oranıyla (1 − λ3)'ü paylaşır; B2: (0, 1, π6); B1: (0, 0, 0) (§5.9)."""
        rc = self.cfg.router
        if self.path_cost == "hop":
            return Lambdas(0.0, 0.0, 0.0)
        l3 = self.cfg.policy.for_topology(topology).energy_weight if rc.lambda3_from_policy else 0.0
        if self.path_cost == "etx_depletion":
            return Lambdas(0.0, 1.0, l3)
        l1, l2 = rc.lambda1, rc.lambda2
        if rc.normalize_lambdas:
            s = l1 + l2
            l1, l2 = (1.0 - l3) * l1 / s, (1.0 - l3) * l2 / s
        return Lambdas(l1, l2, l3)

    def link_prr(self, dist_m: float) -> float:
        """Maliyette kullanılan PRR: radyo modelinin PRR'si PRR_min'e kırpılır (§5.9; bağlantı dışlanmaz)."""
        return max(self.radio.prr(dist_m), self.prr_min)

    def energy_term(self, dist_m: float) -> float:
        """E_hop(d)/E_hop(R) ∈ [E_hop(0)/E_hop(R), 1] (§5.9); E_hop RadioModel'den (tek kaynak)."""
        return min(1.0, self.radio.hop_energy_J_per_bit(dist_m) / self.e_hop_max)

    def link_term(self, dist_m: float) -> float:
        """(−log PRR)/L_max ∈ [0, 1] (§5.9 weighted)."""
        return -math.log(self.link_prr(dist_m)) / self.l_max

    def etx(self, dist_m: float) -> float:
        """ETX = 1/max(PRR, PRR_min) ∈ [1/prr_max, 1/PRR_min]; normalize edilmez (§5.9 B2)."""
        return 1.0 / self.link_prr(dist_m)

    def edge_cost(self, sender: NodeState, dist_m: float, lam: Lambdas) -> float:
        """weighted: l1·E_hop(d)/E_hop(R) + l2·(−log PRR)/L_max + l3·depletion(u); B2: ETX + π6·depletion(u); B1: 1 (§5.9)."""
        if self.path_cost == "hop":
            return 1.0
        if self.path_cost == "etx_depletion":
            return self.etx(dist_m) + lam.l3 * sender.depletion
        return lam.l1 * self.energy_term(dist_m) + lam.l2 * self.link_term(dist_m) + lam.l3 * sender.depletion

    def cost(self, net: PhysicalNetwork, u: int, v: int, topology: Any) -> float:
        """Fiziksel kenarın güncel maliyeti (u gönderici); kenar yoksa ValueError."""
        if not net.graph.has_edge(u, v):
            raise ValueError(f"{u}-{v} fiziksel kenar değil")
        return self.edge_cost(net.nodes[u], net.distance(u, v), self.lambdas(topology))

    def path_cost_of(self, net: PhysicalNetwork, path: Sequence[int], topology: Any) -> float:
        """Verilen yolun maliyeti (günlük/kâhin için; §5.9)."""
        lam = self.lambdas(topology)
        return sum(self.edge_cost(net.nodes[u], net.distance(u, v), lam) for u, v in zip(path, path[1:]))

    def path_prr(self, net: PhysicalNetwork, path: Sequence[int]) -> float:
        """Yolun uçtan uca teslim olasılığı Π PRR (kırpılmış PRR ile; §5.9, §5.16 kâhin)."""
        return math.prod(self.link_prr(net.distance(u, v)) for u, v in zip(path, path[1:]))

    # ------------------------------------------------------------------ çizge ve Dijkstra (§5.9)

    def routing_graph(self, net: PhysicalNetwork, part: Partition, topology: Any) -> nx.DiGraph:
        """Güncel yönlü yönlendirme çizgesi (§5.9 relay_mode): kenar (u -> v) ağırlığı cost(u, v).

        strict: topolojinin canlı üyeleri + kendi edge router'ı; shared: bütün canlı sensörler + kendi edge router'ı.
        Edge router hedef düğümdür, çıkan kenarı yoktur (aktarıcı değil); diğer edge router'lar çizgede yoktur.
        """
        t = _topology(topology)
        edge_id = net.edge_router_ids[t]
        pool = part.members[t] if self.relay_mode == "strict" else net.sensor_ids
        senders = {v for v in pool if net.nodes[v].alive}
        allowed = senders | {edge_id}
        lam = self.lambdas(t)
        g = nx.DiGraph(topology=t, edge_router=edge_id, relay_mode=self.relay_mode, path_cost=self.path_cost)
        g.add_nodes_from(allowed)
        for u in senders:
            su = net.nodes[u]
            for v in net.graph.neighbors(u):
                if v in allowed:
                    d = net.distance(u, v)
                    g.add_edge(u, v, cost=self.edge_cost(su, d, lam), dist=d, prr=self.link_prr(d))
        return g

    @staticmethod
    def _paths_to(g: nx.DiGraph, target: int) -> tuple[dict[int, float], dict[int, tuple[int, ...]]]:
        """Ters çizgede tek kaynaklı Dijkstra: her v için v -> ... -> target en ucuz yolu ve maliyeti (§5.9)."""
        rev = g.reverse(copy=False)
        dist, rev_paths = nx.single_source_dijkstra(rev, target, weight="cost")
        return dict(dist), {v: tuple(reversed(p)) for v, p in rev_paths.items()}

    def reach_from_edge(self, net: PhysicalNetwork, part: Partition, topology: Any) -> EdgeReach:
        """Edge router'dan ters Dijkstra (C, en ucuz yol) + BFS (H): bütün ulaşabilen düğümler (§5.9, §5.17)."""
        t = _topology(topology)
        g = self.routing_graph(net, part, t)
        edge_id = g.graph["edge_router"]
        cost, paths = self._paths_to(g, edge_id)
        hops = dict(nx.single_source_shortest_path_length(g.reverse(copy=False), edge_id))  # §5.17 H = en az atlama
        del cost[edge_id], paths[edge_id], hops[edge_id]
        candidates = frozenset(v for v in part.members[t] if v in paths)
        return EdgeReach(t, edge_id, self.relay_mode, hops, cost, paths, candidates)

    def plan_routes(
        self, net: PhysicalNetwork, part: Partition, topology: Any, root: int, reach: EdgeReach | None = None
    ) -> RoutePlan:
        """İki aşamalı yol planı (§5.4, §5.9): root'tan ters Dijkstra ile sensör -> root, reach ile root -> edge router.

        root: topolojinin canlı, edge router'a yolu olan bir üyesi olmalı (§5.17 aday), değilse ValueError.
        Üye sınıflaması: to_root (yol var) / bypass (root'a yok, edge'e var) / no_route (ikisi de yok).
        """
        t = _topology(topology)
        if reach is None:
            reach = self.reach_from_edge(net, part, t)
        if root not in reach.candidates:
            raise ValueError(f"root {root} topoloji {t.value} için aday değil (canlı, üye ve edge router'a yolu olan olmalı)")
        g = self.routing_graph(net, part, t)
        to_root_cost, to_root = self._paths_to(g, root)
        plan_to_root: dict[int, tuple[int, ...]] = {}
        plan_cost: dict[int, float] = {}
        bypass: dict[int, tuple[int, ...]] = {}
        no_route: set[int] = set()
        for v in part.members[t]:
            if not net.nodes[v].alive:
                continue
            if v in to_root:
                plan_to_root[v] = to_root[v]
                plan_cost[v] = to_root_cost[v]
            elif v in reach.paths:
                bypass[v] = reach.paths[v]
            else:
                no_route.add(v)
        if bypass or no_route:
            log.debug("topoloji %s root %d: bypass %d %s, no_route %d %s (§5.4)",
                      t.value, root, len(bypass), sorted(bypass), len(no_route), sorted(no_route))
        return RoutePlan(
            topology=t,
            root=root,
            edge_router=reach.edge_router,
            relay_mode=self.relay_mode,
            root_to_edge=reach.paths[root],
            root_to_edge_cost=reach.cost[root],
            to_root=plan_to_root,
            to_root_cost=plan_cost,
            bypass=bypass,
            no_route=frozenset(no_route),
        )

    def plan_all(self, net: PhysicalNetwork, part: Partition, roots: dict[Any, int]) -> dict[Topology, RoutePlan]:
        """Her topoloji için plan (roots: topoloji -> root)."""
        return {_topology(t): self.plan_routes(net, part, t, r) for t, r in roots.items()}


def static_root(net: PhysicalNetwork, reach: EdgeReach) -> int:
    """§5.12 sabit root kuralı (nearest_to_edge): adaylar arasında edge router'a Öklid mesafesi en küçük düğüm."""
    if not reach.candidates:
        raise ValueError(f"topoloji {reach.topology.value}: aday yok, sabit root seçilemez")
    return min(reach.candidates, key=lambda v: (net.distance(v, reach.edge_router), v))


def relays_on_paths(paths: Iterable[Sequence[int]]) -> set[int]:
    """Yollarda aktarıcı olarak görev alan düğümler (uç noktalar hariç) — sıcak nokta / shared mod analizi."""
    relays: set[int] = set()
    for p in paths:
        relays.update(p[1:-1])
    return relays

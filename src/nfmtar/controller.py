"""V-SDN controller (minimal M9): dinamik root seçimi, tetikleyiciler ve histerezis (CLAUDE.md §5.8, §5.17, §5.18).

`DynamicRootPolicy`, simulation.py'deki `RootPolicy` arayüzünü uygular; simülasyon onu her yol güncellemesinde çağırır.
Karar (engine: bulanık skor + rank) ile uygulama (simülasyonun yol planı) ayrıdır (§5.8).

Uygulanan kararlar:
- Tetikleyiciler (§5.18): (1) periyodik — simülasyon topoloji başına π11'de bir yol güncellemesi yapar ve bu politikayı
  çağırır; (3) root sağlığı — root öldü (aday değil), tükenme > depletion_threshold (0,9) veya edge router'a yolu yok
  (aday değil): r* hemen seçilir, marj/bekleme atlanır. Ölüm de simülasyonda anında yol güncellemesi tetikler.
  (2) trafik tetikleyicisi bu minimal sürümde YOK (sınırlılık).
- Değiştirme kuralı (§5.18): değiştir ⇔ R(current) − R(r*) > ε VE son değişiklikten ≥ min_dwell (= 2 × π11)
  VE (confidence_gate kapalı VEYA güven(r*) ≥ conf_min). Varsayılan ε = 0,05, güven kapısı kapalı.
- Skor (§5.14): engine/fuzzy.extract_features (nominal trafik hızları §5.6, o anki kuyruk doluluğu ve enerji) +
  FuzzyStrategy.score_batch; rank (§5.17): engine/rank.py, H = EdgeReach.hops, C = EdgeReach.cost.
- Her değişim `changes` listesine yazılır (zaman, topoloji, eski/yeni, R farkı, tetikleyici).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from nfmtar.config import Config
from nfmtar.engine.baselines import FuzzyStrategy, strategy_for_method
from nfmtar.engine.fuzzy import extract_features, nominal_rates_pps
from nfmtar.engine.rank import RankResult, rank_candidates
from nfmtar.network.graph import Topology

log = logging.getLogger(__name__)


class DynamicRootPolicy:
    """§5.17 rank + §5.18 histerezis ile root seçimi (B3: uzman bulanık; NF-MTAR: eğitilmiş, M8)."""

    name = "dynamic"

    def __init__(self, cfg: Config, strategy: FuzzyStrategy, prr, rates_pps: dict[int, float]) -> None:
        ctl = cfg.controller
        self.cfg = cfg
        self.strategy = strategy
        self.prr = prr
        self.rates = rates_pps
        sw = ctl["switching"]
        self.epsilon = float(sw["epsilon"])
        self.dwell_mult = float(sw["min_dwell_multiplier"])
        self.confidence_gate = bool(sw["confidence_gate"])
        self.conf_min = float(sw["conf_min"])
        health = ctl["triggers"]["root_health"]
        self.health_enabled = bool(health["enabled"])
        self.depletion_threshold = float(health["depletion_threshold"])
        self.weights = {t: tuple(cfg.rank["weights"][t.value]) for t in Topology}
        self.last_change: dict[Topology, float] = {}
        self.changes: list[dict[str, Any]] = []
        self.last_rank: dict[Topology, RankResult] = {}

    @classmethod
    def for_method(cls, cfg: Config, method: str, radio, net, params=None) -> "DynamicRootPolicy":
        strategy = strategy_for_method(cfg, method, params)
        if not isinstance(strategy, FuzzyStrategy):
            raise ValueError(f"{method}: dinamik root bulanık puanlama ister (§5.12)")
        return cls(cfg, strategy, radio.prr, nominal_rates_pps(cfg, net))

    def min_dwell_s(self, topology: Topology) -> float:
        """min_dwell = 2 × π11 (§5.18)."""
        return self.dwell_mult * float(self.cfg.policy.for_topology(topology).route_update_s)

    def rank(self, topology: Topology, net, part, reach) -> RankResult:
        """Karar: adayların bulanık skoru ve rank'ı (§5.14, §5.17)."""
        occupancy = {v: net.nodes[v].queue.occupancy for v in reach.candidates}
        feats = extract_features(self.strategy.engine.spec, net, part, topology, self.rates, self.prr,
                                 occupancy, reach.hops)
        cands = sorted(reach.candidates)
        out = self.strategy.score_batch([feats[v] for v in cands])
        return rank_candidates(cands, out.score, out.confidence, [reach.hops[v] for v in cands],
                               [reach.cost[v] for v in cands], self.weights[topology])

    def select(self, topology, net, part, reach, current: int | None, now: float) -> int | None:
        topology = Topology(topology)
        if not reach.candidates:
            return None
        rr = self.rank(topology, net, part, reach)
        self.last_rank[topology] = rr
        best = rr.best
        since = now - self.last_change.get(topology, -math.inf)
        if current is None:
            reason = "initial"
        elif current not in reach.candidates:                      # root öldü veya edge yolu yok
            if self.health_enabled:
                reason = "health:dead_or_no_edge_path"
            elif since >= self.min_dwell_s(topology):              # tetikleyici kapalı: R(ölü) = ∞, dwell geçerli
                reason = "periodic:dead_root"
            else:
                return current
        elif self.health_enabled and net.nodes[current].depletion > self.depletion_threshold:
            reason = "health:depletion"
        else:
            delta = rr.rank_of(current) - rr.rank_of(best)
            conf_ok = (not self.confidence_gate) or rr.confidence_of(best) >= self.conf_min
            if best != current and delta > self.epsilon and since >= self.min_dwell_s(topology) and conf_ok:
                reason = "periodic:hysteresis"
            else:
                return current
        if best != current:
            delta = rr.rank_of(current) - rr.rank_of(best) if current in rr.candidates else math.inf
            self.changes.append({"time_s": now, "topology": topology.value, "old": current, "new": best,
                                 "delta_R": delta, "trigger": reason})
            self.last_change[topology] = now
            log.info("t=%.2f %s: root %s -> %s (%s, dR=%.3f)", now, topology.value, current, best, reason, delta)
        return best

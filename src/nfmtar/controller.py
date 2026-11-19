"""V-SDN controller (minimal M9): dinamik root seçimi, tetikleyiciler ve histerezis (CLAUDE.md §5.8, §5.17, §5.18).

`DynamicRootPolicy`, simulation.py'deki `RootPolicy` arayüzünü uygular; simülasyon onu her yol güncellemesinde çağırır.
Karar (engine: bulanık skor + rank) ile uygulama (simülasyonun yol planı) ayrıdır (§5.8).

Uygulanan kararlar:
- Tetikleyiciler (§5.18): (1) periyodik — simülasyon topoloji başına π11'de bir yol güncellemesi yapar ve bu politikayı
  çağırır; (3) root sağlığı — root öldü (aday değil), tükenme > depletion_threshold (0,9) veya edge router'a yolu yok
  (aday değil): r* hemen seçilir, marj/bekleme atlanır. Ölüm de simülasyonda anında yol güncellemesi tetikler.
  (2) trafik — `TrafficMonitor` (sınıf başına kayan pencere, Poisson anlamlılık koşuluyla); simülasyon her
  check_interval_s'de kontrol eder ve tetiklenen sınıfın topolojisi için bu politikayı çağırır (değiştirme kuralı
  periyodikle aynı: ε, min_dwell, güven kapısı).
- Değiştirme kuralı (§5.18): değiştir ⇔ R(current) − R(r*) > ε VE son değişiklikten ≥ min_dwell (= 2 × π11)
  VE (confidence_gate kapalı VEYA güven(r*) ≥ conf_min). Varsayılan ε = 0,2 (§5.21, K9), güven kapısı kapalı.
- Skor (§5.14): engine/fuzzy.extract_features (nominal trafik hızları §5.6, radyo meşguliyeti x2 ve enerji) +
  FuzzyStrategy.score_batch; rank (§5.17): engine/rank.py, H = EdgeReach.hops, C = EdgeReach.cost.
- Her değişim `changes` listesine yazılır (zaman, topoloji, eski/yeni, R farkı, tetikleyici türü, kural);
  simülasyonun root değişim logu aynı tetikleyici türünü `last_decision`'dan alır.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from nfmtar.config import Config
from nfmtar.engine.baselines import FuzzyStrategy, strategy_for_method
from nfmtar.engine.fuzzy import extract_features, nominal_rates_pps
from nfmtar.engine.rank import RankResult, rank_candidates
from nfmtar.network.graph import Topology

log = logging.getLogger(__name__)


class TrafficMonitor:
    """§5.18 tetikleyici 2: sınıf başına kayan pencerede varış hızı ve göreli değişim.

    λ_c(t) = (t − Δt, t] aralığında oluşan c sınıfı paket sayısı / Δt. Her kontrol anında (check_interval_s'de bir)
    |λ_c(t) − λ_c(t − Δt)| / max(λ_c(t − Δt), ε0) > τ_traffic ise c tetiklenir. Pencere kayar: iki ardışık pencere
    (t − 2Δt, t − Δt] ve (t − Δt, t] her kontrol anında yeniden sayılır. Nedensel: yalnızca `observe` ile o ana kadar
    bildirilmiş varışları kullanır (izin geleceğini görmez). Tetiklenen sınıf, karşılaştırma penceresi tazelensin diye
    cooldown_s boyunca yeniden tetiklenmez. İlk 2Δt'de (λ(t − Δt) tanımsız) tetiklenmez.
    Poisson anlamlılık koşulu (§5.18 sapması, 18 Kasım): ayrıca |n(t) − n(t − Δt)| > z·sqrt(n(t) + n(t − Δt)) olmalı
    (n = penceredeki paket sayısı; iki Poisson sayımının farkının std'si). Gerekçe: z olmadan durağan trafikte düşük
    hızlı sınıflar gürültüyle sürekli tetikleniyordu (kritik olmayan, yük 0,25: koşu başına ~41); z = 3 ile ~1.
    z = 0 koşulu kapatır (makaledeki saf formül).
    """

    def __init__(self, classes, delta_t_s: float, tau: float, eps0_pps: float, cooldown_s: float,
                 significance_z: float = 0.0) -> None:
        if delta_t_s <= 0:
            raise ValueError("delta_t_s > 0 olmalı")
        self.delta_t_s = float(delta_t_s)
        self.tau = float(tau)
        self.eps0 = float(eps0_pps)
        self.cooldown_s = float(cooldown_s)
        self.z = float(significance_z)
        self.times: dict[str, deque[float]] = {c: deque() for c in classes}
        self.last_fire: dict[str, float] = {c: -math.inf for c in classes}
        self.events: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg: Config, classes) -> "TrafficMonitor":
        tr = cfg.controller["triggers"]["traffic"]
        dt = float(tr["delta_t_s"])
        return cls(classes, dt, float(tr["tau_traffic"]), float(tr["eps0_pps"]), float(tr.get("cooldown_s", dt)),
                   float(tr.get("significance_z", 0.0)))

    def observe(self, traffic_class: str, t: float) -> None:
        """Kaynakta yeni paket oluştu (§5.6 izinden, oluşturulma anında)."""
        self.times[traffic_class].append(t)

    def rates(self, traffic_class: str, now: float) -> tuple[float, float]:
        """(λ(t − Δt), λ(t)) paket/s; 2Δt'den eski varışlar atılır."""
        q = self.times[traffic_class]
        dt = self.delta_t_s
        while q and q[0] <= now - 2 * dt:
            q.popleft()
        n_now = sum(1 for x in q if x > now - dt)
        return (len(q) - n_now) / dt, n_now / dt

    def check(self, now: float) -> list[str]:
        """Kontrol anı: tetiklenen sınıfları döndürür ve `events`'e yazar."""
        fired: list[str] = []
        for c in self.times:
            lam_prev, lam_now = self.rates(c, now)
            if now < 2 * self.delta_t_s or now - self.last_fire[c] < self.cooldown_s:
                continue
            change = abs(lam_now - lam_prev) / max(lam_prev, self.eps0)
            n_prev, n_now = lam_prev * self.delta_t_s, lam_now * self.delta_t_s
            significant = abs(n_now - n_prev) > self.z * math.sqrt(n_now + n_prev)
            if change > self.tau and significant:
                self.last_fire[c] = now
                fired.append(c)
                self.events.append({"time_s": now, "traffic_class": c, "lambda_prev_pps": lam_prev,
                                    "lambda_now_pps": lam_now, "relative_change": change})
        return fired


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
        self.traffic_trigger = bool(ctl["triggers"]["traffic"]["enabled"])   # §5.18 (2); simülasyon izler
        self.weights = {t: tuple(cfg.rank["weights"][t.value]) for t in Topology}
        self.last_change: dict[Topology, float] = {}
        self.changes: list[dict[str, Any]] = []
        self.last_rank: dict[Topology, RankResult] = {}
        self.last_decision: dict[str, Any] = {}
        self.busy_tracker = None              # §5.14 x2: simülasyon RadioBusyTracker'ını bağlar (yoksa x2 = 0)

    @classmethod
    def for_method(cls, cfg: Config, method: str, radio, net, params=None) -> "DynamicRootPolicy":
        if params is None and cfg.engine["methods"][method]["scoring"] == "fuzzy_trained":
            from nfmtar.config import PROJECT_ROOT
            from nfmtar.engine.fuzzy import FuzzyParams

            params = FuzzyParams.load_json(PROJECT_ROOT / cfg.engine["trained_model"])   # NF-MTAR (§5.16)
        strategy = strategy_for_method(cfg, method, params)
        if not isinstance(strategy, FuzzyStrategy):
            raise ValueError(f"{method}: dinamik root bulanık puanlama ister (§5.12)")
        return cls(cfg, strategy, radio.prr, nominal_rates_pps(cfg, net))

    def min_dwell_s(self, topology: Topology) -> float:
        """min_dwell = 2 × π11 (§5.18)."""
        return self.dwell_mult * float(self.cfg.policy.for_topology(topology).route_update_s)

    def rank(self, topology: Topology, net, part, reach, now: float = 0.0) -> RankResult:
        """Karar: adayların bulanık skoru ve rank'ı (§5.14, §5.17); x2 = radyo meşguliyeti (busy_tracker, now)."""
        util = self.busy_tracker.busy_ratios(reach.candidates, now) if self.busy_tracker is not None else {}
        feats = extract_features(self.strategy.engine.spec, net, part, topology, self.rates, self.prr,
                                 util, reach.hops)
        cands = sorted(reach.candidates)
        out = self.strategy.score_batch([feats[v] for v in cands])
        return rank_candidates(cands, out.score, out.confidence, [reach.hops[v] for v in cands],
                               [reach.cost[v] for v in cands], self.weights[topology])

    def select(self, topology, net, part, reach, current: int | None, now: float,
               trigger: str = "periodic") -> int | None:
        """trigger: yeniden değerlendirmeyi başlatan olay — "initial", "periodic" (§5.18-1), "traffic" (§5.18-2),
        "node_death" (§5.10a: bir düğüm öldü). Root sağlığı (§5.18-3) her çağrıda ayrıca denetlenir; root değişirse
        `last_decision` = {trigger, rule, delta_R}: trigger root sağlığı ise "root_health", değilse çağıran olay."""
        topology = Topology(topology)
        if not reach.candidates:
            return None
        rr = self.rank(topology, net, part, reach, now)
        self.last_rank[topology] = rr
        best = rr.best
        since = now - self.last_change.get(topology, -math.inf)
        if current is None:
            trig, rule = "initial", "initial"
        elif current not in reach.candidates:                      # root öldü veya edge yolu yok
            if self.health_enabled:
                trig, rule = "root_health", "dead_or_no_edge_path"
            elif since >= self.min_dwell_s(topology):              # tetikleyici kapalı: R(ölü) = ∞, dwell geçerli
                trig, rule = trigger, "dead_root"
            else:
                return current
        elif self.health_enabled and net.nodes[current].depletion > self.depletion_threshold:
            trig, rule = "root_health", "depletion"
        else:
            delta = rr.rank_of(current) - rr.rank_of(best)
            conf_ok = (not self.confidence_gate) or rr.confidence_of(best) >= self.conf_min
            if best != current and delta > self.epsilon and since >= self.min_dwell_s(topology) and conf_ok:
                trig, rule = trigger, "hysteresis"
            else:
                return current
        if best != current:
            delta = rr.rank_of(current) - rr.rank_of(best) if current in rr.candidates else math.inf
            self.last_decision = {"trigger": trig, "rule": rule, "delta_R": delta}
            self.changes.append({"time_s": now, "topology": topology.value, "old": current, "new": best,
                                 **self.last_decision})
            self.last_change[topology] = now
            log.info("t=%.2f %s: root %s -> %s (%s/%s, dR=%.3f)", now, topology.value, current, best, trig, rule, delta)
        return best

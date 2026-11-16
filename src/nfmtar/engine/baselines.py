"""Yöntemlerin ScoringStrategy sarmalayıcıları: B1-plain, B1, B2 (sabit root) ve B3 (eğitilmemiş bulanık) (CLAUDE.md §5.11, §5.12, §5.15).

Uygulanan kararlar:
- §5.11 bütün yöntemler aynı arayüzü uygular; yöntemler arasındaki fark config engine.methods'tadır
  (scoring, root, path_cost, vsdn_policy).
- §5.12 B1-plain / B1 / B2: puanlama YOK, root sabit (`dynamic_root() = False`). Root, yöntemden bağımsız
  `router.static_root` kuralıyla seçilir (edge router'a Öklid en yakın aday; root ölürse aynı kuralla yenisi);
  `select_root` bu kurala yönlendirir. Sabit root'lu yöntemde `score` çağrılması bir hatadır (rank yalnızca dinamik
  root'lu yöntemlerde çalışır, §5.17) ve sessizce yanlış root üretmemesi için RuntimeError verir.
- §5.15 B3: §5.14 motoru UZMAN parametreleriyle (FuzzyParams.expert); rank (§5.17) ve histerezis (§5.18) NF-MTAR ile
  aynıdır (controller/rank tarafında, M9). `dynamic_root() = True`.
- NF-MTAR aynı sınıfı eğitilmiş parametrelerle kullanır (`FuzzyStrategy(name="NF-MTAR", params=...)`; M8 üretir).
"""

from __future__ import annotations

from typing import Any, Sequence

from nfmtar.config import Config
from nfmtar.engine.base import NodeFeatures, ScoreOutput
from nfmtar.engine.fuzzy import FuzzyBatchOutput, FuzzyEngine, FuzzyParams

STATIC_SCORING = "none"
FUZZY_SCORINGS = ("fuzzy_untrained", "fuzzy_trained")


class StaticRootStrategy:
    """§5.12 sabit root'lu yöntem (B1-plain, B1, B2): puanlama yok, root `router.static_root` ile."""

    def __init__(self, name: str, path_cost: str, vsdn_policy: bool) -> None:
        self.name = name
        self.path_cost = path_cost
        self.vsdn_policy = vsdn_policy

    def dynamic_root(self) -> bool:
        return False

    def score(self, features: NodeFeatures) -> ScoreOutput:
        raise RuntimeError(f"{self.name}: sabit root'lu yöntem puanlama yapmaz (§5.12); root static_root kuralıyla seçilir")

    @staticmethod
    def select_root(net: Any, reach: Any) -> int:
        """§5.12 kuralı: router.static_root (edge router'a Öklid en yakın aday, eşitlikte küçük id)."""
        from nfmtar.router import static_root

        return static_root(net, reach)

    def __repr__(self) -> str:
        return f"StaticRootStrategy({self.name!r}, path_cost={self.path_cost!r}, vsdn_policy={self.vsdn_policy})"


class FuzzyStrategy:
    """§5.14 motorunu ScoringStrategy olarak saran sınıf; B3 (uzman parametreleri, §5.15) ve NF-MTAR (eğitilmiş)."""

    def __init__(self, engine: FuzzyEngine, name: str = "B3", path_cost: str = "weighted", vsdn_policy: bool = True) -> None:
        self.engine = engine
        self.name = name
        self.path_cost = path_cost
        self.vsdn_policy = vsdn_policy

    def dynamic_root(self) -> bool:
        return True

    def score(self, features: NodeFeatures) -> ScoreOutput:
        """Skor = f_norm (Denk. 13a), güven (Denk. 13b), en baskın kural."""
        return self.engine.evaluate(features).to_score_output()

    def score_batch(self, features: Sequence[NodeFeatures]) -> FuzzyBatchOutput:
        """Bir karar turundaki bütün adaylar tek seferde (§5.14 toplu değerlendirme)."""
        return self.engine.evaluate_batch(list(features))

    def __repr__(self) -> str:
        return f"FuzzyStrategy({self.name!r}, rules={self.engine.n_rules}, pruned={self.engine.pruned})"


def b3_strategy(cfg: Config) -> FuzzyStrategy:
    """§5.15 B3: uzman parametreli motor (config fuzzy bölümü)."""
    spec = cfg.engine["methods"]["B3"]
    return FuzzyStrategy(FuzzyEngine.from_config(cfg), name="B3", path_cost=spec["path_cost"], vsdn_policy=bool(spec["vsdn_policy"]))


def strategy_for_method(cfg: Config, method: str, params: FuzzyParams | None = None) -> StaticRootStrategy | FuzzyStrategy:
    """engine.methods[method]'a göre strateji (§5.12 tablosu). fuzzy_trained (NF-MTAR) eğitilmiş params ister (M8)."""
    try:
        spec = cfg.engine["methods"][method]
    except KeyError as exc:
        raise KeyError(f"bilinmeyen yöntem {method!r}; engine.methods: {sorted(cfg.engine['methods'])}") from exc
    scoring, path_cost, vsdn = spec["scoring"], spec["path_cost"], bool(spec["vsdn_policy"])
    if scoring == STATIC_SCORING:
        if spec["root"] != "static":
            raise ValueError(f"{method}: puanlamasız yöntemin root'u sabit olmalı (§5.12)")
        return StaticRootStrategy(method, path_cost, vsdn)
    if scoring == "fuzzy_untrained":
        if params is not None:
            raise ValueError(f"{method}: eğitilmemiş bulanık yöntem uzman parametreleriyle çalışır (§5.15)")
        return FuzzyStrategy(FuzzyEngine.from_config(cfg), method, path_cost, vsdn)
    if scoring == "fuzzy_trained":
        if params is None:
            raise ValueError(f"{method}: eğitilmiş parametreler gerekli (results/models/<run_id>.json, §5.16; M8)")
        return FuzzyStrategy(FuzzyEngine.from_config(cfg, params), method, path_cost, vsdn)
    raise ValueError(f"{method}: bilinmeyen puanlama {scoring!r}")

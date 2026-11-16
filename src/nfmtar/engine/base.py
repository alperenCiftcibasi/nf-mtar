"""Engine arayüzü: ScoringStrategy (Protocol), NodeFeatures ve ScoreOutput (CLAUDE.md §5.11, §5.14).

Uygulanan kararlar:
- §5.11 bütün yöntemler aynı arayüzü uygular; yalnızca engine değişir (ağ, trafik, metrik aynı).
  `score(features) -> ScoreOutput`: yüksek skor = root için uygun. `dynamic_root()`: root'u engine mi seçer
  (B3, NF-MTAR; §5.17 rank + §5.18 histerezis) yoksa §5.12 sabit root kuralı mı (B1-plain, B1, B2).
- §5.14 NodeFeatures: beş girdi x1–x5, hepsi sabit normalizasyonla [0,1] (normalizasyon fuzzy.py'de,
  `extract_features`); ayrıca rank için edge router'a EN AZ atlama sayısı H (§5.17; BFS, router.EdgeReach.hops).
  Girdi sırası (x1, ..., x5) bulanık motorun kural-terim tablosunun sütun sırasıdır.
- §5.11 ScoreOutput: score (bulanık motorda f_norm, Denk. 13a), confidence (Denk. 13b, opsiyonel),
  top_rule (en yüksek ateşleme gücüne sahip kuralın indeksi, opsiyonel; canlı demo / log).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

FEATURE_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5")
FEATURE_LABELS: dict[str, str] = {   # §5.14 girdi tablosu; rule_to_text ve demo için
    "x1": "demand",
    "x2": "utilization",
    "x3": "depletion",
    "x4": "link_quality",
    "x5": "connectivity",
}


@dataclass(frozen=True)
class NodeFeatures:
    """Bir aday düğümün bulanık girdileri (§5.14) ve rank için edge router'a atlama mesafesi (§5.17).

    x1 talep (yüksek = iyi), x2 kuyruk kullanımı (yüksek = kötü), x3 enerji tükenmesi (yüksek = kötü),
    x4 topoloji içi ortalama link kalitesi (yüksek = iyi), x5 bağlantısallık (yüksek = iyi); hepsi [0,1].
    hops_to_edge: edge router'a en az atlama (H); yol yoksa None (o düğüm root adayı değildir, §5.17).
    """

    node_id: int
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    hops_to_edge: int | None = None

    def __post_init__(self) -> None:
        for name in FEATURE_NAMES:
            value = getattr(self, name)
            if math.isnan(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"düğüm {self.node_id}: {name} = {value} [0,1] dışında (§5.14 sabit normalizasyon)")

    def as_array(self) -> np.ndarray:
        """(x1, ..., x5) vektörü; bulanık motorun girdi sırası."""
        return np.array([self.x1, self.x2, self.x3, self.x4, self.x5], dtype=float)


@dataclass(frozen=True)
class ScoreOutput:
    """Puanlama sonucu (§5.11): score ∈ [0,1] yüksek = root için uygun; confidence ve top_rule opsiyonel."""

    score: float
    confidence: float | None = None
    top_rule: int | None = None


@runtime_checkable
class ScoringStrategy(Protocol):
    """§5.11 ortak engine arayüzü."""

    name: str

    def score(self, features: NodeFeatures) -> ScoreOutput: ...

    def dynamic_root(self) -> bool: ...


def features_matrix(features: list[NodeFeatures]) -> np.ndarray:
    """Toplu değerlendirme için (B, 5) matris (§5.14 vektörize çıkarım)."""
    if not features:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.stack([f.as_array() for f in features])

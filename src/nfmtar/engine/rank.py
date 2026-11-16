"""Rank ve root seçimi (CLAUDE.md §5.17): R(v) = w1·(1 − f_norm(v)) + w2·H(v)/H_max + w3·C(v)/C_max ; r* = argmin R.

- (1 − f): makaledeki işaret hatasının düzeltmesi (yüksek bulanık skor = düşük rank = iyi root).
- H: edge router'a EN AZ atlama (BFS, router.EdgeReach.hops); C: en ucuz yolun §5.9 maliyeti (EdgeReach.cost).
- H_max, C_max: o turdaki adaylar arasında en büyük (0 ise 1 alınır).
- Ağırlıklar topolojiye özgü, config rank.weights (E 0.4/0.4/0.2, L 0.5/0.3/0.2, R 0.6/0.2/0.2).
- Eşitlikte küçük düğüm kimliği (belirlenimci).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RankResult:
    """Bir karar turunun aday tablosu (§5.17 log: R, f_norm, güven, H, C) ve seçilen r*."""

    candidates: tuple[int, ...]
    R: np.ndarray
    f_norm: np.ndarray
    confidence: np.ndarray
    H: np.ndarray
    C: np.ndarray

    @property
    def best(self) -> int:
        return self.candidates[int(np.argmin(self.R))]

    def rank_of(self, v: int) -> float:
        return float(self.R[self.candidates.index(v)])

    def confidence_of(self, v: int) -> float:
        return float(self.confidence[self.candidates.index(v)])


def rank_values(f_norm: np.ndarray, H: np.ndarray, C: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """§5.17 R = w1(1 − f) + w2·H/H_max + w3·C/C_max (vektörize)."""
    w1, w2, w3 = weights
    H = np.asarray(H, dtype=float)
    C = np.asarray(C, dtype=float)
    h_max = H.max() if len(H) and H.max() > 0 else 1.0
    c_max = C.max() if len(C) and C.max() > 0 else 1.0
    return w1 * (1.0 - np.asarray(f_norm, dtype=float)) + w2 * H / h_max + w3 * C / c_max


def rank_candidates(candidates: Sequence[int], f_norm, confidence, H, C, weights: Sequence[float]) -> RankResult:
    """Adayları kimlik sırasıyla sıralayıp R hesaplar (argmin eşitliğinde küçük id kazanır)."""
    order = np.argsort(np.asarray(candidates))
    cands = tuple(int(candidates[i]) for i in order)
    f = np.asarray(f_norm, dtype=float)[order]
    conf = np.asarray(confidence, dtype=float)[order]
    h = np.asarray(H, dtype=float)[order]
    c = np.asarray(C, dtype=float)[order]
    return RankResult(cands, rank_values(f, h, c, weights), f, conf, h, c)

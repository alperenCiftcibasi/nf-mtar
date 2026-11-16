"""Yorumlanabilirlik ölçütleri (CLAUDE.md §5.20, K1): eğitim terimlerin anlamını bozuyor mu?

Bir FuzzyParams için (uzman parametreleri θ0 ile karşılaştırarak):
- order_violations: her girdide komşu merkezlerin sırası bozulan çift sayısı (c_j > c_{j+1}); toplam.
- coverage_min: [0,1] üzerinde (1000 nokta) max_terim μ(x)'in en küçüğü, girdiler arasında en küçük; < 0,5 boşluk.
- distinguishability_max: komşu terimlerin kesişim yüksekliği max_x min(μ_j, μ_{j+1}), bütün komşu çiftler arasında
  en büyük (uzman 0,5; 1'e yaklaştıkça iki terim aynı anlamda).
- monotonicity_violation: tek girdide komşu terimle farklılaşan (diğer terimleri aynı) kural çiftlerinden y_k
  değişimi uzman yönüne (§5.14 "Yön": high_good -> artmalı, high_bad -> azalmalı) ters olanların oranı.
- dev_c / dev_sigma / dev_y: uzmandan ortalama mutlak sapma |Δc|, |Δσ|, |Δy|.
"""

from __future__ import annotations

import numpy as np

from nfmtar.engine.fuzzy import HIGH_GOOD, FuzzyParams, FuzzySpec, gaussian_mf

GRID = np.linspace(0.0, 1.0, 1000)


def order_violations(p: FuzzyParams) -> int:
    return int(sum(int(np.sum(np.diff(c) < 0)) for c in p.centers))


def coverage_min(p: FuzzyParams) -> float:
    return float(min(gaussian_mf(GRID[:, None], c[None, :], s[None, :]).max(axis=1).min()
                     for c, s in zip(p.centers, p.sigmas)))


def distinguishability_max(p: FuzzyParams) -> float:
    heights = []
    for c, s in zip(p.centers, p.sigmas):
        mu = gaussian_mf(GRID[:, None], c[None, :], s[None, :])
        for j in range(len(c) - 1):
            heights.append(float(np.minimum(mu[:, j], mu[:, j + 1]).max()))
    return max(heights)


def monotonicity_violation(spec: FuzzySpec, p: FuzzyParams) -> float:
    """Komşu terim çiftlerinde y değişiminin uzman yönüne ters olma oranı (tam kural tablosu gerekir)."""
    index = {tuple(r): k for k, r in enumerate(p.table)}
    bad = total = 0
    for k, row in enumerate(p.table):
        for i, (n, d) in enumerate(zip(spec.n_terms, spec.directions)):
            if row[i] + 1 >= n:
                continue
            nxt = list(row)
            nxt[i] += 1
            k2 = index[tuple(nxt)]
            dy = p.y[k2] - p.y[k]
            total += 1
            if (d == HIGH_GOOD and dy < 0) or (d != HIGH_GOOD and dy > 0):
                bad += 1
    return bad / total if total else float("nan")


def deviation(p: FuzzyParams, p0: FuzzyParams) -> dict[str, float]:
    return {
        "dev_c": float(np.mean(np.concatenate([np.abs(a - b) for a, b in zip(p.centers, p0.centers)]))),
        "dev_sigma": float(np.mean(np.concatenate([np.abs(a - b) for a, b in zip(p.sigmas, p0.sigmas)]))),
        "dev_y": float(np.mean(np.abs(p.y - p0.y))),
    }


def interpretability_report(spec: FuzzySpec, p: FuzzyParams) -> dict[str, float]:
    """§5.20 beş ölçüt (sapma üç bileşenli)."""
    return {
        "order_violations": order_violations(p),
        "coverage_min": coverage_min(p),
        "distinguishability_max": distinguishability_max(p),
        "monotonicity_violation": monotonicity_violation(spec, p),
        **deviation(p, FuzzyParams.expert(spec)),
    }

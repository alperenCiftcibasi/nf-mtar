"""Kâhin etiketleri: aday root için toplama kalitesi + dayanıklılık skoru (CLAUDE.md §5.16).

Bir ağ durumu ve topoloji için her aday r:
1. Toplama kalitesi: topolojideki her canlı sensörden r'ye §5.9 maliyetiyle en ucuz yol (yönlendirme çizgesinde, r'den
   ters yönlü tek Dijkstra). teslim = Π PRR (yol üzerindeki kenarlar), gecikme = Σ atlama gecikmesi tahmini;
   atlama gecikmesi tahmini = (sınıfın ortalama paket iletim süresi + beklenen ilk geri çekilme) / PRR (beklenen deneme
   sayısı) + işlem süresi. Sensör trafik hızlarıyla ağırlıklı ortalama; r'ye yolu olmayan sensör teslim 0 ve
   o turdaki en büyük gecikmeyle sayılır. r'nin kendisi teslim 1, gecikme 0.
2. Dayanıklılık: r'nin kalan enerjisi / r root olduğunda tahmini tüketim hızı. Tüketim = topolojinin toplam bit hızı ×
   (alım E_elec + gönderim E_elec + ε_amp·d²), d = r'nin edge router'a giden en ucuz yolundaki ilk atlama.
3. Skor = 0,4·teslim + 0,3·(1 − gecikme_norm) + 0,3·dayanıklılık_norm (config oracle.weights); gecikme ve dayanıklılık
   adaylar arasında min-max ile normalize; sonuç aynı durumdaki adaylar arasında min-max ile [0,1; 1,0]'a (score_range).
NOT: kâhin root -> edge router yolunu İÇERMEZ (o kısım rank'taki H ve C'de, §5.17).
"""

from __future__ import annotations

import math
from typing import Mapping

import networkx as nx
import numpy as np

from nfmtar.config import Config


def _minmax(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    span = x.max() - x.min() if len(x) else 0.0
    if span <= 0:
        return np.full(len(x), (lo + hi) / 2)
    return lo + (hi - lo) * (x - x.min()) / span


def oracle_scores(cfg: Config, router, net, part, topology, reach, rates_pps: Mapping[int, float]) -> dict[int, float]:
    """§5.16 kâhin skoru, aday -> [0,1; 1,0]. router: §5.9 ağırlıklı maliyetli Router; reach: EdgeReach (adaylar)."""
    radio = router.radio
    cands = sorted(reach.candidates)
    if not cands:
        return {}
    g = router.routing_graph(net, part, topology)
    sensors = [v for v in g.nodes if not net.nodes[v].is_edge_router]
    rate = {v: float(rates_pps.get(v, 0.0)) for v in sensors}
    total_rate = sum(rate.values()) or 1.0
    # sınıfın ortalama paket boyutu (§5.6) -> iletim süresi ve bit hızı
    cls = net.nodes[cands[0]].traffic_class
    pb = cfg.traffic.classes[cls].packet_bytes
    mean_bytes = int(round(float(np.mean(pb))))
    tx_s = radio.transmission_time_s(mean_bytes)
    csma = cfg.radio.csma
    backoff_s = ((2 ** csma.mac_min_be[cls] - 1) / 2 + csma.aifs_slots[cls]) * csma.backoff_slot_s
    proc_s = radio.processing_delay_s
    bits_per_s = total_rate * mean_bytes * 8
    rev = g.reverse(copy=False)

    deliv, lat, resil = [], [], []
    for r in cands:
        dist, paths = nx.single_source_dijkstra(rev, r, weight="cost")
        d_list, l_list, w_list = [], [], []
        for v in sensors:
            w = rate[v]
            if v == r:
                d_list.append(1.0); l_list.append(0.0); w_list.append(w)
                continue
            p = paths.get(v)
            if p is None:
                d_list.append(0.0); l_list.append(math.nan); w_list.append(w)
                continue
            p = p[::-1]                                     # v -> ... -> r
            prr = [g.edges[a, b]["prr"] for a, b in zip(p, p[1:])]
            d_list.append(float(np.prod(prr)))
            l_list.append(sum((tx_s + backoff_s) / q + proc_s for q in prr))
            w_list.append(w)
        w_arr = np.array(w_list) / (sum(w_list) or 1.0)
        l_arr = np.array(l_list)
        if np.isnan(l_arr).any():
            l_arr = np.where(np.isnan(l_arr), np.nanmax(l_arr) if np.isfinite(np.nanmax(l_arr)) else 0.0, l_arr)
        deliv.append(float(w_arr @ np.array(d_list)))
        lat.append(float(w_arr @ l_arr))
        path = reach.paths.get(r)
        d_next = net.distance(path[0], path[1]) if path and len(path) > 1 else 0.0
        e_bit = 2 * radio.e_elec_J_per_bit + radio.eps_amp_J_per_bit_m2 * d_next ** 2
        resil.append(net.nodes[r].energy_J / max(bits_per_s * e_bit, 1e-30))

    wts = cfg.oracle["weights"]
    score = (wts["delivery"] * np.array(deliv) + wts["latency"] * (1 - _minmax(np.array(lat)))
             + wts["resilience"] * _minmax(np.array(resil)))
    lo, hi = cfg.oracle["score_range"]
    return dict(zip(cands, _minmax(score, float(lo), float(hi)).tolist()))

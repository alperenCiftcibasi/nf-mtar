"""Trafik üretimi: sınıf bazlı varış süreçleri, paket boyutları, kritik bakım epizotları ve önceden üretilen iz (CLAUDE.md §5.6).

Uygulanan kararlar:
- §5.6 sınıf modeli (config traffic.classes):
    kritik (E):        64 B sabit;            Poisson, temel hız rate_pps + kritik bakım epizotları
    düzenli (L):       düzgün 128–256 B;      periyodik, periyot = 1/rate_pps, aralık = periyot × (1 + jitter_frac·U(−1,1)),
                                              ilk varış U(0, periyot) (random_phase)
    kritik olmayan (R): düzgün 512–1024 B;    Poisson, rate_pps
  Süreç türü ve parametreleri sınıf başına config'den gelir (`arrival`); kod sınıf adlarını gömmez.
- §5.6 kritik bakım epizotları (traffic.critical_episodes): epizotların uygulandığı sınıfın her düğümü için epizot
  başlangıçları Poisson (ortalama mean_interval_s'de bir), her epizot duration_s sürer, örtüşen epizotlar birleşir
  (pencerelerin birleşimi); pencere içinde varış hızı rate_pps (epizot hızı), dışında temel hız. Süreç parçalı sabit
  hızlı Poisson'dur: her parçada üstel aralıklar parça başından yeniden başlatılır (belleksizlik → kesin).
  Beklenen düğüm başına uzun dönem hız (`expected_rate_pps`): temel·(1 − p_on) + epizot·p_on,
  p_on = 1 − exp(−duration_s / mean_interval_s) (başlangıçların Poisson olması ve pencere birleşimi nedeniyle).
- §5.6 load_multiplier yalnızca temel hızları (rate_pps) ölçekler; epizot hızı ve epizot zamanlaması değişmez.
- §5.6 ortak rastgele sayılar: iz yöntemden BAĞIMSIZ olarak (seed, ağ, süre) üçlüsünden önceden üretilir ve bütün
  yöntemlere aynen verilir. Her düğüm kendi RNG akışını kullanır: `SeedSequence(seed).spawn(N)` ile düğüm başına bir
  çocuk; çocuk üç alt akışa bölünür (varışlar, boyutlar, epizotlar). Böylece bir sınıfın parametresini değiştirmek
  diğer sınıfların izini kaydırmaz; load_multiplier değişince epizot pencereleri aynı kalır.
- §5.6 paket kaydı (`Packet`): id, kaynak, sınıf, boyut, oluşturulma zamanı, son tarih = oluşturulma + π3
  (sınıfın topolojisinin max_delay_ms'i, §5.7); ayrıca epizot içinde üretilip üretilmediği (analiz için).
  Paket id'leri, bütün düğümlerin paketleri zamana göre birleştirildikten sonra sırayla verilir (eşitlikte kaynak id).

Bu modül yalnızca "hangi düğüm, ne zaman, kaç baytlık paket üretti" sorusunu yanıtlar; kuyruk, iletim ve kayıp
simulation.py / radio.py'dedir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from nfmtar.config import Config, EpisodeConfig, TrafficClassConfig, TrafficConfig
from nfmtar.network.graph import PhysicalNetwork

Window = tuple[float, float]


# --------------------------------------------------------------------------- #
# Paket kaydı ve iz
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Packet:
    """§5.6 paket kaydı: id, kaynak, sınıf, boyut, oluşturulma zamanı, son tarih (oluşturulma + π3); epizot bayrağı."""

    packet_id: int
    source: int
    traffic_class: str
    size_bytes: int
    created_s: float
    deadline_s: float
    in_episode: bool = False

    def key(self) -> tuple[int, str, int, float, float, bool]:
        """id hariç içerik (izler karşılaştırılırken; id'ler birleşik zaman sırasına bağlıdır)."""
        return (self.source, self.traffic_class, self.size_bytes, self.created_s, self.deadline_s, self.in_episode)


@dataclass(frozen=True)
class TrafficTrace:
    """Önceden üretilmiş varış izi (§5.6): zaman sıralı paketler + kaynak/sınıf dizinleri + epizot pencereleri."""

    packets: tuple[Packet, ...]
    by_source: dict[int, tuple[Packet, ...]]
    sources_by_class: dict[str, tuple[int, ...]]
    episodes: dict[int, tuple[Window, ...]]   # kaynak -> birleşik epizot pencereleri (yalnızca epizot sınıfı)
    duration_s: float
    seed: int
    load_multiplier: float

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.sources_by_class)

    def packets_of_class(self, traffic_class: str) -> list[Packet]:
        return [p for p in self.packets if p.traffic_class == traffic_class]

    def packets_between(self, t0: float, t1: float) -> list[Packet]:
        """[t0, t1) aralığında oluşturulan paketler (ısınma dışlama, pencere metrikleri; §5.19)."""
        return [p for p in self.packets if t0 <= p.created_s < t1]

    def mean_rate_pps(self, traffic_class: str, t0: float = 0.0, t1: float | None = None) -> float:
        """Sınıfın düğüm başına ölçülen ortalama hızı: paket sayısı / (düğüm sayısı × pencere) (§5.6 doğrulama)."""
        t1 = self.duration_s if t1 is None else t1
        n_sources = len(self.sources_by_class[traffic_class])
        if n_sources == 0 or t1 <= t0:
            return 0.0
        count = sum(1 for p in self.packets if p.traffic_class == traffic_class and t0 <= p.created_s < t1)
        return count / (n_sources * (t1 - t0))

    def episode_rate_pps(self, traffic_class: str, inside: bool) -> float:
        """Epizot pencerelerinin içinde (inside=True) veya dışında düğüm başına ölçülen hız (§5.6 doğrulama)."""
        total_time = 0.0
        count = 0
        for src in self.sources_by_class[traffic_class]:
            on_time = sum(b - a for a, b in self.episodes.get(src, ()))
            total_time += on_time if inside else self.duration_s - on_time
            count += sum(1 for p in self.by_source[src] if p.in_episode == inside)
        return count / total_time if total_time > 0 else float("nan")

    def to_dataframe(self) -> pd.DataFrame:
        """Paket logu tabanı (§5.19): bir satır bir paket."""
        return pd.DataFrame(
            {
                "packet_id": [p.packet_id for p in self.packets],
                "source": [p.source for p in self.packets],
                "traffic_class": [p.traffic_class for p in self.packets],
                "size_bytes": [p.size_bytes for p in self.packets],
                "created_s": [p.created_s for p in self.packets],
                "deadline_s": [p.deadline_s for p in self.packets],
                "in_episode": [p.in_episode for p in self.packets],
            }
        )

    def to_record(self) -> dict[str, Any]:
        """runs.csv alanları (§5.19): sınıf başına üretilen paket sayısı, düğüm başına hız ve sunulan yük (kbit/s)."""
        rec: dict[str, Any] = {
            "traffic_seed": self.seed,
            "load_multiplier": self.load_multiplier,
            "n_packets_total": len(self.packets),
        }
        for cls in self.classes:
            pk = self.packets_of_class(cls)
            rec[f"n_packets_{cls}"] = len(pk)
            rec[f"rate_pps_per_node_{cls}"] = self.mean_rate_pps(cls)
            rec[f"offered_kbps_{cls}"] = sum(p.size_bytes for p in pk) * 8 / self.duration_s / 1000.0
        rec["n_episodes"] = sum(len(w) for w in self.episodes.values())
        return rec


# --------------------------------------------------------------------------- #
# Yardımcılar: boyut, varış süreçleri, epizot pencereleri (§5.6)
# --------------------------------------------------------------------------- #


def draw_packet_bytes(rng: np.random.Generator, packet_bytes: int | tuple[int, int]) -> int:
    """§5.6 paket boyutu: sabit, veya [alt, üst] aralığında (kapalı) düzgün tam sayı."""
    if isinstance(packet_bytes, int):
        return packet_bytes
    lo, hi = packet_bytes
    return int(rng.integers(lo, hi + 1))


def poisson_arrivals(rng: np.random.Generator, rate_pps: float, t0: float, t1: float) -> list[float]:
    """[t0, t1) aralığında hızı rate_pps olan Poisson süreci: üstel aralıklar t0'dan başlar (§5.6)."""
    times: list[float] = []
    if rate_pps <= 0 or t1 <= t0:
        return times
    t = t0
    while True:
        t += rng.exponential(1.0 / rate_pps)
        if t >= t1:
            return times
        times.append(t)


def periodic_arrivals(
    rng: np.random.Generator, rate_pps: float, jitter_frac: float, random_phase: bool, t1: float
) -> list[float]:
    """[0, t1) aralığında periyodik varışlar (§5.6): ilk varış U(0, periyot) (random_phase) veya periyot;
    ardışık aralık = periyot × (1 + jitter_frac × U(−1, 1)). Aralıkların ortalaması periyot olduğundan uzun dönem
    hız 1/periyot = rate_pps kalır (titreşim yürüyüşü hızı kaydırmaz)."""
    period = 1.0 / rate_pps
    times: list[float] = []
    t = float(rng.uniform(0.0, period)) if random_phase else period
    while t < t1:
        times.append(t)
        t += period * (1.0 + jitter_frac * float(rng.uniform(-1.0, 1.0)))
    return times


def episode_windows(rng: np.random.Generator, ep: EpisodeConfig, t1: float) -> list[Window]:
    """Bir düğümün kritik bakım epizotları (§5.6): Poisson başlangıçlar, sabit süre, [0, t1)'e kırpılmış pencerelerin birleşimi."""
    starts = poisson_arrivals(rng, 1.0 / ep.mean_interval_s, 0.0, t1)
    merged: list[Window] = []
    for s in starts:
        a, b = s, min(s + ep.duration_s, t1)
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def piecewise_poisson_arrivals(
    rng: np.random.Generator, base_rate_pps: float, windows: Iterable[Window], episode_rate_pps: float, t1: float
) -> tuple[list[float], list[bool]]:
    """Pencere dışında base_rate, içinde episode_rate ile Poisson varışlar (§5.6); (zamanlar, epizot bayrakları)."""
    times: list[float] = []
    flags: list[bool] = []
    t = 0.0
    for a, b in windows:
        seg = poisson_arrivals(rng, base_rate_pps, t, a)
        times += seg
        flags += [False] * len(seg)
        seg = poisson_arrivals(rng, episode_rate_pps, a, b)
        times += seg
        flags += [True] * len(seg)
        t = b
    seg = poisson_arrivals(rng, base_rate_pps, t, t1)
    times += seg
    flags += [False] * len(seg)
    return times, flags


def episode_on_fraction(ep: EpisodeConfig) -> float:
    """Epizot pencerelerinin kapladığı beklenen zaman oranı: 1 − exp(−duration/mean_interval) (durağan rejim, §5.6)."""
    return 1.0 - math.exp(-ep.duration_s / ep.mean_interval_s)


def expected_rate_pps(traffic: TrafficConfig, traffic_class: str) -> float:
    """Düğüm başına beklenen uzun dönem hız (§5.6): temel × load_multiplier; epizot sınıfında pencere ağırlıklı ortalama."""
    base = traffic.classes[traffic_class].rate_pps * traffic.load_multiplier
    ep = traffic.critical_episodes
    if ep.enabled and ep.traffic_class == traffic_class:
        p_on = episode_on_fraction(ep)
        return base * (1.0 - p_on) + ep.rate_pps * p_on
    return base


# --------------------------------------------------------------------------- #
# İz üretimi (§5.6 ortak rastgele sayılar)
# --------------------------------------------------------------------------- #


def node_streams(seed: int, n_sources: int) -> list[tuple[np.random.Generator, np.random.Generator, np.random.Generator]]:
    """Düğüm başına bağımsız RNG akışları (§5.6): SeedSequence(seed).spawn(N); her çocuk (varış, boyut, epizot) üçlüsü."""
    streams = []
    for child in np.random.SeedSequence(seed).spawn(n_sources):
        arrivals, sizes, episodes = (np.random.default_rng(s) for s in child.spawn(3))
        streams.append((arrivals, sizes, episodes))
    return streams


def generate_source_packets(
    cls_cfg: TrafficClassConfig,
    traffic: TrafficConfig,
    traffic_class: str,
    max_delay_s: float,
    source: int,
    duration_s: float,
    streams: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
) -> tuple[list[Packet], list[Window]]:
    """Tek düğümün paketleri (id'siz, packet_id = −1) ve epizot pencereleri (§5.6)."""
    rng_arr, rng_size, rng_ep = streams
    base_rate = cls_cfg.rate_pps * traffic.load_multiplier
    ep = traffic.critical_episodes
    windows: list[Window] = []
    if cls_cfg.arrival.process == "poisson":
        if ep.enabled and ep.traffic_class == traffic_class:
            windows = episode_windows(rng_ep, ep, duration_s)
            times, flags = piecewise_poisson_arrivals(rng_arr, base_rate, windows, ep.rate_pps, duration_s)
        else:
            times = poisson_arrivals(rng_arr, base_rate, 0.0, duration_s)
            flags = [False] * len(times)
    elif cls_cfg.arrival.process == "periodic":
        a = cls_cfg.arrival
        times = periodic_arrivals(rng_arr, base_rate, a.jitter_frac, a.random_phase, duration_s)
        flags = [False] * len(times)
    else:  # config doğrulaması bunu engeller
        raise ValueError(f"bilinmeyen varış süreci {cls_cfg.arrival.process!r}")
    packets = [
        Packet(
            packet_id=-1,
            source=source,
            traffic_class=traffic_class,
            size_bytes=draw_packet_bytes(rng_size, cls_cfg.packet_bytes),
            created_s=t,
            deadline_s=t + max_delay_s,
            in_episode=flag,
        )
        for t, flag in zip(times, flags)
    ]
    return packets, windows


def generate_trace(cfg: Config, sources: Mapping[int, str], seed: int, duration_s: float) -> TrafficTrace:
    """Kaynak -> sınıf eşlemesinden iz üretir (§5.6). Yöntemden bağımsızdır; aynı (seed, kaynaklar, süre) aynı izi verir.

    RNG akışları `sources`'ın artan kimlik sırasına göre atanır (ağda sensör kimlikleri 0..N−1).
    """
    traffic = cfg.traffic
    ordered = sorted(sources)
    streams = node_streams(seed, len(ordered))
    all_packets: list[Packet] = []
    episodes: dict[int, tuple[Window, ...]] = {}
    sources_by_class: dict[str, list[int]] = {name: [] for name in traffic.classes}
    for src, st in zip(ordered, streams):
        cls = sources[src]
        cls_cfg = traffic.classes[cls]
        max_delay_s = cfg.policy.for_topology(cls_cfg.topology).max_delay_ms / 1000.0  # §5.7 π3
        packets, windows = generate_source_packets(cls_cfg, traffic, cls, max_delay_s, src, duration_s, st)
        all_packets += packets
        if windows:
            episodes[src] = tuple(windows)
        sources_by_class[cls].append(src)

    all_packets.sort(key=lambda p: (p.created_s, p.source))
    numbered = tuple(
        Packet(i, p.source, p.traffic_class, p.size_bytes, p.created_s, p.deadline_s, p.in_episode)
        for i, p in enumerate(all_packets)
    )
    by_source: dict[int, list[Packet]] = {src: [] for src in ordered}
    for p in numbered:
        by_source[p.source].append(p)
    return TrafficTrace(
        packets=numbered,
        by_source={src: tuple(v) for src, v in by_source.items()},
        sources_by_class={name: tuple(v) for name, v in sources_by_class.items()},
        episodes=episodes,
        duration_s=duration_s,
        seed=seed,
        load_multiplier=traffic.load_multiplier,
    )


def build_traffic_trace(cfg: Config, net: PhysicalNetwork, seed: int, duration_s: float | None = None) -> TrafficTrace:
    """Kanonik giriş noktası: ağın sensörleri (kimlik -> ürettiği sınıf, §5.3) için iz; süre varsayılanı simulation.duration_s."""
    sources = {i: net.nodes[i].traffic_class for i in net.sensor_ids}
    if any(c is None for c in sources.values()):
        raise ValueError("her sensörün bir trafik sınıfı olmalı (§5.3)")
    return generate_trace(cfg, sources, seed, cfg.simulation.duration_s if duration_s is None else duration_s)

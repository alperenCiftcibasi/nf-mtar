"""Metrikler: PDR, throughput, gecikme ve bileşenleri, kayıp nedenleri, airtime payı, enerji, FND/HND, QoS memnuniyeti,
root değişimi, karar süresi; runs.csv satırı ve opsiyonel paket logu (CLAUDE.md §5.7, §5.13, §5.19).

Uygulanan kararlar:
- §5.19 ısınma: metriklere yalnızca oluşturulma zamanı >= warm_up_s olan paketler girer (drain içinde teslim edilenler
  dahil). Ölçüm penceresi W = duration_s − warm_up_s (throughput ve airtime için).
- §5.13 PDR = teslim / üretilen (paket); normalize throughput = teslim edilen bayt / üretilen bayt (boyutlar sınıf içinde
  değiştiği için PDR'den ayrı raporlanır); mutlak throughput (kbit/s) = teslim edilen bit / W / 1000.
- §5.13 gecikme: ortalama, config `metrics.latency_percentiles` yüzdelikleri (varsayılan 50/90/95/99; medyan = p50),
  ms; bileşenler (kuyruk / geri çekilme / AIFS payı / iletim / işlem) teslim edilen paketler üzerinde ortalama, ms.
  Geri çekilme AIFS'yi içerir; AIFS ayrıca verilir (§5.5 "AIFS'nin bedeli").
- §5.13 kayıp nedenleri (§5.10a): sınıf başına sayı; `drop_total`; bypass edilen paket oranı (§5.4).
- §5.13 airtime payı: ısınma sonrası sınıf airtime'ı / toplam; kanal kullanımı = toplam airtime / W (mekânsal
  yeniden kullanım nedeniyle 1'i aşabilir).
- §5.13 enerji: Σ (başlangıç − kalan) sensörler üzerinde (toplam ve topoloji başına), ölü düğüm sayısı, FND (ilk ölüm)
  ve HND (⌈N/2⌉. ölüm) zamanı; ölüm yoksa NaN.
- §5.7 / §5.13 QoS memnuniyeti, sınıf başına: (i) süre sınırı oranı = π3 içinde teslim / üretilen; (ii) pencere oranı =
  π12 uzunluğundaki pencerelerden PDR >= π4/100 VE kayıp <= π5/100 olanların oranı (paket oluşturma zamanına göre
  pencereye atanır; paketsiz pencere sayılmaz). π4 ve π5 aynı büyüklüğün iki eşiği olduğundan bağlayıcı olan π5'tir.
- §5.13 root değişim sayısı (topoloji başına), karar turu (yol güncellemesi) başına ortalama çalışma süresi (ms).
- Φ (makale Denk. 6 düzeltilmiş) M6'da eklenecek.
- §5.19 runs.csv: bir koşu bir satır; sütunlar koşu tanımlayıcıları + ağ kaydı + yukarıdaki metrikler (sınıf soneki).
  Dosya varsa sütun birleşimiyle eklenir. run_id belirlenimcidir (yöntem, seed, yük, mod); `timestamp` ayrı sütundur.
- §5.19 paket logu (opsiyonel): bir satır bir paket; parquet (pyarrow varsa) yoksa csv.gz (yeni bağımlılık yok).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfmtar.config import PROJECT_ROOT
from nfmtar.network.graph import TOPOLOGY_ORDER
from nfmtar.simulation import DELIVERED, DROP_REASONS, PacketState, SimulationResult

COMPONENTS: tuple[str, ...] = ("queue", "backoff", "aifs", "transmission", "processing")


def is_runtime_column(name: str) -> bool:
    """Duvar saatine bağlı sütunlar (determinizm karşılaştırmasında dışlanır): zaman damgası, koşu süresi, karar süreleri."""
    return name in ("timestamp", "wall_clock_s") or name.startswith("decision_ms_")


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else math.nan


def _windows(result: SimulationResult) -> float:
    return result.duration_s - result.warm_up_s


# --------------------------------------------------------------------------- #
# Sınıf bazlı metrikler (§5.13)
# --------------------------------------------------------------------------- #


def class_metrics(result: SimulationResult, traffic_class: str) -> dict[str, float]:
    """Bir sınıfın ısınma sonrası metrikleri (§5.13; QoS memnuniyeti §5.7)."""
    cfg = result.cfg
    topology = result.class_topology[traffic_class]
    pol = cfg.policy.for_topology(topology)
    window_s = _windows(result)
    pkts = [ps for ps in result.packets_after_warm_up() if ps.packet.traffic_class == traffic_class]
    delivered = [ps for ps in pkts if ps.delivered]
    n_gen, n_del = len(pkts), len(delivered)
    bytes_gen = sum(ps.packet.size_bytes for ps in pkts)
    bytes_del = sum(ps.packet.size_bytes for ps in delivered)
    lat_ms = np.array([ps.latency_s for ps in delivered]) * 1000.0 if delivered else np.array([])
    m: dict[str, float] = {
        "generated": n_gen,
        "delivered": n_del,
        "pdr": n_del / n_gen if n_gen else math.nan,
        "norm_throughput": bytes_del / bytes_gen if bytes_gen else math.nan,
        "throughput_kbps": bytes_del * 8 / 1000.0 / window_s if window_s > 0 else math.nan,
        "lat_mean_ms": float(lat_ms.mean()) if n_del else math.nan,
    }
    for p in cfg.metrics["latency_percentiles"]:
        m[f"lat_p{int(p)}_ms"] = float(np.percentile(lat_ms, p)) if n_del else math.nan
    for comp in COMPONENTS:
        m[f"{comp}_ms"] = _mean([ps.components_s()[comp] * 1000.0 for ps in delivered])
    m["hops_mean"] = _mean([float(ps.hops) for ps in delivered])
    m["frame_attempts_mean"] = _mean([float(ps.frame_attempts) for ps in delivered])
    for reason in DROP_REASONS:
        m[f"drop_{reason}"] = sum(1 for ps in pkts if ps.status == reason)
    m["drop_total"] = n_gen - n_del
    m["bypass_frac"] = sum(1 for ps in pkts if ps.bypass) / n_gen if n_gen else math.nan
    # §5.7 QoS memnuniyeti
    m["deadline_ratio"] = sum(1 for ps in delivered if ps.deadline_met) / n_gen if n_gen else math.nan
    m["window_ratio"] = qos_window_ratio(pkts, result.warm_up_s, pol.qos_monitor_ms / 1000.0,
                                         pol.min_throughput_pct / 100.0, pol.loss_tolerance_pct / 100.0)
    # §5.13 airtime payı (ısınma sonrası)
    air = result.airtime_after_warm_up_s()
    total_air = sum(air.values())
    m["airtime_share"] = air.get(traffic_class, 0.0) / total_air if total_air > 0 else math.nan
    m["airtime_s"] = air.get(traffic_class, 0.0)
    return m


def qos_window_ratio(pkts: list[PacketState], t0: float, window_s: float, min_pdr: float, max_loss: float) -> float:
    """§5.7 pencere oranı: π12 pencerelerinden (oluşturma zamanına göre) PDR >= π4 ve kayıp <= π5 olanların oranı."""
    if not pkts or window_s <= 0:
        return math.nan
    gen: dict[int, int] = {}
    deliv: dict[int, int] = {}
    for ps in pkts:
        idx = int((ps.packet.created_s - t0) // window_s)
        gen[idx] = gen.get(idx, 0) + 1
        if ps.delivered:
            deliv[idx] = deliv.get(idx, 0) + 1
    ok = 0
    for idx, n in gen.items():
        pdr = deliv.get(idx, 0) / n
        if pdr >= min_pdr and (1.0 - pdr) <= max_loss:
            ok += 1
    return ok / len(gen)


# --------------------------------------------------------------------------- #
# Ağ geneli metrikler (§5.13)
# --------------------------------------------------------------------------- #


def network_metrics(result: SimulationResult) -> dict[str, float]:
    """Enerji, ömür, ölüm, root değişimi, karar süresi, kanal kullanımı, toplam PDR (§5.13)."""
    net, part = result.pn.net, result.pn.part
    consumed = result.energy_consumed_J()
    n = len(net.sensor_ids)
    deaths = sorted(t for t, _ in result.deaths)
    half = math.ceil(n / 2)
    pkts = result.packets_after_warm_up()
    n_del = sum(1 for ps in pkts if ps.delivered)
    window_s = _windows(result)
    air = result.airtime_after_warm_up_s()
    m: dict[str, float] = {
        "generated_total": len(pkts),
        "delivered_total": n_del,
        "pdr_total": n_del / len(pkts) if pkts else math.nan,
        "energy_total_J": sum(consumed.values()),
        "energy_tx_J": result.energy_tx_J,
        "energy_rx_J": result.energy_rx_J,
        "energy_rx_edge_J": result.energy_rx_edge_J,
        "dead_nodes": len(result.deaths),
        "fnd_s": deaths[0] if deaths else math.nan,
        "hnd_s": deaths[half - 1] if len(deaths) >= half else math.nan,
        "channel_utilization": sum(air.values()) / window_s if window_s > 0 else math.nan,
        "n_hops_completed": result.n_hops_completed,
        "hops_active_at_end": result.hops_active_at_end,
        "ttl_drops": result.ttl_drops,
        "root_changes_total": len(result.root_changes),
    }
    for t in TOPOLOGY_ORDER:
        m[f"energy_J_{t.value}"] = sum(consumed[v] for v in part.members[t])
        m[f"root_changes_{t.value}"] = sum(1 for rc in result.root_changes if rc["topology"] == t.value)
        m[f"route_updates_{t.value}"] = len(result.route_update_ms[t])
        m[f"decision_ms_{t.value}"] = _mean(result.route_update_ms[t])
        m[f"root_{t.value}"] = result.roots[t] if result.roots[t] is not None else -1
    return m


# --------------------------------------------------------------------------- #
# runs.csv ve tablolar (§5.19)
# --------------------------------------------------------------------------- #


def run_id(result: SimulationResult, experiment: str = "") -> str:
    """Belirlenimci koşu kimliği: deney-yöntem-seed-yük-mod."""
    tag = f"{experiment}-" if experiment else ""
    return f"{tag}{result.method}-s{result.seed}-l{result.trace.load_multiplier:g}-{result.relay_mode}"


def runs_row(result: SimulationResult, experiment: str = "") -> dict[str, Any]:
    """Bir koşunun runs.csv satırı: tanımlayıcılar, ağ/iz kaydı, ağ geneli metrikler, sınıf metrikleri (sonek _sınıf)."""
    row: dict[str, Any] = {
        "run_id": run_id(result, experiment),
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **result.to_record(),
        **{k: v for k, v in result.trace.to_record().items() if k != "load_multiplier"},
        **network_metrics(result),
    }
    for cls in result.trace.classes:
        for k, v in class_metrics(result, cls).items():
            row[f"{k}_{cls}"] = v
    return row


def per_class_table(result: SimulationResult) -> pd.DataFrame:
    """Konsol/rapor tablosu: satır = sınıf, sütun = seçilmiş metrikler (§5.13)."""
    rows = {cls: class_metrics(result, cls) for cls in result.trace.classes}
    df = pd.DataFrame(rows).T
    df.index.name = "class"
    return df


def append_runs_row(row: dict[str, Any], path: str | Path | None = None, cfg=None) -> Path:
    """Satırı runs.csv'ye ekler (§5.19); dosya varsa sütun birleşimiyle. Yol: verilen, config run.runs_table veya varsayılan."""
    if path is None:
        rel = cfg.run.runs_table if cfg is not None else "results/tables/runs.csv"
        path = PROJECT_ROOT / rel
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Paket logu (§5.19, opsiyonel)
# --------------------------------------------------------------------------- #


def packet_log_frame(result: SimulationResult) -> pd.DataFrame:
    """Bir satır bir paket: kimlik, sınıf, boyut, zamanlar, sonuç, atlama, gecikme bileşenleri (ms), kayıp düğümü."""
    recs = []
    for ps in result.packets:
        p = ps.packet
        comps = ps.components_s()
        recs.append({
            "packet_id": p.packet_id, "source": p.source, "topology": ps.topology.value, "traffic_class": p.traffic_class,
            "size_bytes": p.size_bytes, "created_s": p.created_s, "deadline_s": p.deadline_s, "in_episode": p.in_episode,
            "status": ps.status, "delivered": ps.delivered, "end_s": ps.end_s, "end_node": ps.end_node,
            "latency_ms": ps.latency_s * 1000.0, "deadline_met": ps.deadline_met, "hops": ps.hops, "bypass": ps.bypass,
            **{f"{k}_ms": v * 1000.0 for k, v in comps.items()},
            "frame_attempts": ps.frame_attempts, "csma_busy_events": ps.csma_busy_events,
            "after_warm_up": p.created_s >= result.warm_up_s,
        })
    return pd.DataFrame.from_records(recs)


def write_packet_log(result: SimulationResult, stem: str | Path, fmt: str = "parquet") -> Path:
    """Paket logunu yazar: parquet (pyarrow varsa) ya da csv.gz; dönen yol gerçek dosyadır."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df = packet_log_frame(result)
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
            out = stem.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return out
        except ImportError:
            pass
    out = stem.with_suffix(".csv.gz")
    df.to_csv(out, index=False, compression="gzip")
    return out


def rows_equal_ignoring_runtime(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """İki runs satırı çalışma süresi sütunları dışında aynı mı (determinizm testi, §3)."""
    ka = {k for k in a if not is_runtime_column(k)}
    kb = {k for k in b if not is_runtime_column(k)}
    if ka != kb:
        return False
    for k in ka:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True

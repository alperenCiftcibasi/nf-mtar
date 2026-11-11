"""MAC AIFS kalibrasyonu: π1 airtime hedefine en yakın sınıf bazlı AIFS çifti (CLAUDE.md §5.5, §5.7 π1).

Düzenek (§5.5 "Kalibrasyon"): sınıf başına bir doygun gönderici (kuyruğu hiç boşalmaz), üçü de birbirinin
taşıyıcı algılama menzilinde; 20 s, 5 seed. Paket boyutları ve yeniden iletim limitleri (π10) config'deki
trafik sınıflarından ve Tablo 1'den gelir. Düğümlerin enerjisi sınırsızdır: yalnızca MAC ölçülür.

Tarama: düzenli AIFS × kritik olmayan AIFS ızgarası (config experiments.mac_aifs_calibration.grid); kritik sınıfın
AIFS'si config'deki değerinde (0) kalır. Her çift için sınıf bazında ulaşılan airtime payı (seed ortalaması) ve
π1 hedefine (0,80 / 0,15 / 0,05) L1 uzaklığı = Σ_sınıf |pay − hedef| hesaplanır. L1'i en küçük çift seçilir
(eşitlikte daha küçük AIFS toplamı: daha az ek gecikme) ve configs/default.yaml -> radio.csma.aifs_slots'a yazılır.
80/15/5'in tam tutması beklenmez; kalan sapma tabloda ve çıktıda raporlanır.

İki referans satırı da ölçülür (§5.5'teki gerekçe sayıları):
- backoff_only: V-SDN politikası açık, AIFS hepsi 0  -> yalnızca (a) sınıf bazlı geri çekilme
- b1_plain:     V-SDN politikası yok (BE 3, AIFS 0)  -> öncelik yokken payın büyük paketli sınıfa kayması

Zamanlama kuralı (§5.5; sonuçlar buna duyarlıdır): doygun gönderici sıradaki pakete `RadioModel.next_packet_ready_s`
anında başlar = önceki paketin son MAC etkinliği + config `radio.processing_delay_s` (başarı/kayıp ayrımı yok).
Boşluksuz modelde yüksek AIFS'li sınıflar tamamen aç kalır. simulation.py'deki düğüm servis döngüsü aynı fonksiyonu
kullanır; `radio.processing_delay_s` değişirse bu kalibrasyon yeniden koşturulmalıdır.

Çıktı: results/tables/mac_aifs_calibration.csv (ızgara + referans satırları; seçilen çift `selected` sütununda).

Kullanım:
    python experiments/calibration/mac_aifs.py [--no-write] [--n-seeds 5] [--duration 20]
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.config import Config, load_config  # noqa: E402
from nfmtar.network.graph import NodeState  # noqa: E402
from nfmtar.network.radio import (  # noqa: E402
    BackoffPolicy,
    Channel,
    HopResult,
    RadioModel,
    run_mac_sessions,
    saturated_sender,
)

AIFS_LINE = re.compile(r"^([ \t]*aifs_slots:[ \t]*)\{[^}\r\n]*\}", re.MULTILINE)


def pi1_targets(cfg: Config) -> dict[str, float]:
    """Sınıf -> π1 hedef airtime payı (Tablo 1: E 0,80 / L 0,15 / R 0,05), sınıfın topolojisi üzerinden (§5.7)."""
    return {
        name: cfg.policy.for_topology(tc.topology).bandwidth_share_pct / 100.0
        for name, tc in cfg.traffic.classes.items()
    }


def run_saturated(cfg: Config, seed: int, duration_s: float, vsdn_policy: bool = True) -> dict[str, dict[str, float]]:
    """Tek koşu: sınıf başına bir doygun gönderici, hepsi cs menzilinde (§5.5). Sınıf -> ölçümler döndürür.

    Yerleşim: göndericiler x = 0'da, alıcıları link_distance_m ötede; çiftler pair_spacing_m aralıkla dizilir.
    Ölçümler: share = sınıfın airtime'ı / toplam airtime; airtime_frac = sınıfın airtime'ı / koşu süresi;
    packets = biten paket sayısı; drop_rate = düşen paket oranı (π10 aşıldı veya kanal erişim hatası).
    """
    cal = cfg.experiments["mac_aifs_calibration"]
    radio = RadioModel.from_config(cfg)
    backoff = BackoffPolicy(cfg.radio.csma, vsdn_policy=vsdn_policy)
    rng = np.random.default_rng(seed)
    classes = list(cfg.traffic.classes)

    senders: dict[str, NodeState] = {}
    receivers: dict[str, NodeState] = {}
    for i, name in enumerate(classes):
        y = i * float(cal["pair_spacing_m"])
        senders[name] = NodeState(2 * i, (0.0, y), name, False, float("inf"), float("inf"), 0.0)
        receivers[name] = NodeState(2 * i + 1, (float(cal["link_distance_m"]), y), name, False, float("inf"), float("inf"), 0.0)
    nodes = [*senders.values(), *receivers.values()]
    channel = Channel({n.node_id: n.pos for n in nodes}, cfg.radio.cs_range_m)
    for s in senders.values():  # §5.5: bütün göndericiler birbirinin algılama menzilinde olmalı
        others = {o.node_id for o in senders.values() if o is not s}
        if not others <= channel.cs_neighbors[s.node_id]:
            raise ValueError("kalibrasyon yerleşimi: göndericiler birbirinin cs menzilinde değil (pair_spacing_m)")

    def size_drawer(packet_bytes: int | tuple[int, int]):
        """§5.6 paket boyutu: sabit veya [alt, üst] aralığında düzgün."""
        if isinstance(packet_bytes, int):
            return lambda: packet_bytes
        lo, hi = packet_bytes
        return lambda: int(rng.integers(lo, hi + 1))

    logs: dict[str, list[HopResult]] = {name: [] for name in classes}
    sessions = {}
    for name in classes:
        tc = cfg.traffic.classes[name]
        sessions[name] = saturated_sender(
            radio, backoff, channel, rng, senders[name], receivers[name], name,
            cfg.policy.for_topology(tc.topology).retransmit_limit, size_drawer(tc.packet_bytes), logs[name],
        )
    run_mac_sessions(sessions, until_s=duration_s)

    share = channel.airtime_share()
    out: dict[str, dict[str, float]] = {}
    for name in classes:
        n_packets = len(logs[name])
        out[name] = {
            "share": share.get(name, 0.0),
            "airtime_frac": channel.airtime_s_by_class.get(name, 0.0) / duration_s,
            "packets": float(n_packets),
            "drop_rate": sum(not r.success for r in logs[name]) / n_packets if n_packets else float("nan"),
        }
    return out


def measure(cfg: Config, seeds: list[int], duration_s: float, vsdn_policy: bool = True) -> dict[str, float]:
    """Bir ayar için seed ortalamaları: sınıf payı (ort, std), π1'e L1 uzaklığı, kanal kullanımı, düşme oranı."""
    runs = [run_saturated(cfg, seed, duration_s, vsdn_policy) for seed in seeds]
    targets = pi1_targets(cfg)
    row: dict[str, float] = {}
    for name in cfg.traffic.classes:
        shares = np.array([r[name]["share"] for r in runs])
        row[f"share_{name}"] = float(shares.mean())
        row[f"share_std_{name}"] = float(shares.std(ddof=1)) if len(shares) > 1 else 0.0
    row["l1_distance"] = float(sum(abs(row[f"share_{name}"] - targets[name]) for name in targets))
    row["utilization"] = float(np.mean([sum(r[name]["airtime_frac"] for name in r) for r in runs]))
    for name in cfg.traffic.classes:  # bütün seed'lerdeki paketler üzerinden; hiç paket bitmediyse (tam açlık) NaN
        packets = sum(r[name]["packets"] for r in runs)
        dropped = sum(r[name]["packets"] * r[name]["drop_rate"] for r in runs if r[name]["packets"])
        row[f"packets_per_s_{name}"] = packets / (len(runs) * duration_s)
        row[f"drop_rate_{name}"] = dropped / packets if packets else float("nan")
    return row


def write_aifs_to_config(config_path: Path, aifs_slots: dict[str, int]) -> None:
    """radio.csma.aifs_slots satırını yerinde günceller; dosyanın geri kalanı (yorumlar, satır sonları) korunur."""
    with open(config_path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    value = "{" + ", ".join(f"{name}: {slots}" for name, slots in aifs_slots.items()) + "}"
    new_text, n = AIFS_LINE.subn(lambda m: m.group(1) + value, text)
    if n != 1:
        raise RuntimeError(f"{config_path}: tek bir 'aifs_slots: {{...}}' satırı bekleniyordu, {n} bulundu")
    with open(config_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)
    if load_config(config_path).radio.csma.aifs_slots != aifs_slots:
        raise RuntimeError(f"{config_path}: aifs_slots yazıldıktan sonra doğrulanamadı")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--n-seeds", type=int, default=None, help="varsayılan: experiments.mac_aifs_calibration.n_seeds")
    parser.add_argument("--duration", type=float, default=None, help="varsayılan: experiments.mac_aifs_calibration.duration_s")
    parser.add_argument("--no-write", action="store_true", help="seçilen çifti config'e yazma (yalnızca tablo)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    base = load_config(config_path)
    cal = base.experiments["mac_aifs_calibration"]
    if cal["objective"] != "l1_to_pi1":
        raise ValueError(f"desteklenmeyen kalibrasyon ölçütü {cal['objective']!r} (§5.5: l1_to_pi1)")
    seeds = list(base.run.seeds[: args.n_seeds or int(cal["n_seeds"])])
    duration_s = args.duration or float(cal["duration_s"])
    grid: dict[str, list[int]] = {name: [int(v) for v in values] for name, values in cal["grid"].items()}
    swept = list(grid)
    classes = list(base.traffic.classes)
    targets = pi1_targets(base)

    def aifs_for(pair: tuple[int, ...]) -> dict[str, int]:
        """Taranan sınıflara ızgara değeri, diğerlerine (kritik) config'deki değer."""
        return {name: (pair[swept.index(name)] if name in swept else base.radio.csma.aifs_slots[name]) for name in classes}

    rows: list[dict] = []
    # --- referans satırları (§5.5 gerekçe sayıları) ---
    zero = {name: 0 for name in classes}
    cfg_zero = load_config(config_path, overrides={"radio.csma.aifs_slots": zero})
    for scenario, vsdn_policy in (("backoff_only", True), ("b1_plain", False)):
        rows.append({"scenario": scenario, **{f"aifs_{n}": 0 for n in classes},
                     **measure(cfg_zero, seeds, duration_s, vsdn_policy)})
    # --- ızgara ---
    for pair in itertools.product(*(grid[name] for name in swept)):
        aifs = aifs_for(pair)
        cfg = load_config(config_path, overrides={"radio.csma.aifs_slots": aifs})
        rows.append({"scenario": "grid", **{f"aifs_{n}": aifs[n] for n in classes}, **measure(cfg, seeds, duration_s)})

    df = pd.DataFrame(rows)
    grid_df = df[df["scenario"] == "grid"]
    aifs_sum = grid_df[[f"aifs_{n}" for n in swept]].sum(axis=1)
    best_idx = grid_df.assign(aifs_sum=aifs_sum).sort_values(["l1_distance", "aifs_sum"]).index[0]
    df["selected"] = df.index == best_idx
    best = df.loc[best_idx]
    best_aifs = {name: int(best[f"aifs_{name}"]) for name in classes}

    out_table = PROJECT_ROOT / cal["table"]
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["scenario", *[f"aifs_{n}" for n in classes], *[f"share_{n}" for n in classes], "l1_distance",
            "utilization", *[f"drop_rate_{n}" for n in classes], "selected"]
    # not: konsol çıktısında "π" kullanılmaz (Windows cp1254 konsolu kodlayamıyor)
    print(f"doygun {len(classes)} gönderici, {duration_s:g} s, seed'ler {seeds}; hedef pi1 = "
          + " / ".join(f"{targets[n]:.2f}" for n in classes) + f" ({' / '.join(classes)})")
    print(df[show].round(3).to_string(index=False))
    print("\nseçilen aifs_slots: " + ", ".join(f"{n} = {best_aifs[n]}" for n in classes))
    print("ulaşılan pay:       " + " / ".join(f"{best[f'share_{n}']:.3f}" for n in classes)
          + f"   (L1 = {best['l1_distance']:.3f})")
    print("kalan sapma:        " + " / ".join(f"{best[f'share_{n}'] - targets[n]:+.3f}" for n in classes))
    print(f"-> {out_table}")
    if args.no_write:
        print("--no-write: config değiştirilmedi")
    else:
        write_aifs_to_config(config_path, best_aifs)
        print(f"-> {config_path}: radio.csma.aifs_slots = {best_aifs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Eğitim verisi üretimi (CLAUDE.md §5.16, §5.21): simülasyon, her K saniyede ağ durumu, kâhin etiketleri.

Her (seed, yük, yöntem) için bir koşu (varsayılan: E9-lite ölçeği, B1, yük 0,5, 300 s); her K = 10 s'de (simülasyonun anlık durumu: enerji, kuyruk
doluluğu, canlılık) her topoloji × aday için §5.14 girdileri (x1–x5; nominal trafik hızları, o anki kuyruk doluluğu) ve
§5.16 kâhin skoru (engine/oracle.py; §5.9 ağırlıklı maliyetli yönlendirici ile). Adaylar: §5.9 ağırlıklı yönlendirme
çizgesinde edge router'a yolu olan canlı sensörler.
Ayrım SEED BAZINDA (veri sızıntısı önlemi): varsayılan seed 1–4 eğitim, 5 doğrulama, 6 test.
--until-dead: koşu, hiçbir topolojide en az 2 aday kalmayınca (ömür sonu) ya da --duration'da biter.

Zengin veri (§5.21: x2 ve x3 değişsin diye düşük başlangıç enerjisi, uzun koşu, üç yük, B1 ve B3; seed 101–120,
14 eğitim / 3 doğrulama / 3 test):
  python experiments/build_training_data.py --seeds 101 102 ... 120 --val-seeds 115 116 117 --test-seeds 118 119 120
      --loads 0.25 0.5 1 --methods B1 B3 --initial-energy 2 --duration 1800 --until-dead --out data/e9_rich.npz

Çıktı: data/e9_lite.npz (varsayılan) — X (n, 5), y (n,), seed (n,), group (n; aynı durum × topoloji = bir grup),
node (n,), topology (n; 0/1/2 = E/L/R), load (n,), method (n; --methods sırasındaki indeks), time_s (n,),
components (n, 3; kâhinin ağırlıklı bileşenleri teslim / gecikme / dayanıklılık, toplamları ham skor; §5.21 teşhisi),
queue_occupancy (n; anlık kuyruk doluluğu, tanılama; x2 18 Kasım'dan beri radyo meşguliyeti, §5.14),
components_raw (n, 3; kâhinin ham bileşenleri: teslim olasılığı, gecikme s, dayanıklılık s — normalizasyonsuz;
§5.22 sabit ölçekli kâhin duyarlılığı yeniden simülasyonsuz bunlardan hesaplanır),
split (n; 0 eğitim, 1 doğrulama, 2 test); ayrıca methods (yöntem adları).

Kullanım: python experiments/build_training_data.py [--seeds 1 2 3 4 5 6] [--loads 0.5] [--methods B1]
          [--duration 300] [--interval 10] [--initial-energy J] [--until-dead] [--workers N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

OUT = PROJECT_ROOT / "data" / "e9_lite.npz"


def collect(task: tuple) -> dict[str, np.ndarray]:
    from nfmtar.config import load_config
    from nfmtar.engine.fuzzy import FuzzySpec, extract_features, nominal_rates_pps
    from nfmtar.engine.oracle import oracle_components
    from nfmtar.network.graph import TOPOLOGY_ORDER
    from nfmtar.network.partition import build_partitioned_network
    from nfmtar.router import Router
    from nfmtar.simulation import Simulation
    from nfmtar.traffic import build_traffic_trace

    seed, load, load_idx, method, method_idx, duration, interval, energy, until_dead = task
    overrides = {"traffic.load_multiplier": load, "simulation.duration_s": duration}
    if energy is not None:
        overrides["energy.initial_J"] = energy
    cfg = load_config(overrides=overrides)
    pn = build_partitioned_network(cfg, seed)
    trace = build_traffic_trace(cfg, pn.net, seed)
    sim = Simulation(cfg, pn, trace, method)
    router = Router.for_method(cfg, sim.radio, "B3")       # §5.9 ağırlıklı maliyet (kâhin ve H/aday kümesi)
    spec = FuzzySpec.from_config(cfg)
    rates = nominal_rates_pps(cfg, pn.net)
    rows: list[tuple] = []
    snap = [0]

    def hook(s: Simulation) -> bool:
        net, part = s.net, s.part
        occ = {v: net.nodes[v].queue.occupancy for v in net.sensor_ids}        # tanılama (eski x2)
        util = s.busy.busy_ratios(net.sensor_ids, s.now)                       # §5.14 x2 radyo meşguliyeti
        alive_topologies = 0
        for ti, t in enumerate(TOPOLOGY_ORDER):
            reach = router.reach_from_edge(net, part, t)
            if len(reach.candidates) < 2:
                continue
            alive_topologies += 1
            feats = extract_features(spec, net, part, t, rates, s.radio.prr, util, reach.hops)
            oc = oracle_components(cfg, router, net, part, t, reach, rates)
            idx = {v: i for i, v in enumerate(oc["candidates"])}
            gid = ((seed * 10 + load_idx) * 10 + method_idx) * 100000 + snap[0] * 10 + ti
            for v in sorted(reach.candidates):
                i = idx[v]
                rows.append((*feats[v].as_array(), oc["score"][i], seed, gid, v, ti, load, method_idx, s.now,
                             oc["w_delivery"][i], oc["w_latency"][i], oc["w_resilience"][i], occ[v],
                             oc["delivery"][i], oc["latency_s"][i], oc["resilience"][i]))
        snap[0] += 1
        return not (until_dead and alive_topologies == 0)          # False -> simülasyon durur (ömür sonu)

    sim.snapshot_interval_s = interval
    sim.snapshot_hook = hook
    sim.run()
    a = np.array(rows, dtype=float).reshape(-1, 20)
    print(f"seed {seed} load {load} {method}: {snap[0]} snapshots, {len(a)} rows, ended t={sim.now:.0f} s, "
          f"{len(sim.deaths)} deaths", flush=True)
    return {"X": a[:, :5], "y": a[:, 5], "seed": a[:, 6].astype(int), "group": a[:, 7].astype(np.int64),
            "node": a[:, 8].astype(int), "topology": a[:, 9].astype(int), "load": a[:, 10],
            "method": a[:, 11].astype(int), "time_s": a[:, 12], "components": a[:, 13:16],
            "queue_occupancy": a[:, 16], "components_raw": a[:, 17:20]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[5])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[6])
    parser.add_argument("--loads", nargs="+", type=float, default=[0.5])
    parser.add_argument("--methods", nargs="+", default=["B1"])
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--initial-energy", type=float, default=None, help="J; verilmezse config (energy.initial_J)")
    parser.add_argument("--until-dead", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    assert not set(args.val_seeds) & set(args.test_seeds), "doğrulama ve test seed'leri ayrık olmalı"
    tasks = [(s, load, li, m, mi, args.duration, args.interval, args.initial_energy, args.until_dead)
             for mi, m in enumerate(args.methods) for li, load in enumerate(args.loads) for s in args.seeds]
    tasks.sort(key=lambda t: (-t[1], t[3] != "B3"))           # uzun koşular (yüksek yük, B3) önce
    with ProcessPoolExecutor(max_workers=args.workers or min(len(tasks), 8)) as pool:
        parts = list(pool.map(collect, tasks))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["methods"] = np.array(args.methods)
    split = np.zeros(len(data["y"]), dtype=int)
    split[np.isin(data["seed"], args.val_seeds)] = 1
    split[np.isin(data["seed"], args.test_seeds)] = 2
    data["split"] = split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"{len(data['y'])} rows, {len(np.unique(data['group']))} groups; "
          f"train/val/test rows = {[(split == k).sum() for k in (0, 1, 2)]}; x means {data['X'].mean(0).round(3)}; "
          f"y mean {data['y'].mean():.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

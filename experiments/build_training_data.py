"""Eğitim verisi üretimi (CLAUDE.md §5.16; E9-lite ölçeği): simülasyon, her K saniyede ağ durumu, kâhin etiketleri.

Her seed için B1 ile koşu (varsayılan yük 0,5, 300 s); her K = 10 s'de (simülasyonun anlık durumu: enerji, kuyruk
doluluğu, canlılık) her topoloji × aday için §5.14 girdileri (x1–x5; nominal trafik hızları, o anki kuyruk doluluğu) ve
§5.16 kâhin skoru (engine/oracle.py; §5.9 ağırlıklı maliyetli yönlendirici ile). Adaylar: §5.9 ağırlıklı yönlendirme
çizgesinde edge router'a yolu olan canlı sensörler.
Ayrım SEED BAZINDA (veri sızıntısı önlemi): varsayılan seed 1–4 eğitim, 5 doğrulama, 6 test.

Çıktı: data/e9_lite.npz — X (n, 5), y (n,), seed (n,), group (n; aynı durum × topoloji = bir grup), node (n,),
topology (n; 0/1/2 = E/L/R), split (n; 0 eğitim, 1 doğrulama, 2 test).

Kullanım: python experiments/build_training_data.py [--seeds 1 2 3 4 5 6] [--load 0.5] [--duration 300] [--interval 10]
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


def collect(task: tuple[int, float, float, float]) -> dict[str, np.ndarray]:
    from nfmtar.config import load_config
    from nfmtar.engine.fuzzy import FuzzySpec, extract_features, nominal_rates_pps
    from nfmtar.engine.oracle import oracle_scores
    from nfmtar.network.graph import TOPOLOGY_ORDER
    from nfmtar.network.partition import build_partitioned_network
    from nfmtar.router import Router
    from nfmtar.simulation import Simulation
    from nfmtar.traffic import build_traffic_trace

    seed, load, duration, interval = task
    cfg = load_config(overrides={"traffic.load_multiplier": load, "simulation.duration_s": duration})
    pn = build_partitioned_network(cfg, seed)
    trace = build_traffic_trace(cfg, pn.net, seed)
    sim = Simulation(cfg, pn, trace, "B1")
    router = Router.for_method(cfg, sim.radio, "B3")       # §5.9 ağırlıklı maliyet (kâhin ve H/aday kümesi)
    spec = FuzzySpec.from_config(cfg)
    rates = nominal_rates_pps(cfg, pn.net)
    rows: list[tuple] = []
    snap = [0]

    def hook(s: Simulation) -> None:
        net, part = s.net, s.part
        occ = {v: net.nodes[v].queue.occupancy for v in net.sensor_ids}
        for ti, t in enumerate(TOPOLOGY_ORDER):
            reach = router.reach_from_edge(net, part, t)
            if len(reach.candidates) < 2:
                continue
            feats = extract_features(spec, net, part, t, rates, s.radio.prr, occ, reach.hops)
            scores = oracle_scores(cfg, router, net, part, t, reach, rates)
            gid = seed * 100000 + snap[0] * 10 + ti
            for v in sorted(reach.candidates):
                rows.append((*feats[v].as_array(), scores[v], seed, gid, v, ti))
        snap[0] += 1

    sim.snapshot_interval_s = interval
    sim.snapshot_hook = hook
    sim.run()
    a = np.array(rows, dtype=float)
    return {"X": a[:, :5], "y": a[:, 5], "seed": a[:, 6].astype(int), "group": a[:, 7].astype(int),
            "node": a[:, 8].astype(int), "topology": a[:, 9].astype(int)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[5])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[6])
    parser.add_argument("--load", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        parts = list(pool.map(collect, [(s, args.load, args.duration, args.interval) for s in args.seeds]))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
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

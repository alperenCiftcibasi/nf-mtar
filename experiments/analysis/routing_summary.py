"""M4 yol seçimi özeti (CLAUDE.md §5.9 doğrulama): bir seed'in ağında yöntem × topoloji × relay_mode yol istatistikleri.

Root sabittir (§5.12 static_root_rule = nearest_to_edge: adaylar arasında edge router'a en yakın düğüm) ki yöntemler
arasında yalnızca yol maliyeti değişsin. Her satır için (bütün üyelerin iki aşamalı tam yolları üzerinden):
- hops_mean / hops_median / hops_max: sensör başına atlama sayısı
- hop_len_mean_m: ortalama atlama uzunluğu (m)
- e2e_prr_mean: uçtan uca Π PRR ortalaması (iki aşama dahil; §5.9 kırpılmış PRR)
- bypass_frac, n_no_route (§5.4); foreign_relays: başka topolojiden aktarıcı sayısı (shared)
- subgraph_mean_degree: topoloji alt çizgesinin ortalama derecesi (yoğunluk)
- reach_ms / plan_ms: reach_from_edge ve plan_routes çalışma süreleri (ms)

Çıktı: results/tables/routing_summary_seed<seed><tag>.csv ve ekrana tablo.

Kullanım:
    python experiments/analysis/routing_summary.py [--seed 1] [--methods B1 B2 NF-MTAR] [--modes strict shared]
                                                   [--set router.lambda2=0 ...] [--tag _lambda2_0]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfmtar.config import load_config, parse_override  # noqa: E402
from nfmtar.network.graph import TOPOLOGY_ORDER  # noqa: E402
from nfmtar.network.partition import build_partitioned_network  # noqa: E402
from nfmtar.network.radio import RadioModel  # noqa: E402
from nfmtar.router import Router, relays_on_paths, static_root  # noqa: E402

COLUMNS = ["mode", "method", "topology", "n_members", "root", "root_H", "root_C", "hops_mean", "hops_median", "hops_max",
           "hop_len_mean_m", "e2e_prr_mean", "bypass_frac", "n_no_route", "foreign_relays", "subgraph_mean_degree",
           "reach_ms", "plan_ms"]


def summarize(cfg, pn, method: str, mode: str) -> list[dict]:
    """Bir yöntem × relay_mode için üç topolojinin satırları."""
    net, part = pn.net, pn.part
    router = Router.for_method(cfg, RadioModel.from_config(cfg), method, relay_mode=mode)
    rows = []
    for t in TOPOLOGY_ORDER:
        t0 = time.perf_counter()
        reach = router.reach_from_edge(net, part, t)
        t1 = time.perf_counter()
        root = static_root(net, reach)
        plan = router.plan_routes(net, part, t, root, reach)
        t2 = time.perf_counter()
        members = sorted(part.members[t])
        paths = [plan.full_path(v) for v in members]
        paths = [p for p in paths if p is not None]
        hops = [len(p) - 1 for p in paths]
        hop_lens = [net.distance(a, b) for p in paths for a, b in zip(p, p[1:])]
        prrs = [router.path_prr(net, p) for p in paths]
        foreign = {v for v in relays_on_paths(paths) if net.nodes[v].topology != t}
        rows.append({
            "mode": mode, "method": method, "topology": t.value, "n_members": len(members),
            "root": root, "root_H": reach.hops[root], "root_C": reach.cost[root],
            "hops_mean": statistics.fmean(hops), "hops_median": statistics.median(hops), "hops_max": max(hops),
            "hop_len_mean_m": statistics.fmean(hop_lens), "e2e_prr_mean": statistics.fmean(prrs),
            "bypass_frac": len(plan.bypass) / len(members), "n_no_route": len(plan.no_route),
            "foreign_relays": len(foreign), "subgraph_mean_degree": part.subgraph_mean_degree(net, t),
            "reach_ms": (t1 - t0) * 1000, "plan_ms": (t2 - t1) * 1000,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["B1", "B2", "NF-MTAR"])
    parser.add_argument("--modes", nargs="+", default=["strict", "shared"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (ör. router.lambda2=0)")
    parser.add_argument("--tag", default="", help="çıktı dosya adı eki")
    args = parser.parse_args(argv)

    overrides = dict(parse_override(s) for s in args.set)
    cfg = load_config(args.config, overrides=overrides)
    pn = build_partitioned_network(cfg, args.seed)
    rows = [r for mode in args.modes for method in args.methods for r in summarize(cfg, pn, method, mode)]
    df = pd.DataFrame(rows, columns=COLUMNS)

    out = PROJECT_ROOT / cfg.run.results_dir / "tables" / f"routing_summary_seed{args.seed}{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"seed {args.seed}, N = {pn.net.n_sensors}, overrides = {overrides or '-'}")
    print(df.round(3).to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M4 yol istatistikleri tanılaması (CLAUDE.md §5.4, §5.9, §5.12, §5.17): bir seed'in ağında TEK karar turu,
yöntem × topoloji × relay_mode.

Root sabittir (§5.12 sabit root kuralı: adaylar arasında edge router'a Öklid mesafesi en küçük düğüm) ki yöntemler
arasında yalnızca yol maliyeti değişsin. Her satır, topolojinin bütün üyelerinin iki aşamalı tam yolları
(sensör -> root -> edge router; bypass'ta sensör -> edge router, §5.4) üzerinden:
- hops_mean / hops_median / hops_max: sensör başına atlama sayısı
- hop_len_mean_m: ortalama atlama uzunluğu (m)
- e2e_prr_mean: uçtan uca PRR çarpımının ortalaması (iki aşama dahil; §5.9 PRR_min'e kırpılmış PRR)
- bypass_frac: root'u atlayan üye oranı; n_no_route: ne root'a ne edge router'a yolu olan üye sayısı (§5.4)
- foreign_relays: başka topolojiden aktarıcı olarak kullanılan düğüm sayısı (yalnızca shared modda > 0, §5.9)
- subgraph_mean_degree: topoloji alt çizgesinin ortalama derecesi (yoğunluk; atlama sayısı farklarını yorumlamak için)
- root_H / root_C: root'un edge router'a en az atlaması (BFS) ve en ucuz yol maliyeti (§5.17)
- reach_ms / plan_ms: reach_from_edge ve plan_routes çalışma süreleri (ms)

Çıktı: results/tables/route_stats_seed<seed><tag>.csv (veya --out) ve ekrana tablo.

Önce/sonra karşılaştırması: script, M4'ün ilk sürümüyle (commit 8650d57: d²/R² enerji vekili, normalize ETX,
`RoutePlan.unreachable`, `static_root` yok) de çalışır. Eski kod bir git worktree'ye alınıp
    PYTHONPATH=<worktree>/src python experiments/diagnostics/route_stats.py --config <worktree>/configs/default.yaml --out ...
ile koşturulur; eski API'de adı farklı olan alanlar aşağıdaki uyumluluk yardımcılarıyla doldurulur
(unreachable -> n_no_route: aynı nicelik, yalnızca ad değişti), olmayanlar boş kalır. `code_rev` sütunu, içe aktarılan
nfmtar paketinin git commit'ini taşır ki hangi kodla ölçüldüğü CSV'den okunabilsin.

Kullanım:
    python experiments/diagnostics/route_stats.py [--seed 1] [--methods B1 B2 NF-MTAR] [--modes strict shared]
                                                  [--set router.lambda2=0 ...] [--tag _lambda2_0] [--out dosya.csv]
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import nfmtar  # noqa: F401

import pandas as pd  # noqa: E402

from nfmtar.config import load_config, parse_override  # noqa: E402
from nfmtar.network.graph import TOPOLOGY_ORDER  # noqa: E402
from nfmtar.network.partition import build_partitioned_network  # noqa: E402
from nfmtar.network.radio import RadioModel  # noqa: E402
from nfmtar.router import Router, relays_on_paths  # noqa: E402

COLUMNS = [
    "code_rev", "mode", "method", "topology", "n_members", "root", "root_H", "root_C",
    "hops_mean", "hops_median", "hops_max", "hop_len_mean_m", "e2e_prr_mean",
    "bypass_frac", "n_no_route", "foreign_relays", "subgraph_mean_degree", "reach_ms", "plan_ms",
]


# ----------------------------------------------------------------------------- eski API uyumluluğu (önce/sonra)


def _static_root(net, reach) -> int:
    """§5.12 sabit root kuralı; yeni kodda router.static_root, eski kodda (8650d57) aynı kural yerel olarak."""
    try:
        from nfmtar.router import static_root
    except ImportError:
        if not reach.candidates:
            raise ValueError(f"topoloji {reach.topology.value}: aday yok, sabit root secilemez")
        return min(reach.candidates, key=lambda v: (net.distance(v, reach.edge_router), v))
    return static_root(net, reach)


def _no_route(plan) -> frozenset[int] | None:
    """RoutePlan.no_route (yeni ad) ya da RoutePlan.unreachable (eski ad); ikisi de yoksa None (CSV'de boş)."""
    for name in ("no_route", "unreachable"):
        if hasattr(plan, name):
            return getattr(plan, name)
    return None


def _code_rev() -> str:
    """İçe aktarılan nfmtar paketinin git commit'i (kısa); bulunamazsa 'unknown'."""
    try:
        repo = Path(nfmtar.__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — tanılama bilgisi, hata yüzünden ölçüm düşmesin
        return "unknown"


# ----------------------------------------------------------------------------- ölçüm


def route_stats(cfg, pn, method: str, mode: str, code_rev: str) -> list[dict]:
    """Bir yöntem × relay_mode için üç topolojinin satırları (tek karar turu, sabit root)."""
    net, part = pn.net, pn.part
    router = Router.for_method(cfg, RadioModel.from_config(cfg), method, relay_mode=mode)
    rows = []
    for t in TOPOLOGY_ORDER:
        t0 = time.perf_counter()
        reach = router.reach_from_edge(net, part, t)
        t1 = time.perf_counter()
        root = _static_root(net, reach)
        plan = router.plan_routes(net, part, t, root, reach)
        t2 = time.perf_counter()

        members = sorted(part.members[t])
        paths = [p for p in (plan.full_path(v) for v in members) if p is not None]
        hops = [len(p) - 1 for p in paths]
        hop_lens = [net.distance(a, b) for p in paths for a, b in zip(p, p[1:])]
        e2e_prr = [router.path_prr(net, p) for p in paths]
        foreign = {v for v in relays_on_paths(paths) if net.nodes[v].topology != t}
        no_route = _no_route(plan)
        rows.append({
            "code_rev": code_rev, "mode": mode, "method": method, "topology": t.value, "n_members": len(members),
            "root": root, "root_H": reach.hops[root], "root_C": reach.cost[root],
            "hops_mean": statistics.fmean(hops), "hops_median": statistics.median(hops), "hops_max": max(hops),
            "hop_len_mean_m": statistics.fmean(hop_lens), "e2e_prr_mean": statistics.fmean(e2e_prr),
            "bypass_frac": len(plan.bypass) / len(members),
            "n_no_route": None if no_route is None else len(no_route),
            "foreign_relays": len(foreign),
            "subgraph_mean_degree": part.subgraph_mean_degree(net, t),
            "reach_ms": (t1 - t0) * 1000.0, "plan_ms": (t2 - t1) * 1000.0,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["B1", "B2", "NF-MTAR"],
                        help="engine.methods anahtarlari (B1 hop, B2 etx_depletion, NF-MTAR/B3 weighted)")
    parser.add_argument("--modes", nargs="+", default=["strict", "shared"], help="relay_mode listesi")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override (or. router.lambda2=0)")
    parser.add_argument("--tag", default="", help="cikti dosya adi eki (or. _lambda2_0)")
    parser.add_argument("--out", default=None, help="cikti CSV yolu (varsayilan results/tables/route_stats_seed<seed><tag>.csv)")
    args = parser.parse_args(argv)

    overrides = dict(parse_override(s) for s in args.set)
    cfg = load_config(args.config, overrides=overrides)
    pn = build_partitioned_network(cfg, args.seed)
    code_rev = _code_rev()
    rows = [r for mode in args.modes for method in args.methods for r in route_stats(cfg, pn, method, mode, code_rev)]
    df = pd.DataFrame(rows, columns=COLUMNS)

    out = Path(args.out) if args.out else PROJECT_ROOT / cfg.run.results_dir / "tables" / f"route_stats_seed{args.seed}{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"nfmtar: {Path(nfmtar.__file__).resolve().parent} (git {code_rev}); config: {args.config}")
    print(f"seed {args.seed}, N = {pn.net.n_sensors}, regen_attempts = {pn.net.regen_attempts}, overrides = {overrides or '-'}")
    print(df.drop(columns=["code_rev"]).round(3).to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

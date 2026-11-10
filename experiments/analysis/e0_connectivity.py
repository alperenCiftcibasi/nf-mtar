"""E0 — Bağlantısallık analizi: ayrık bölümlemenin bedeli (CLAUDE.md §5.19 E0; §5.1, §5.2, §5.3).

r_placement ∈ {uniform, region} × seed'ler, `enforce_partition_connectivity: false` (yeniden üretim yok):
- topoloji başına kopuk sensör sayısı (ort / min / max),
- kopukların fiziksel ağ üzerinden edge router'a yolu olanlarının oranı,
- alt çizge ortalama derecesi,
- ayrıca §5.1 ölçütü ZORUNLU tutulduğunda gereken yeniden üretim sayısı (ort / max) ve başarı oranı.

Çıktılar (config experiments.E0):
- results/tables/e0_connectivity.csv            (uzun biçim: yerleşim × seed × topoloji)
- results/tables/e0_connectivity_summary.csv    (özet)
- results/figures/e0_topology_uniform_vs_region.png (yan yana, aynı seed, yeniden üretimsiz)

Kullanım:
    python experiments/analysis/e0_connectivity.py [--n-seeds 30] [--figure-seed 1] [--skip-regen]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from nfmtar.config import Config, load_config  # noqa: E402
from nfmtar.network.graph import TOPOLOGY_ORDER  # noqa: E402
from nfmtar.network.partition import PartitionedNetwork, build_partitioned_network  # noqa: E402
from plot_topology import SURFACE, draw_topology, legend_handles  # noqa: E402

log = logging.getLogger("e0")


def measure_rows(pn: PartitionedNetwork, placement: str) -> list[dict]:
    """Bir ağ için topoloji başına E0 ölçümleri (§5.19 E0): kopuk, fiziksel yolu olan kopuk, ortalama derece."""
    net, part = pn.net, pn.part
    physical_connected = net.is_connected()
    rows = []
    for t in TOPOLOGY_ORDER:
        eid = net.edge_router_ids[t]
        disc = part.disconnected[t]
        with_phys = sum(1 for v in disc if nx.has_path(net.graph, v, eid))
        rows.append({
            "placement": placement,
            "seed": net.seed,
            "topology": t.value,
            "n_members": len(part.members[t]),
            "disconnected": len(disc),
            "disconnected_frac": len(disc) / len(part.members[t]) if part.members[t] else 0.0,
            "disconnected_with_physical_path": with_phys,
            "subgraph_mean_degree": part.subgraph_mean_degree(net, t),
            "physical_connected": physical_connected,
            "physical_components": net.component_count(),
        })
    return rows


def regen_attempts_when_enforced(cfg_enforced: Config, seed: int) -> tuple[int, bool]:
    """§5.1 ölçütü zorunluyken gereken yeniden üretim sayısı ve başarı (max_regen_attempts içinde)."""
    try:
        pn = build_partitioned_network(cfg_enforced, seed)
        return pn.net.regen_attempts, True
    except RuntimeError:
        return cfg_enforced.network.max_regen_attempts, False


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Yerleşim × topoloji özeti: kopuk ort/min/max, fiziksel yol oranı, derece, yeniden üretim (§5.19 E0)."""
    g = df.groupby(["placement", "topology"], sort=False)
    out = g.agg(
        n_seeds=("seed", "nunique"),
        n_members=("n_members", "first"),
        disconnected_mean=("disconnected", "mean"),
        disconnected_min=("disconnected", "min"),
        disconnected_max=("disconnected", "max"),
        disconnected_frac_mean=("disconnected_frac", "mean"),
        disconnected_total=("disconnected", "sum"),
        disconnected_with_physical_path_total=("disconnected_with_physical_path", "sum"),
        subgraph_mean_degree=("subgraph_mean_degree", "mean"),
        physical_connected_rate=("physical_connected", "mean"),
    ).reset_index()
    out["physical_path_share"] = (
        out["disconnected_with_physical_path_total"] / out["disconnected_total"].where(out["disconnected_total"] > 0)
    )
    if "regen_attempts" in df.columns:
        per_seed = df.drop_duplicates(["placement", "seed"])
        r = per_seed.groupby("placement", sort=False).agg(
            regen_attempts_mean=("regen_attempts", "mean"),
            regen_attempts_max=("regen_attempts", "max"),
            regen_success_rate=("regen_success", "mean"),
        ).reset_index()
        out = out.merge(r, on="placement", how="left")
    return out


def side_by_side_figure(cfgs: dict[str, Config], seed: int, out_path: Path) -> Path:
    """Aynı seed için uniform ve region yerleşimlerini yan yana çizer (yeniden üretimsiz)."""
    fig, axes = plt.subplots(1, len(cfgs), figsize=(8.0 * len(cfgs), 8.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (placement, cfg) in zip(axes, cfgs.items()):
        pn = build_partitioned_network(cfg, seed)
        draw_topology(ax, pn.net, pn.part, cfg, legend=False)
    fig.suptitle("E0 — ayrık bölümlemede R topolojisinin bağlantısallığı: uniform vs region yerleşimi (§5.2)",
                 fontsize=11, x=0.01, y=0.995, ha="left")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--n-seeds", type=int, default=None, help="varsayılan: experiments.E0.n_seeds")
    parser.add_argument("--figure-seed", type=int, default=None, help="varsayılan: config'deki ilk seed")
    parser.add_argument("--skip-regen", action="store_true", help="zorunlu yeniden üretim ölçümünü atla")
    parser.add_argument("--verbose", action="store_true", help="kopuk düğüm uyarılarını göster")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.getLogger("nfmtar.network").setLevel(logging.ERROR)

    base = load_config(args.config)
    e0 = base.experiments["E0"]
    placements: list[str] = list(e0["placements"])
    n_seeds = args.n_seeds or int(e0["n_seeds"])
    seeds = base.run.seeds[:n_seeds]
    out_table = PROJECT_ROOT / e0["table"]
    out_summary = PROJECT_ROOT / e0["summary_table"]
    out_figure = PROJECT_ROOT / e0["figure"]

    cfg_measure = {
        p: load_config(args.config, overrides={
            "placement.random.r_placement": p,
            "network.enforce_partition_connectivity": bool(e0["enforce_partition_connectivity"]),
        })
        for p in placements
    }
    cfg_enforced = {p: load_config(args.config, overrides={"placement.random.r_placement": p}) for p in placements}

    rows: list[dict] = []
    for p in placements:
        log.info("yerleşim %s: %d seed ölçülüyor", p, len(seeds))
        for seed in seeds:
            pn = build_partitioned_network(cfg_measure[p], seed)
            seed_rows = measure_rows(pn, p)
            if not args.skip_regen:
                attempts, ok = regen_attempts_when_enforced(cfg_enforced[p], seed)
                for r in seed_rows:
                    r["regen_attempts"] = attempts
                    r["regen_success"] = ok
            rows.extend(seed_rows)

    df = pd.DataFrame(rows)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)
    summary = summarize(df)
    summary.to_csv(out_summary, index=False)

    fig_seed = args.figure_seed if args.figure_seed is not None else base.run.seeds[0]
    fig_path = side_by_side_figure(cfg_measure, fig_seed, out_figure)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.round(3).to_string(index=False))
    print(f"-> {out_table}\n-> {out_summary}\n-> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

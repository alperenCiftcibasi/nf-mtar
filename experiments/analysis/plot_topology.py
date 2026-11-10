"""Fiziksel ağ ve V-SDN bölümlemesini results/figures/topology.png olarak çizer (CLAUDE.md §5.1–5.4).

Çizilenler:
- unit disk kenarları (§5.1), sensörler topolojiye göre renkli (§5.2 / §5.3),
- üç sabit edge router kare işaretle (§5.4),
- eliptik bölge, koridor şeridi ve (region modunda) R dikdörtgeninin sınırları kesikli çizgiyle (§5.2),
- edge router'a yolu olmayan sensörler "x" ile (§5.3 bağlantısallık kontrolü).

`draw_topology(ax, ...)` başka scriptlerce (E0 yan yana görseli) yeniden kullanılır.

Kullanım:
    python experiments/analysis/plot_topology.py [--seed 1] [--config configs/default.yaml]
        [--out results/figures/topology.png] [--set placement.random.r_placement=uniform ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:  # `pip install -e .` yapılmadıysa src/ düzenini elle ekle
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

from nfmtar.config import Config, load_config, parse_override  # noqa: E402
from nfmtar.network.graph import TOPOLOGY_ORDER, PhysicalNetwork, Topology  # noqa: E402
from nfmtar.network.partition import Partition, build_partitioned_network  # noqa: E402

# Kategorik palet (dataviz referansı, ilk üç slot; üçü birlikte renk körlüğü testini geçer)
TOPOLOGY_COLORS = {Topology.E: "#2a78d6", Topology.L: "#eb6834", Topology.R: "#1baf7a"}
TOPOLOGY_LABELS = {
    Topology.E: "E — eliptik (kritik)",
    Topology.L: "L — doğrusal (düzenli)",
    Topology.R: "R — rastgele (kritik olmayan)",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
EDGE_GRAY = "#c3c2b7"
SURFACE = "#fcfcfb"


def _region_patches(cfg: Config, net: PhysicalNetwork) -> list:
    """§5.2 bölge sınırları: elips (E), koridor dikdörtgeni (L) ve region modunda R dikdörtgeni."""
    w, h = cfg.network.area_m
    ell = cfg.placement.elliptic
    patches = [
        Ellipse(
            (ell.center_frac[0] * w, ell.center_frac[1] * h),
            width=2 * ell.semi_axes_frac[0] * w,
            height=2 * ell.semi_axes_frac[1] * h,
            fill=False, linestyle="--", linewidth=1.0, edgecolor=TOPOLOGY_COLORS[Topology.E],
        )
    ]
    lin = cfg.placement.linear
    if lin.axis == "x":
        xy, rw, rh = (0.0, lin.center_frac * h - lin.width_m / 2), w, lin.width_m
    else:
        xy, rw, rh = (lin.center_frac * w - lin.width_m / 2, 0.0), lin.width_m, h
    patches.append(Rectangle(xy, rw, rh, fill=False, linestyle="--", linewidth=1.0,
                             edgecolor=TOPOLOGY_COLORS[Topology.L]))
    if net.r_region is not None:
        x0, y0, rw, rh = net.r_region
        patches.append(Rectangle((x0, y0), rw, rh, fill=False, linestyle="--", linewidth=1.0,
                                 edgecolor=TOPOLOGY_COLORS[Topology.R]))
    return patches


def default_title(net: PhysicalNetwork, part: Partition, cfg: Config) -> str:
    """Başlık: N, R, seed, R yerleşim modu, topoloji başına kopuk sayısı, yeniden üretim sayısı."""
    disc_txt = ", ".join(f"{t.value}: {len(part.disconnected[t])}" for t in TOPOLOGY_ORDER)
    return (
        f"N = {net.n_sensors}, R = {net.radio_range_m:g} m, seed = {net.seed}, "
        f"r_placement = {cfg.placement.random.r_placement}\n"
        f"kopuk sensör ({disc_txt}); yeniden üretim: {net.regen_attempts}"
    )


def draw_topology(ax, net: PhysicalNetwork, part: Partition, cfg: Config,
                  title: str | None = None, legend: bool = True) -> None:
    """Verilen eksene ağı çizer (§5.1 kenarlar, §5.2 bölgeler, §5.3 kopuk düğümler, §5.4 edge router)."""
    w, h = cfg.network.area_m
    ax.set_facecolor(SURFACE)

    pos = net.positions()
    segments = [(pos[u], pos[v]) for u, v in net.graph.edges()]
    ax.add_collection(LineCollection(segments, colors=EDGE_GRAY, linewidths=0.5, alpha=0.7, zorder=1))

    for patch in _region_patches(cfg, net):
        ax.add_patch(patch)

    for t in TOPOLOGY_ORDER:
        ids = sorted(part.members[t])
        if ids:
            ax.scatter(pos[ids, 0], pos[ids, 1], s=26, color=TOPOLOGY_COLORS[t], edgecolors=SURFACE,
                       linewidths=0.6, zorder=3)

    disc = sorted(v for s in part.disconnected.values() for v in s)
    if disc:
        ax.scatter(pos[disc, 0], pos[disc, 1], s=70, marker="x", color=INK, linewidths=1.2, zorder=4)

    for t in TOPOLOGY_ORDER:
        x, y = net.nodes[net.edge_router_ids[t]].pos
        ax.scatter([x], [y], s=150, marker="s", color=TOPOLOGY_COLORS[t], edgecolors=INK,
                   linewidths=1.3, zorder=5)
        ax.annotate(f"Edge-{t.value}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=INK_SECONDARY, zorder=6)

    margin = 0.03 * max(w, h)
    ax.set_xlim(-margin, w + margin)
    ax.set_ylim(-margin, h + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=INK_SECONDARY)
    ax.set_ylabel("y (m)", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(EDGE_GRAY)
    ax.set_title(title if title is not None else default_title(net, part, cfg),
                 fontsize=10, color=INK, loc="left")

    if legend:
        ax.legend(handles=legend_handles(part), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
                  fontsize=8, frameon=False, labelcolor=INK)


def legend_handles(part: Partition | None = None) -> list[Line2D]:
    """Lejant öğeleri: üç topoloji (part verilirse üye sayısıyla), edge router, kopuk işareti, kenar."""
    handles = [
        Line2D([], [], marker="o", linestyle="", color=TOPOLOGY_COLORS[t],
               label=TOPOLOGY_LABELS[t] + (f"  n={len(part.members[t])}" if part is not None else ""))
        for t in TOPOLOGY_ORDER
    ]
    handles.append(Line2D([], [], marker="s", linestyle="", color=INK_SECONDARY,
                          markeredgecolor=INK, label="edge router (sabit)"))
    handles.append(Line2D([], [], marker="x", linestyle="", color=INK, label="edge router'a yolu yok"))
    handles.append(Line2D([], [], color=EDGE_GRAY, linewidth=1, label="menzil içi bağlantı (≤ R)"))
    return handles


def plot_topology(net: PhysicalNetwork, part: Partition, cfg: Config, out_path: Path) -> Path:
    """Tek panelli şekli PNG'ye yazar."""
    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    draw_topology(ax, net, part, cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Komut satırı: config + seed -> ağ üret (§5.1 ölçütüyle), bölümle, çiz, özet yazdır."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=None, help="varsayılan: config'deki ilk seed")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "figures" / "topology.png"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override, ör. --set placement.random.r_placement=uniform")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides = dict(parse_override(s) for s in args.set)
    cfg = load_config(args.config, overrides=overrides)
    seed = args.seed if args.seed is not None else cfg.run.seeds[0]

    pn = build_partitioned_network(cfg, seed)
    out = plot_topology(pn.net, pn.part, cfg, Path(args.out))

    rec = pn.to_record()
    print(f"seed={seed} N={pn.net.n_sensors} kenar={pn.net.graph.number_of_edges()} "
          f"r_placement={rec['r_placement']} relay_mode={rec['relay_mode']} "
          f"yeniden_uretim={rec['regen_attempts']} fiziksel_bagli={rec['physical_connected']}")
    for t in TOPOLOGY_ORDER:
        print(f"  {t.value}: uye={len(pn.part.members[t])} kopuk={len(pn.part.disconnected[t])}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

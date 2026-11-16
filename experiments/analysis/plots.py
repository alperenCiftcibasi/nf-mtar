"""Grafikler (CLAUDE.md §5.19 "Grafikler"; results/figures/*.png, 300 dpi).

Her fonksiyon düzenli (tidy) bir DataFrame alır, PNG yazar ve yolunu döndürür. Kaynaklar:
- runs.csv (stats.normalize_runs biçimi): e1_throughput_by_class, e1_latency_by_class, e1_latency_percentiles,
  e2_scalability, e8_lifetime.
- paket logları (metrics.packet_log_frame + `method` sütunu): e1_latency_cdf, e3_latency_vs_hops.
- Henüz üreticisi olmayan kaynaklar (E5, E8 root enerjisi, eğitim, E9): beklenen sütunlar her fonksiyonun
  docstring'inde; üretici yazılınca sütun adları buna uydurulur ya da burada güncellenir.

Görsel kurallar (tek palet, okunur font):
- Yöntem rengi sabit sıradan gelir (METHOD_COLORS), sıralamaya göre değil; her yöntemin ayrıca kendi işaretçisi var
  (renk tek başına kimlik taşımasın; açık renklerin yüzeye karşıtlığı < 3:1 olduğu için).
- Topoloji renkleri plot_topology.py ile aynı (E mavi, L turuncu, R su yeşili).
- Tek y ekseni; ölçekleri farklı ölçüler ayrı panellere.
- Hata çubukları: seed'ler arası %95 GA (t dağılımı; stats.mean_ci).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
if "analysis_stats" in sys.modules:
    st = sys.modules["analysis_stats"]
else:
    _spec = importlib.util.spec_from_file_location("analysis_stats", _HERE / "stats.py")
    st = importlib.util.module_from_spec(_spec)
    sys.modules["analysis_stats"] = st
    _spec.loader.exec_module(st)

DPI = 300

# Tek palet (kategorik sıra sabit; renk varlığa bağlı, sıralamaya değil)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

METHOD_COLORS = {m: SERIES[i] for i, m in enumerate(st.METHOD_ORDER)}
METHOD_MARKERS = {"B1-plain": "o", "B1": "s", "B2": "^", "B3": "D", "NF-MTAR": "P", "NF-MTAR-paper": "X"}
TOPOLOGY_COLORS = {"E": SERIES[0], "L": SERIES[1], "R": SERIES[2]}
CLASS_LABELS = {"critical": "Kritik (E)", "regular": "Düzenli (L)", "noncritical": "Kritik olmayan (R)",
                "all": "Toplam"}
CLASS_TOPOLOGY = {"critical": "E", "regular": "L", "noncritical": "R"}
REG_MODE_COLORS = {"anchored": SERIES[0], "zero_l2": SERIES[1], "none": SERIES[2], "B3": MUTED}
INPUT_LABELS = ("x1 talep", "x2 kullanım", "x3 tükenme", "x4 link kalitesi", "x5 bağlantısallık")   # §5.14


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "lines.linewidth": 2.0, "lines.markersize": 6, "errorbar.capsize": 0,
    })


def method_color(m: str) -> str:
    return METHOD_COLORS.get(m, MUTED)


def _methods(df: pd.DataFrame, methods: Sequence[str] | None) -> list[str]:
    present = sorted(df["method"].dropna().unique(), key=st.method_sort_key)
    return [m for m in (methods or present) if m in present]


def _finish(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _legend_below(fig, axes, ncol: int | None = None) -> None:
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.tight_layout()
    if handles:
        # figürün altına (eksen etiketlerinin dışına); bbox_inches="tight" kaydederken bu alanı da kapsar
        fig.legend(handles, labels, loc="upper center", ncol=ncol or min(len(labels), 6),
                   bbox_to_anchor=(0.5, 0.0))


def _summ(values: pd.Series) -> tuple[float, float, float]:
    r = st.mean_ci(values.to_numpy(dtype=float))
    return r["mean"], r["ci_lo"], r["ci_hi"]


# --------------------------------------------------------------------------- #
# runs.csv tabanlı grafikler
# --------------------------------------------------------------------------- #


def bar_by_class(runs: pd.DataFrame, metric: str, out: str | Path, ylabel: str, *, scale: float = 1.0,
                 methods: Sequence[str] | None = None, classes: Sequence[str] = st.CLASSES,
                 title: str | None = None, paper: Mapping[str, Mapping[str, float]] | None = None) -> Path:
    """Sınıf bazında çubuk + %95 GA (e1_throughput_by_class, e1_latency_by_class). Yük başına bir panel.

    runs: `<metric>_<sınıf>` sütunları, method, seed, load_multiplier. scale: gösterim çarpanı (ör. 100 -> %).
    paper: {makale_yöntemi: {sınıf: değer}} verilirse sınıf başına yatay kesik işaret olarak çizilir (aynı birim).
    """
    apply_style()
    runs = st.normalize_runs(runs)
    loads = sorted(runs["load_multiplier"].dropna().unique()) or [math.nan]
    meths = _methods(runs, methods)
    fig, axes = plt.subplots(1, len(loads), figsize=(3.6 * len(loads) + 0.6, 3.6), sharey=True, squeeze=False)
    width = 0.8 / max(len(meths), 1)
    x = np.arange(len(classes))
    for ax, load in zip(axes[0], loads):
        sub = runs[runs["load_multiplier"] == load] if not math.isnan(load) else runs
        for j, m in enumerate(meths):
            vals = [_summ(sub.loc[sub["method"] == m, f"{metric}_{c}"] * scale) if f"{metric}_{c}" in sub
                    else (math.nan,) * 3 for c in classes]
            mean = np.array([v[0] for v in vals])
            err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
            pos = x - 0.4 + width * (j + 0.5)
            ax.bar(pos, mean, width * 0.9, color=method_color(m), label=m, edgecolor=SURFACE, linewidth=0.8)
            ax.errorbar(pos, mean, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=0.9)
        if paper:
            for k, (pm, vals) in enumerate(paper.items()):
                for i, c in enumerate(classes):
                    if c in vals:
                        ax.hlines(vals[c], i - 0.42, i + 0.42, colors=INK if k == 0 else MUTED,
                                  linestyles="--" if k == 0 else ":", linewidth=1.2,
                                  label=f"Makale: {pm}" if i == 0 else None)
        ax.set_xticks(x, [CLASS_LABELS[c].replace(" (", "\n(") for c in classes])
        ax.grid(axis="x", visible=False)
        if not math.isnan(load):
            ax.set_title(f"Yük × {load:g}")
    axes[0][0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    _legend_below(fig, axes)
    return _finish(fig, out)


def latency_percentiles(runs: pd.DataFrame, out: str | Path, *, load: float | None = None,
                        percentiles: Sequence[int] = (50, 90, 95, 99), methods: Sequence[str] | None = None,
                        classes: Sequence[str] = st.CLASSES) -> Path:
    """e1_latency_percentiles: x = yüzdelik, çizgi = yöntem (seed ortalaması ± %95 GA), sınıf başına panel."""
    apply_style()
    runs = st.normalize_runs(runs)
    if load is not None:
        runs = runs[runs["load_multiplier"] == load]
    meths = _methods(runs, methods)
    fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 3.4), squeeze=False)
    for ax, c in zip(axes[0], classes):
        for m in meths:
            sub = runs[runs["method"] == m]
            vals = [_summ(sub[f"lat_p{p}_ms_{c}"]) if f"lat_p{p}_ms_{c}" in sub else (math.nan,) * 3
                    for p in percentiles]
            mean = np.array([v[0] for v in vals])
            err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
            ax.errorbar(percentiles, mean, yerr=np.nan_to_num(err), color=method_color(m),
                        marker=METHOD_MARKERS.get(m, "o"), label=m, elinewidth=0.9)
        ax.set_title(CLASS_LABELS[c])
        ax.set_xticks(list(percentiles), [f"p{p}" for p in percentiles])
        ax.set_xlabel("Yüzdelik")
    axes[0][0].set_ylabel("Uçtan uca gecikme (ms)")
    if load is not None:
        fig.suptitle(f"Gecikme yüzdelikleri, yük × {load:g}")
    _legend_below(fig, axes)
    return _finish(fig, out)


def scalability(runs: pd.DataFrame, out: str | Path, *, x_col: str = "n_nodes",
                metric: str = "lat_mean_ms_critical", ylabel: str = "Kritik sınıf ort. gecikme (ms)",
                methods: Sequence[str] | None = None, paper: Mapping[str, Mapping[int, float]] | None = None,
                xlabel: str = "Sensör sayısı N (yoğunluk sabit)", log_x: bool = False) -> Path:
    """e2_scalability (ve e4_packet_size): x = sayısal ızgara değişkeni, y = metrik (seed ortalaması ± %95 GA);
    paper verilirse kesik referans."""
    apply_style()
    meths = _methods(runs, methods)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in meths:
        sub = runs[runs["method"] == m]
        g = sorted(sub[x_col].dropna().unique())
        vals = [_summ(sub.loc[sub[x_col] == v, metric]) for v in g]
        mean = np.array([v[0] for v in vals])
        ax.plot(g, mean, color=method_color(m), marker=METHOD_MARKERS.get(m, "o"), label=m)
        ax.fill_between(g, [v[1] for v in vals], [v[2] for v in vals], color=method_color(m), alpha=0.15,
                        linewidth=0)
    for k, (pm, vals) in enumerate((paper or {}).items()):
        xs = sorted(vals)
        ax.plot(xs, [vals[v] for v in xs], color=INK if k == 0 else MUTED, linestyle="--", linewidth=1.2,
                marker="x", label=f"Makale: {pm}")
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    return _finish(fig, out)


def lifetime(runs: pd.DataFrame, out: str | Path, *, methods: Sequence[str] | None = None) -> Path:
    """e8_lifetime: FND ve HND (s), yöntem başına çubuk + %95 GA (ölüm olmayan seed'ler NaN, sayıya girmez)."""
    apply_style()
    meths = _methods(runs, methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    metrics = [("fnd_s", "FND"), ("hnd_s", "HND")]
    width = 0.8 / max(len(meths), 1)
    x = np.arange(len(metrics))
    for j, m in enumerate(meths):
        sub = runs[runs["method"] == m]
        vals = [_summ(sub[c]) if c in sub else (math.nan,) * 3 for c, _ in metrics]
        mean = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, mean, width * 0.9, color=method_color(m), label=m, edgecolor=SURFACE, linewidth=0.8)
        ax.errorbar(pos, mean, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=0.9)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Ölüm zamanı (s)")
    ax.legend(loc="upper left")
    return _finish(fig, out)


# --------------------------------------------------------------------------- #
# Paket logu tabanlı grafikler
# --------------------------------------------------------------------------- #


def latency_cdf(packets: pd.DataFrame, out: str | Path, *, methods: Sequence[str] | None = None,
                classes: Sequence[str] = st.CLASSES, log_x: bool = True,
                deadlines_ms: Mapping[str, float] | None = None) -> Path:
    """e1_latency_cdf: ECDF, sınıf başına panel, çizgi = yöntem (seed'ler birleştirilmiş, teslim edilenler).

    packets: method, traffic_class, latency_ms [, delivered, after_warm_up]. deadlines_ms: {sınıf: π3} dikey çizgi.
    """
    apply_style()
    pk = packets
    if "delivered" in pk:
        pk = pk[pk["delivered"].astype(bool)]
    if "after_warm_up" in pk:
        pk = pk[pk["after_warm_up"].astype(bool)]
    meths = _methods(pk, methods)
    fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 3.4), squeeze=False)
    for ax, c in zip(axes[0], classes):
        for m in meths:
            v = np.sort(pk.loc[(pk["method"] == m) & (pk["traffic_class"] == c), "latency_ms"].to_numpy(dtype=float))
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=method_color(m), label=m,
                        linewidth=1.6)
        if deadlines_ms and c in deadlines_ms:
            ax.axvline(deadlines_ms[c], color=MUTED, linestyle="--", linewidth=1.0, label="π3 süre sınırı")
        if log_x:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.0)
        ax.set_title(CLASS_LABELS[c])
        ax.set_xlabel("Gecikme (ms)")
    axes[0][0].set_ylabel("Birikimli oran")
    _legend_below(fig, axes)
    return _finish(fig, out)


def latency_vs_hops(packets: pd.DataFrame, out: str | Path, *, methods: Sequence[str] | None = None,
                    traffic_class: str | None = None, max_hops: int | None = None,
                    paper: Mapping[str, Mapping[int, float]] | None = None) -> tuple[Path, pd.DataFrame]:
    """e3_latency_vs_hops: atlama sayısına göre ortalama gecikme ve doğrusal eğim (ms/atlama), yöntem başına.

    Dönüş: (png yolu, eğim tablosu [method, slope_ms_per_hop, intercept_ms, n_packets]).
    """
    apply_style()
    pk = packets
    if "delivered" in pk:
        pk = pk[pk["delivered"].astype(bool)]
    if "after_warm_up" in pk:
        pk = pk[pk["after_warm_up"].astype(bool)]
    if traffic_class:
        pk = pk[pk["traffic_class"] == traffic_class]
    if max_hops:
        pk = pk[pk["hops"] <= max_hops]
    meths = _methods(pk, methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    slopes = []
    for m in meths:
        sub = pk[pk["method"] == m]
        g = sub.groupby("hops")["latency_ms"].mean()
        ax.plot(g.index, g.values, color=method_color(m), marker=METHOD_MARKERS.get(m, "o"), label=m)
        if sub["hops"].nunique() >= 2:
            b, a = np.polyfit(sub["hops"].to_numpy(float), sub["latency_ms"].to_numpy(float), 1)
        else:
            b, a = math.nan, math.nan
        slopes.append({"method": m, "slope_ms_per_hop": b, "intercept_ms": a, "n_packets": len(sub)})
    for k, (pm, vals) in enumerate((paper or {}).items()):
        xs = sorted(vals)
        ax.plot(xs, [vals[v] for v in xs], color=INK if k == 0 else MUTED, linestyle="--", linewidth=1.2,
                marker="x", label=f"Makale: {pm}")
    ax.set_xlabel("Atlama sayısı")
    ax.set_ylabel("Ortalama gecikme (ms)")
    ax.legend(loc="upper left")
    return _finish(fig, out), pd.DataFrame(slopes)


# --------------------------------------------------------------------------- #
# E8 root enerjisi, E5 arıza, eğitim, E9
# --------------------------------------------------------------------------- #


def root_energy(curves: pd.DataFrame, out: str | Path, *, methods: Sequence[str] | None = None) -> Path:
    """e8_root_energy: root görevi yapmış düğümlerin kalan enerji eğrileri (sıcak nokta analizi, §5.13).

    curves: method, topology (E/L/R), node, t_s, energy_J [, seed]. Yöntem başına panel, renk = topoloji.
    Birden çok seed varsa yalnızca en küçük seed çizilir (okunurluk).
    """
    apply_style()
    if "seed" in curves:
        curves = curves[curves["seed"] == curves["seed"].min()]
    meths = _methods(curves, methods)
    fig, axes = plt.subplots(1, len(meths), figsize=(3.4 * len(meths), 3.3), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], meths):
        sub = curves[curves["method"] == m]
        for (topo, node), g in sub.groupby(["topology", "node"]):
            g = g.sort_values("t_s")
            ax.plot(g["t_s"], g["energy_J"], color=TOPOLOGY_COLORS.get(topo, MUTED), linewidth=1.2, alpha=0.85,
                    label=f"Topoloji {topo}")
        ax.set_title(m)
        ax.set_xlabel("Zaman (s)")
    axes[0][0].set_ylabel("Kalan enerji (J)")
    _legend_below(fig, axes)
    return _finish(fig, out)


def root_failure_timeline(timeline: pd.DataFrame, out: str | Path, *, failure_s: float = 300.0,
                          window: tuple[float, float] = (300.0, 330.0), value_col: str = "pdr",
                          ylabel: str = "Teslim oranı (zaman dilimi)", methods: Sequence[str] | None = None) -> Path:
    """e5_root_failure: zaman dilimi başına teslim oranı, topoloji başına panel, çizgi = yöntem (seed ortalaması).

    timeline: method, topology, t_s (dilim başı), <value_col> [, seed]. failure_s'de dikey çizgi, window gölgeli.
    """
    apply_style()
    meths = _methods(timeline, methods)
    topos = [t for t in ("E", "L", "R") if t in set(timeline["topology"])]
    fig, axes = plt.subplots(1, len(topos), figsize=(3.8 * len(topos), 3.3), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        ax.axvspan(*window, color=GRID, alpha=0.7, linewidth=0, label="Arıza penceresi")
        ax.axvline(failure_s, color=INK_SECONDARY, linestyle="--", linewidth=1.0, label="Root öldürüldü")
        for m in meths:
            g = timeline[(timeline["method"] == m) & (timeline["topology"] == topo)].groupby("t_s")[value_col].mean()
            ax.plot(g.index, g.values, color=method_color(m), label=m, linewidth=1.6)
        ax.set_title(f"Topoloji {topo}")
        ax.set_xlabel("Zaman (s)")
    axes[0][0].set_ylabel(ylabel)
    _legend_below(fig, axes)
    return _finish(fig, out)


def training_curves(history: pd.DataFrame, out: str | Path, *, log_y: bool = True) -> Path:
    """training_curves: epoch'a göre kayıp; renk = koşu/koşul (`run`), çizgi stili = ayrım (train düz, val kesik).

    history: run, epoch, split ∈ {train, val}, loss.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    runs = list(dict.fromkeys(history["run"]))
    if len(runs) > len(SERIES):
        raise ValueError(f"training_curves: en çok {len(SERIES)} koşu (renk döngüsü yok); küçük çoklu kullanın")
    for i, r in enumerate(runs):
        for split, ls in (("train", "-"), ("val", "--")):
            g = history[(history["run"] == r) & (history["split"] == split)].sort_values("epoch")
            if len(g):
                ax.plot(g["epoch"], g["loss"], color=SERIES[i], linestyle=ls, linewidth=1.6,
                        label=f"{r} ({'eğitim' if split == 'train' else 'doğrulama'})")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kayıp E (MSE + düzenlileştirme)")
    ax.legend(loc="upper right", ncol=1 if len(runs) <= 3 else 2)
    return _finish(fig, out)


def e9_pareto(summary: pd.DataFrame, out: str | Path, *, x_col: str = "violation", y_col: str = "regret",
              xlabel: str = "Yorumlanabilirlik ihlali", ylabel: str = "Doğrulama pişmanlığı") -> Path:
    """e9_pareto: x = yorumlanabilirlik ihlali, y = pişmanlık (§5.20); renk = reg_mode, nokta etiketi = λ.

    summary: reg_mode ∈ {anchored, zero_l2, none, B3}, lam (NaN: none/B3), <x_col>, <y_col>
    [, <x_col>_sd, <y_col>_sd: tekrarlar arası std -> hata çubuğu]. Pareto cephesi gri kesik çizgi.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    markers = {"anchored": "o", "zero_l2": "s", "none": "^", "B3": "D"}
    for mode, g in summary.groupby("reg_mode", sort=False):
        g = g.sort_values("lam") if "lam" in g else g
        xerr = g[f"{x_col}_sd"] if f"{x_col}_sd" in g else None
        yerr = g[f"{y_col}_sd"] if f"{y_col}_sd" in g else None
        ax.errorbar(g[x_col], g[y_col], xerr=xerr, yerr=yerr, color=REG_MODE_COLORS.get(mode, MUTED),
                    marker=markers.get(mode, "o"), linestyle="-" if len(g) > 1 else "none", linewidth=1.2,
                    elinewidth=0.8, label=mode)
        if "lam" in g:
            for _, r in g.iterrows():
                if pd.notna(r["lam"]):
                    ax.annotate(f"λ={r['lam']:g}", (r[x_col], r[y_col]), textcoords="offset points", xytext=(5, 4),
                                fontsize=8, color=INK_SECONDARY)
    front = pareto_front(summary[[x_col, y_col]].to_numpy(float))
    if len(front) > 1:
        ax.plot(front[:, 0], front[:, 1], color=MUTED, linestyle="--", linewidth=1.0, zorder=0, label="Pareto cephesi")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return _finish(fig, out)


def pareto_front(points: np.ndarray) -> np.ndarray:
    """İki ölçütte de küçük = iyi; baskılanmayan noktalar x'e göre sıralı."""
    pts = points[~np.isnan(points).any(axis=1)]
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    front, best_y = [], math.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return np.array(front)


def _gauss(x: np.ndarray, c: float, s: float) -> np.ndarray:
    return np.exp(-(((x - c) / s) ** 2))   # §5.14 makale Denk. 9 (MATLAB gaussmf değil)


def term_shapes(before: Mapping, after: Mapping, out: str | Path, *, input_labels: Sequence[str] = INPUT_LABELS,
                before_label: str = "Uzman (θ0)", after_label: str = "Eğitilmiş") -> Path:
    """e9_term_shapes: girdi başına üyelik fonksiyonları, önce (gri kesik) / sonra (renkli düz).

    before/after: FuzzyParams.to_dict() biçimi ({"centers": [[...] × 5], "sigmas": [[...] × 5], ...}).
    """
    apply_style()
    n_in = len(before["centers"])
    x = np.linspace(0.0, 1.0, 400)
    fig, axes = plt.subplots(1, n_in, figsize=(2.9 * n_in, 2.8), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        for c, s in zip(before["centers"][i], before["sigmas"][i]):
            ax.plot(x, _gauss(x, c, s), color=MUTED, linestyle="--", linewidth=1.2, label=before_label)
        for c, s in zip(after["centers"][i], after["sigmas"][i]):
            ax.plot(x, _gauss(x, c, s), color=SERIES[0], linewidth=1.8, label=after_label)
        ax.set_title(input_labels[i] if i < len(input_labels) else f"x{i + 1}")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Normalize girdi")
    axes[0][0].set_ylabel("Üyelik μ")
    _legend_below(fig, axes, ncol=2)
    return _finish(fig, out)


def expert_params(n_terms: Sequence[int] = (5, 5, 5, 2, 3)) -> dict:
    """§5.14 uzman merkezleri ve σ'ları (eşit aralık, komşular 0.5'te kesişir): σ = (Δ/2)/√ln2."""
    centers = [list(np.linspace(0.0, 1.0, n)) for n in n_terms]
    sigmas = [[(1.0 / (n - 1) / 2.0) / math.sqrt(math.log(2.0))] * n for n in n_terms]
    return {"centers": centers, "sigmas": sigmas}

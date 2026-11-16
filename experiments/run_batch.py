"""Paralel koşu altyapısı (CLAUDE.md §5.19 genel düzen): yöntem × yük × seed [× parametre ızgarası] -> runs.csv satırları.

- Her görev bir `run_simulation` koşusudur (experiments/baseline/run.py ile aynı akış: config + override'lar -> ağ ->
  iz -> Simulation -> metrics.runs_row). Görevler `--workers` kadar süreçte (varsayılan: çekirdek sayısı) koşar.
- Tek yazıcı: satırları yalnızca ana süreç, her koşu bittikçe çıktı CSV'sine ekler (metrics.append_runs_row).
- Kaldığı yerden devam: çıktı CSV'sinde run_id'si bulunan görevler atlanır; yarıda kesilen bir toplu koşu aynı komutla
  yeniden başlatılabilir. run_id = <deney>-<yöntem>-s<seed>-l<yük>-<mod>[-<anahtar>=<değer>...] (ızgara değişkenleri).
- Izgara: `--grid energy.initial_J=5,10,20,50` (birden çok --grid kartezyen çarpım). Sabit override'lar `--set`.
  Her satıra `variant` (ızgara anahtar=değer'leri) ve `grid_<anahtar>` sütunları eklenir.
- Belirlenimcilik: koşu sonucu yalnızca (config, yöntem, seed)'e bağlıdır, süreç atamasına ve sırasına bağlı değildir.

Kullanım:
    python experiments/run_batch.py --experiment m6_energy --methods B1 --loads 2.0 --n-seeds 5 \\
        --grid energy.initial_J=5,10,20,50 [--seeds 1 2 3] [--duration 600] [--relay-mode strict] \\
        [--set key=value ...] [--workers 8] [--out results/tables/runs.csv] [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # bu çalışma kopyasının src'si (alt süreçlerde de)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from nfmtar.config import load_config, parse_override  # noqa: E402


def task_run_id(experiment: str, method: str, seed: int, load: float, relay_mode: str, variant: dict[str, Any]) -> str:
    """Belirlenimci koşu kimliği (devam etme anahtarı); metrics.run_id biçimi + ızgara değişkenleri."""
    tag = f"{experiment}-" if experiment else ""
    suffix = "".join(f"-{k}={v}" for k, v in variant.items())
    return f"{tag}{method}-s{seed}-l{load:g}-{relay_mode}{suffix}"


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Alt süreçte tek koşu; runs.csv satırını döndürür."""
    from nfmtar.metrics import runs_row
    from nfmtar.simulation import run_simulation

    cfg = load_config(task["config"], overrides=task["overrides"])
    result = run_simulation(cfg, task["method"], task["seed"])
    row = runs_row(result, task["experiment"])
    row["run_id"] = task["run_id"]
    row["variant"] = ";".join(f"{k}={v}" for k, v in task["variant"].items())
    for k, v in task["variant"].items():
        row[f"grid_{k}"] = v
    row["overrides"] = ";".join(f"{k}={v}" for k, v in sorted(task["overrides"].items()))
    return row


def parse_grid(items: list[str]) -> list[dict[str, Any]]:
    """['a.b=1,2', 'c=x,y'] -> kartezyen çarpım [{a.b: 1, c: x}, ...]; değerler YAML olarak ayrıştırılır."""
    axes: list[tuple[str, list[Any]]] = []
    for item in items:
        key, raw = item.split("=", 1)
        axes.append((key.strip(), [yaml.safe_load(v) for v in raw.split(",")]))
    if not axes:
        return [{}]
    return [dict(zip([k for k, _ in axes], combo)) for combo in itertools.product(*(vals for _, vals in axes))]


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_cfg = load_config(args.config)
    seeds = args.seeds or list(base_cfg.run.seeds[: args.n_seeds])
    fixed = dict(parse_override(s) for s in args.set)
    if args.duration is not None:
        fixed["simulation.duration_s"] = args.duration
    relay_mode = args.relay_mode or base_cfg.router.relay_mode
    fixed["router.relay_mode"] = relay_mode
    tasks = []
    for variant in parse_grid(args.grid):
        for method, load, seed in itertools.product(args.methods, args.loads, seeds):
            overrides = {**fixed, **variant, "traffic.load_multiplier": float(load)}
            tasks.append({
                "config": args.config, "experiment": args.experiment, "method": method, "seed": int(seed),
                "overrides": overrides, "variant": variant,
                "run_id": task_run_id(args.experiment, method, int(seed), float(load), relay_mode, variant),
            })
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--experiment", required=True, help="runs.csv experiment etiketi (or. m6_energy, E1)")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--loads", nargs="+", type=float, required=True, help="traffic.load_multiplier degerleri")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="varsayilan: config run.seeds'in ilk --n-seeds tanesi")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--duration", type=float, default=None, help="simulation.duration_s")
    parser.add_argument("--relay-mode", choices=["strict", "shared"], default=None)
    parser.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2,...", help="parametre izgarasi")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="sabit config override")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", default=None, help="cikti CSV (varsayilan: config run.runs_table)")
    parser.add_argument("--dry-run", action="store_true", help="gorevleri listele, kosma")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else PROJECT_ROOT / load_config(args.config).run.runs_table
    tasks = build_tasks(args)
    done: set[str] = set()
    if out.exists() and out.stat().st_size > 0:
        done = set(pd.read_csv(out, usecols=["run_id"])["run_id"].astype(str))
    todo = [t for t in tasks if t["run_id"] not in done]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already in {out}, {len(todo)} to run on {args.workers} workers")
    if args.dry_run:
        for t in todo:
            print("  ", t["run_id"])
        return 0
    if not todo:
        return 0

    from nfmtar.metrics import append_runs_row

    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
        futures = {pool.submit(run_task, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                row = fut.result()
            except Exception:  # noqa: BLE001 — bir koşunun hatası diğerlerini durdurmasın; sonda raporlanır
                failures.append((t["run_id"], traceback.format_exc()))
                print(f"[{i}/{len(todo)}] FAILED {t['run_id']}", flush=True)
                continue
            append_runs_row(row, out)
            print(f"[{i}/{len(todo)}] {t['run_id']}: wall {row['wall_clock_s']:.0f} s, "
                  f"pdr_total {row['pdr_total']:.3f}, elapsed {time.perf_counter() - t0:.0f} s", flush=True)
    for run_id, tb in failures:
        print(f"--- {run_id}\n{tb}")
    print(f"done in {time.perf_counter() - t0:.0f} s; {len(failures)} failures; -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tek koşu (M5; CLAUDE.md §5.10a, §5.13, §5.19): yöntem × seed × süre × yük × relay_mode -> runs.csv satırı + konsol tablosu.

Akış: config + override'lar -> ağ ve bölümleme (§5.1, §5.3; relay_mode §5.1 yeniden üretim ölçütünü de etkiler, bu
yüzden config'e yazılır) -> trafik izi (§5.6; yöntemden bağımsız) -> Simulation (§5.10a) -> metrikler (§5.13) ->
results/tables/runs.csv'ye bir satır (§5.19) [+ opsiyonel paket logu results/packet_logs/<run_id>.parquet].

Kullanım:
    python experiments/baseline/run.py --method B1 --seed 1 --duration 600 --load 1.0 --relay-mode strict
        [--warm-up 60] [--drain 5] [--experiment smoke] [--packet-log] [--set key=value ...]
        [--runs-table results/tables/runs.csv] [--no-append]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    import nfmtar  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfmtar.config import load_config, parse_override  # noqa: E402
from nfmtar.metrics import append_runs_row, network_metrics, per_class_table, run_id, runs_row, write_packet_log  # noqa: E402
from nfmtar.simulation import run_simulation  # noqa: E402

TABLE_COLUMNS = ["generated", "delivered", "pdr", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "queue_ms", "backoff_ms",
                 "aifs_ms", "transmission_ms", "processing_ms", "hops_mean", "deadline_ratio", "window_ratio",
                 "airtime_share", "drop_queue_overflow", "drop_channel_access_failure", "drop_link_loss", "drop_no_route",
                 "drop_node_dead", "drop_in_flight_at_end"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--method", default="B1", help="engine.methods anahtari (B1-plain, B1, B2, B3, NF-MTAR)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None, help="simulation.duration_s (uretim suresi)")
    parser.add_argument("--load", type=float, default=None, help="traffic.load_multiplier")
    parser.add_argument("--relay-mode", choices=["strict", "shared"], default=None, help="router.relay_mode")
    parser.add_argument("--warm-up", type=float, default=None, help="run.warm_up_s")
    parser.add_argument("--drain", type=float, default=None, help="simulation.drain_s")
    parser.add_argument("--experiment", default="", help="runs.csv experiment etiketi (or. smoke, E1)")
    parser.add_argument("--packet-log", action="store_true", help="paket logunu results/packet_logs/<run_id> altina yaz")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="ek config override")
    parser.add_argument("--runs-table", default=None, help="varsayilan: config run.runs_table")
    parser.add_argument("--no-append", action="store_true", help="runs.csv'ye yazma (yalnizca konsol)")
    args = parser.parse_args(argv)

    overrides = dict(parse_override(s) for s in args.set)
    if args.duration is not None:
        overrides["simulation.duration_s"] = args.duration
    if args.load is not None:
        overrides["traffic.load_multiplier"] = args.load
    if args.relay_mode is not None:
        overrides["router.relay_mode"] = args.relay_mode
    if args.warm_up is not None:
        overrides["run.warm_up_s"] = args.warm_up
    if args.drain is not None:
        overrides["simulation.drain_s"] = args.drain
    cfg = load_config(args.config, overrides=overrides)

    result = run_simulation(cfg, args.method, args.seed)
    row = runs_row(result, args.experiment)
    net_m = network_metrics(result)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"{row['run_id']}: N = {result.pn.net.n_sensors}, regen = {result.pn.net.regen_attempts}, "
          f"duration {result.duration_s:g} s (+{result.drain_s:g} s drain), warm-up {result.warm_up_s:g} s, "
          f"load {result.trace.load_multiplier:g}, relay {result.relay_mode}, vsdn_policy {result.vsdn_policy}")
    print(f"wall clock {result.wall_clock_s:.1f} s; hops completed {result.n_hops_completed}; overrides {overrides or '-'}")
    table = per_class_table(result)[TABLE_COLUMNS]
    print(table.T.to_string(float_format=lambda x: f"{x:.4g}"))
    print(f"energy total {net_m['energy_total_J']:.4f} J (tx {net_m['energy_tx_J']:.4f}, rx {net_m['energy_rx_J']:.4f}); "
          f"dead nodes {net_m['dead_nodes']}; FND {net_m['fnd_s']}; HND {net_m['hnd_s']}; "
          f"utilization {net_m['channel_utilization']:.3f}; root changes {net_m['root_changes_total']}; ttl drops {net_m['ttl_drops']}")
    print("decision ms (E/L/R): " + " / ".join(f"{net_m[f'decision_ms_{t}']:.2f}" for t in ("E", "L", "R"))
          + "; route updates: " + " / ".join(str(net_m[f"route_updates_{t}"]) for t in ("E", "L", "R")))

    if args.packet_log:
        out = write_packet_log(result, PROJECT_ROOT / cfg.run.results_dir / "packet_logs" / run_id(result, args.experiment),
                               cfg.run.packet_log.format)
        print(f"-> {out}")
    if not args.no_append:
        path = append_runs_row(row, args.runs_table, cfg)
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

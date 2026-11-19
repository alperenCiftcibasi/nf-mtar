# NF-MTAR Re-implementation: Neuro-Fuzzy Multi-Topology Adaptive Routing for Healthcare IoT

## Status (18 November 2026)

| Milestone | State |
|---|---|
| M0–M4: config, physical network and V-SDN partition, IEEE 802.15.4 radio/MAC (AIFS calibrated), traffic trace, router (§5.9 edge cost, B1/B2 costs, two-stage paths) | done |
| M5: event-driven simulation core (`simulation.py`) and metrics (`metrics.py`, incl. QoS satisfaction and Φ) | done |
| M6: operating point calibration (PRR curve, 20 J initial energy, E1 load grid {0.25, 0.5, 1}); B1-plain / B1 / B2 at 3 loads × 5 seeds | done |
| M7: zero-order Sugeno fuzzy engine (750 rules), PyTorch equivalent, feature extraction, B3 strategy, paper-rule calibration report | done |
| M9: rank (§5.17), dynamic-root controller with periodic, traffic (§5.18-2: per-class sliding window + Poisson significance guard, z = 3) and root-health triggers, ε hysteresis (default ε = 0.2 since 18 Nov) and min_dwell; root-change log records the trigger type; B3 runs in the simulator | done |
| E5-lite: forced root failure at t = 300 s, B1 vs B3, 3 seeds | done (`results/tables/e5_lite.csv`) |
| E9-lite (contribution K1, small scale): §5.16 oracle, training data (6 seeds, split by seed), trainer with anchored / zero_l2 / none, §5.20 interpretability measures | done (`results/tables/e9_lite.csv`) |
| M12: paired statistics, figures, paper comparison (`experiments/analysis/make_all.py`) | done (code; waits for full experiment runs) |
| E1 (partial): B1, B2, B3 × loads {0.25, 0.5} × 10 seeds; paired statistics (`E1_stats.csv`), B3 vs B1/B2 (`e1_b3_comparison.csv`), E1 figures | done |
| E9 λ sweep (small scale): anchored / zero_l2 λ ∈ {1e-3, 1e-2, 1e-1}, none; 2 repeats (`e9_lambda.csv`, `e9_pareto.png`) | done |
| Unit tests for oracle, trainer and interpretability; references (IEEE) in `references/README.md` | done |
| M8 at full scale (5 training repeats, more data) and NF-MTAR network runs | **missing** |
| Full experiments: E1 with 30 seeds, load 1, B1-plain and NF-MTAR; E2, E3 (packet logs), E4, E5 (30 seeds), E8, E9 network part | **missing** |
| Report and presentation PDFs | submitted 16 Nov (`report/report.pdf`, `presentation/presentation.pdf`, repo `aefc282`) |
| P1 first half (§5.21): rich training data (`data/e9_rich.npz`, 20 seeds × 3 loads × B1/B3, 2 J, 1800 s; `e9_data_summary.csv`), top-1 diagnosis (`top1_*.csv`), E9-lite rerun on rich data at λ = 1e-2 (`e9_lambda_rich.csv`, `e9_rich_vs_lite.csv`) | done |
| Full E9 (§5.22, contribution K1): {none, zero_l2, anchored, anchored+cov} × λ ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} × 3 repeats on the rich data (x2 = radio busy ratio), λ chosen per mode by validation regret (`e9_full_grid.csv`, `e9_full_selected.csv`, K1 comparison `e9_full_k1_cov_vs_anchored.csv`, `e9_pareto.png`, `e9_terms_x*.png`); NF-MTAR main model `results/models/nfmtar_main.json` (anchored+cov, λ = 1e-3), default in config | done |
| Post-submission plan (CLAUDE.md §5.21): P0 repo + traffic trigger; P1 richer training data, top-1 diagnosis, full E9; P2–P4 full experiments and live demo. Evaluations after 18 Nov use new seeds (11+); seeds 1–10 are the development set | in progress |


## Project Title

**Re-implementation and extension of NF-MTAR: a neuro-fuzzy multi-topology adaptive routing framework for QoS-aware healthcare IoT communications.**

Fuzzy Logic course term project. The original work is re-implemented in a purpose-built Python discrete-event simulator, compared against baseline routing strategies, and extended with an original contribution.

## Student Information

| | |
|---|---|
| Name | TODO |
| Student ID | TODO |
| Department | Computer Engineering, İstanbul University-Cerrahpaşa |
| Course | Fuzzy Logic (Bulanık Mantık) |
| Term | TODO |
| Instructor | TODO |

## Original Paper

M. Shabana Parveen and P. T. V. Bhuvaneswari, "A neuro-fuzzy multi-topology adaptive routing framework for QoS-aware healthcare IoT communications," *Scientific Reports*, vol. 16, art. 3330, 2026. doi: [10.1038/s41598-025-33235-9](https://doi.org/10.1038/s41598-025-33235-9)

The paper proposes NF-MTAR, which combines:

1. **V-SDN (virtual software-defined networking):** a central controller partitions one physical sensor network into three virtual topologies, Elliptical (critical traffic), Linear (regular traffic) and Random (non-critical traffic), each governed by its own QoS policy (paper Table 1).
2. **Neuro-fuzzy search engine:** nodes in each topology are scored with a 5-input fuzzy inference system, ranked, and the best node is selected as the topology root; the lowest-cost path is then used. Fuzzy parameters are tuned by backpropagation.

The authors used COOJA, MATLAB and OpenDaylight and did not release code or data. Exact numerical agreement is therefore not expected; trends and relative differences are compared.

## Problem Description

TODO: summarize the routing problem (heterogeneous healthcare traffic with different latency, reliability and energy requirements on a shared low-power IEEE 802.15.4 network), why a single-topology, single-metric protocol such as RPL is insufficient, and what NF-MTAR aims to improve (per-class throughput, latency, PDR, network lifetime).

## Method

The simulator is written from scratch in Python (no external network simulator). High-level closed loop, per topology, at each re-evaluation trigger:

1. **Measure:** feature vectors of all alive candidate nodes.
2. **Decide:** scoring engine, then rank, then root proposal.
3. **Apply:** the controller updates the root and the paths (with hysteresis).
4. **Forward:** traffic generation and multi-hop forwarding continue; energy, queues and losses are updated.
5. **Feed back:** metrics are collected.

All methods share the same network, traffic, radio and metric code; only the scoring/routing engine differs.

| Method | Scoring | Root | Path cost | Learning | V-SDN policy |
|---|---|---|---|---|---|
| B1-plain | none | static | hop count | no | no (single FIFO queue) |
| B1 (hop) | none | static | hop count | no | yes |
| B2 (energy) | none | static | ETX + depletion | no | yes |
| B3 (fuzzy, untrained) | fuzzy, expert parameters | dynamic | weighted | no | yes |
| NF-MTAR | fuzzy, trained | dynamic | weighted | yes | yes |

TODO: expand with the partitioning rule, the edge-cost formula, the rank formula and the root-switching rule once implemented.

## Fuzzy System Architecture

- **Type:** zero-order Takagi–Sugeno (weighted-average defuzzification).
- **Inputs (all normalized to [0, 1]):**

| Input | Meaning | Terms |
|---|---|---|
| x1 demand | traffic rate in the 2-hop neighborhood / topology total | 5 |
| x2 utilization | queue occupancy ratio | 5 |
| x3 depletion | 1 − remaining / initial energy | 5 |
| x4 link quality | mean PRR to in-topology neighbors | 2 |
| x5 connectivity | in-topology neighbor count / 15, clipped at 1 | 3 |

- **Membership functions:** Gaussian, μ(x) = exp(−((x − c)/σ)²), equally spaced centers, neighboring terms crossing at 0.5.
- **Rule base:** full grid, 5 × 5 × 5 × 2 × 3 = 750 rules; consequents initialized from an expert goodness function.
- **Inference:** product t-norm, weighted average of consequents, output normalized to [0, 1]; confidence = max rule firing / total firing.
- **Learning:** centers, widths and consequents are trained offline (SGD with momentum, anchored L2 regularization toward the expert initialization) against oracle labels; there is no separate neural network.

TODO: add an architecture figure and the trained-vs-untrained parameter comparison.

## Dataset

There is no public dataset. All data is synthetic and generated by the simulator:

- **Simulation scenarios** are fully defined by `configs/default.yaml` and a seed; every run is reproducible.
- **Training data** for the fuzzy engine is produced by `experiments/build_training_data.py`: network snapshots from baseline runs, labeled with an oracle score for every candidate root. The train/validation split is by seed.
- Generated files live in `data/` and are not committed (see `.gitignore`); the regeneration command is given under How to Run.

## Installation

```bash
git clone <repo-url>
cd nf-mtar
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Requirements

- Python 3.11 or newer
- Pinned dependencies in `requirements.txt`: numpy, networkx, scipy, pandas, matplotlib, pyyaml, torch (CPU build is sufficient), pyarrow (optional packet logs), pytest

## How to Run

All commands are run from the repository root. Every script prints its options with `--help`; every parameter not given on the command line comes from `configs/default.yaml` (single source of truth) and can be overridden with `--set key=value`.

### Tests

```bash
pytest                         # all unit tests (tests/)
pytest tests/test_stats.py -q  # statistics and analysis only
```

### Single run

One method × one seed; appends one row to `results/tables/runs.csv` and prints a per-class table.

```bash
python experiments/baseline/run.py --method B1 --seed 1 --duration 600 --load 1.0 --relay-mode strict \
    --experiment smoke [--packet-log] [--no-append] [--set key=value ...]
```

`--packet-log` also writes one row per packet to `results/packet_logs/<run_id>.parquet` (needed for the latency CDF and E3).

### Batch runs

Method × load × seed [× parameter grid], one process per core, resumable (runs whose `run_id` is already in the output CSV are skipped).

```bash
python experiments/run_batch.py --experiment E1 --methods B1-plain B1 B2 B3 NF-MTAR \
    --loads 0.25 0.5 1 --n-seeds 30 --duration 600
python experiments/run_batch.py --experiment m6_energy --methods B1 --loads 2.0 --n-seeds 5 \
    --grid energy.initial_J=5,10,20,50          # --dry-run lists the tasks without running them
```

TODO: batch runs do not write packet logs yet; E1 packet logs currently come from `run.py --packet-log`.

### Analysis

```bash
python experiments/analysis/make_all.py [--trained-model results/models/<run_id>.json]
```

reads `results/tables/runs.csv` (plus packet logs and auxiliary tables when present) and produces, skipping with a warning whatever is missing:

| Output | Script | Content |
|---|---|---|
| `results/tables/<experiment>_stats.csv` | `stats.py` | per experiment × load × class × metric: mean ± 95% CI; paired t-test, Wilcoxon signed-rank, d_z and 95% CI of the paired difference for every method pair; Friedman; Holm–Bonferroni; internal consistency checks (the run fails if t ≠ diff/SE, d_z ≠ t/√n, W or χ² out of range, ...) |
| `results/figures/*.png` (300 dpi) | `plots.py` | e1_throughput_by_class, e1_latency_by_class, e1_latency_percentiles, e1_latency_cdf, e3_latency_vs_hops, e2_scalability, e4_packet_size, e8_lifetime, e8_root_energy, e5_root_failure, training_curves, e9_pareto, e9_term_shapes |
| `results/tables/paper_comparison.csv` | `paper_comparison.py` | paper vs. our values, relative improvement (positive = NF-MTAR better), trend agreement |

The scripts can also be run on their own, e.g. `python experiments/analysis/stats.py --experiments E1 --holm-family cell`.
Holm family: by default all pairwise comparisons inside one (experiment, load, class, metric) cell; `--holm-family experiment` pools a whole experiment (more conservative).

TODO: producers for `e8_root_energy.csv`, `e5_timeline.csv`, `training_curves.csv` and `e9_summary.csv` do not exist yet; the expected columns are documented in the docstrings of `experiments/analysis/plots.py`.

### Other scripts

```bash
# physical network and V-SDN partition for one seed
python experiments/analysis/plot_topology.py --seed 1            # -> results/figures/topology.png

# E0: connectivity analysis of the disjoint partition (uniform vs region placement, 30 seeds)
python experiments/analysis/e0_connectivity.py                   # -> results/tables/e0_connectivity*.csv, results/figures/e0_topology_uniform_vs_region.png

# MAC AIFS calibration: saturated three-sender sweep; writes the best AIFS pair to configs/default.yaml
python experiments/calibration/mac_aifs.py                       # -> results/tables/mac_aifs_calibration.csv (--no-write: table only)

# M6 operating point calibration: reads runs.csv rows made by run_batch.py and applies the §5.18a rules
python experiments/calibration/m6_operating_point.py energy|load [--write]   # -> results/tables/m6_calibration.csv
python experiments/calibration/m6_operating_point.py baselines               # -> results/tables/m6_baselines.csv
```

TODO: full-scale training (`experiments/train.py`, M8) and the proposed-method scripts (`experiments/proposed/`) are not implemented yet; `experiments/build_training_data.py` and `experiments/e9_lite.py` cover the small-scale E9-lite only.

### Fuzzy engine, B3 and the lite experiments

```bash
# fuzzy engine demo: scores, confidence and the dominant rule for example nodes and for a real network
python experiments/demo/fuzzy_demo.py [--seed 1] [--pruned]
python experiments/analysis/fuzzy_paper_rules.py                 # paper rules 1/47/98 vs our expert consequents

# B3 (expert fuzzy engine, dynamic root with rank + hysteresis) runs like any other method
python experiments/baseline/run.py --method B3 --seed 1 --load 0.25 --no-append
python experiments/analysis/m6_plots.py                          # -> b3_check.csv, m6_pdr_by_class.png, m6_latency_by_class.png

# E5-lite: kill every topology root at t = 300 s (B1 vs B3, load 0.25, 3 seeds)
python experiments/e5_lite.py                                    # -> results/tables/e5_lite.csv

# E9-lite (contribution K1, small scale): oracle-labelled snapshots, then training with anchored / zero_l2 / none
python experiments/build_training_data.py                        # -> data/e9_lite.npz (B1, load 0.5, seeds 1-6, K = 10 s)
python experiments/e9_lite.py                                    # -> results/tables/e9_lite.csv, results/figures/e9_lite_terms.png

# Rich training data (CLAUDE.md §5.21): 2 J, until end of life or 1800 s, loads {0.25, 0.5, 1}, B1 and B3,
# seeds 101-120 split 14/3/3 by seed; ~1.9M rows, ~30 min on 8 workers (data/ is not in git)
python experiments/build_training_data.py --seeds $(seq 101 120) --val-seeds 115 116 117 --test-seeds 118 119 120 \
    --loads 0.25 0.5 1 --methods B1 B3 --initial-energy 2 --duration 1800 --until-dead --workers 8 --out data/e9_rich.npz
python experiments/analysis/e9_data_summary.py                   # -> results/tables/e9_data_summary.csv (x1-x5, oracle)
python experiments/analysis/top1_diagnosis.py                    # -> results/tables/top1_*.csv (§5.21 top-1 diagnosis)
python experiments/e9_lambda.py --data data/e9_rich.npz --lams 1e-2 --tag rich   # E9-lite rerun on rich data

# Full E9 (CLAUDE.md §5.22): 49 trainings on a fixed 200k-row subsample, ~2.5 h on 5 workers (~0.7 GB RAM each)
python experiments/e9_full.py --workers 5                        # -> e9_full_*.csv, e9_pareto.png, e9_terms_x*.png,
                                                                 #    results/models/e9_full/*.json, nfmtar_main.json
python experiments/e9_full.py --from-csv                         # rebuild tables/figures from e9_full_reps.csv
```

## Repository Structure

```
├── CLAUDE.md                   design decisions (§ numbers are cited in the code)
├── configs/
│   ├── default.yaml            all simulation, policy, engine and experiment parameters
│   └── paper_reference.yaml    values reported in the paper (for paper_comparison.py)
├── src/nfmtar/
│   ├── config.py               YAML -> dataclasses, --set overrides
│   ├── network/                graph.py (unit disk graph, node state), radio.py (802.15.4, PRR, energy), partition.py (V_E, V_L, V_R)
│   ├── traffic.py              method-independent traffic trace (common random numbers)
│   ├── router.py               edge cost, reverse Dijkstra, two-stage paths, bypass
│   ├── controller.py           V-SDN controller (triggers, hysteresis)
│   ├── engine/                 base, baselines (B1-plain, B1, B2, B3), fuzzy (+ fuzzy_torch), oracle, trainer, rank, interpretability
│   ├── metrics.py              per-class and network metrics, runs.csv row, packet log
│   └── simulation.py           event-driven closed-loop simulation
├── experiments/
│   ├── baseline/run.py         single run
│   ├── run_batch.py            parallel batch runner
│   ├── calibration/            MAC AIFS and M6 operating-point calibration
│   ├── diagnostics/            route statistics
│   ├── demo/                   fuzzy engine demo
│   ├── analysis/               stats, plots, paper comparison, make_all, topology and E0 scripts
│   ├── proposed/               TODO
│   ├── build_training_data.py  oracle-labelled training snapshots (E9-lite scale)
│   └── train.py                TODO (skeleton)
├── results/
│   ├── tables/                 runs.csv, *_stats.csv, paper_comparison.csv, calibration tables
│   ├── figures/                PNG figures
│   └── models/                 trained fuzzy parameters (JSON)
├── tests/                      pytest unit tests
├── data/                       generated training data (not committed)
├── presentation/               TODO
├── report/                     TODO
└── references/
```

## Reproduction Experiments

| ID | Paper reference | What is varied | Methods | Seeds |
|---|---|---|---|---|
| E1 | Fig. 4a, 5a, Table 4 | load multiplier in {0.25, 0.5, 1.0} (M6 calibration); per-class throughput, PDR, latency | B1-plain, B1, B2, B3, NF-MTAR | 30 |
| E2 | Fig. 6b | network size N in {20, 50, 100, 150, 200} at constant density | B1-plain, B1, B3, NF-MTAR | 20 |
| E3 | Fig. 6d | latency vs. hop count (derived from E1 packet logs) | same as E1 | — |
| E4 | Fig. 6c | critical packet size in {64, 128, 256, 512, 1024} B | TODO | 20 |
| E5 | motivation scenario (not tested in the paper) | forced root failure at t = 300 s; recovery time, losses in the failure window | B1, NF-MTAR, NF-MTAR-paper | 30 |
| E8 | Table 4 | network lifetime (FND, HND) with 0.5 J initial energy | B1, B2, B3, NF-MTAR | 15 |

Statistics: runs are paired by seed; paired t-test and Wilcoxon signed-rank test with effect size d_z and 95% CI, Friedman test across methods, Holm–Bonferroni correction. Paper values used for comparison are in `configs/paper_reference.yaml`.

## Proposed Extension

Chosen (K1, experiment E9; not yet run): *interpretability-preserving training* of the fuzzy engine — anchored regularization toward the expert rule base vs. plain L2 vs. none, λ ∈ {1e-4, 1e-3, 1e-2, 1e-1} — evaluated by interpretability measures (term-order violations, coverage, distinguishability, consequent monotonicity, deviation from the expert) and performance (validation MSE, Spearman, top-1, regret, and the E1 network metrics). Supporting ablations: input ablation (K3) and the hysteresis margin / confidence-gate sweep (K9). See `CLAUDE.md` §5.20.

## Experimental Results

Preliminary (M6, 5 seeds): `results/tables/m6_baselines.csv`, `results/tables/b3_check.csv`, `results/tables/e5_lite.csv`, figures `results/figures/m6_pdr_by_class.png` and `m6_latency_by_class.png`. The full E1–E9 results with 95% CIs and statistical tests are not yet produced.

## Limitations

- Control-message overhead of the V-SDN controller is assumed to be zero.
- The security level (π7) and redundancy (π8) policies of the paper are out of scope.
- Each node generates a single traffic class; partitions are disjoint and deterministic.
- Traffic is a pre-generated, method-independent trace (common random numbers): critical 64 B Poisson at 1 packet/s plus Poisson-started 30 s "critical-care episodes" at 10 packets/s (mean one per 5 min per node); regular 128–256 B periodic at 1 packet/s with ±10 % jitter and random phase; non-critical 512–1024 B Poisson at 0.2 packets/s. `load_multiplier` scales only the base rates. Each node draws from its own `SeedSequence.spawn` stream, so changing one class's parameters leaves the other classes' traces unchanged.
- The radio is a simplified IEEE 802.15.4 model (logistic distance-based PRR with d50 = 45 m, s = 3.5 m, first-order radio energy model, a shared channel without an explicit interference model).
- Several paper parameters are undefined or inconsistent; the choices made are documented in `CLAUDE.md` and in the report.
- TODO: extend after the experiments.

## References

1. M. Shabana Parveen and P. T. V. Bhuvaneswari, "A neuro-fuzzy multi-topology adaptive routing framework for QoS-aware healthcare IoT communications," *Scientific Reports*, vol. 16, art. 3330, 2026. doi:10.1038/s41598-025-33235-9
2. TODO: RPL / MRHOF, IEEE 802.15.4, first-order radio energy model (Heinzelman et al.), ANFIS (Jang), and any other sources used in the implementation. External code, if any, is credited here and in the source comments.

# NF-MTAR Re-implementation: Neuro-Fuzzy Multi-Topology Adaptive Routing for Healthcare IoT

## Status (16 November 2026)

| Milestone | State |
|---|---|
| M0–M4: config, physical network and V-SDN partition, IEEE 802.15.4 radio/MAC (AIFS calibrated), traffic trace, router (§5.9 edge cost, B1/B2 costs, two-stage paths) | done |
| M5: event-driven simulation core (`simulation.py`) and metrics (`metrics.py`, incl. QoS satisfaction and Φ) | done |
| M6: operating point calibration (PRR curve, 20 J initial energy, E1 load grid {0.25, 0.5, 1}); B1-plain / B1 / B2 at 3 loads × 5 seeds | done |
| M7: zero-order Sugeno fuzzy engine (750 rules), PyTorch equivalent, feature extraction, B3 strategy, paper-rule calibration report | done |
| M9 (minimal): rank (§5.17), dynamic-root controller with periodic + root-health triggers, ε hysteresis and min_dwell; B3 runs in the simulator | done (traffic trigger not implemented) |
| E5-lite: forced root failure at t = 300 s, B1 vs B3, 3 seeds | done (`results/tables/e5_lite.csv`) |
| M8: oracle labels and training of the fuzzy engine (NF-MTAR) | **missing** |
| Full experiments E1–E5, E8, E9 (contribution K1), statistics, paper comparison | **missing** |


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

```bash
pytest                                                           # unit tests (tests/)

# one simulation run (method, seed, duration, load, relay mode) -> console table + one row in results/tables/runs.csv
python experiments/baseline/run.py --method B1 --seed 1 --duration 600 --load 0.5 --relay-mode strict
python experiments/baseline/run.py --method B3 --seed 1 --load 0.25 --no-append    # dynamic fuzzy root (B3)

# parallel batch: method x load x seed [x parameter grid], one process per core, resumes from runs.csv by run_id
python experiments/run_batch.py --experiment m6_baselines --methods B1-plain B1 B2 --loads 0.25 0.5 1.0 --n-seeds 5
python experiments/run_batch.py --experiment m6_energy --methods B1 --loads 2.0 --grid energy.initial_J=5,10,20,50

# M6 operating point: apply the CLAUDE.md §5.18a rules to the batch results (--write updates configs/default.yaml)
python experiments/calibration/m6_operating_point.py energy|load|baselines
python experiments/analysis/m6_plots.py                          # -> b3_check.csv, m6_pdr_by_class.png, m6_latency_by_class.png

# fuzzy engine demo: scores, confidence and the dominant rule for example nodes and for a real network
python experiments/demo/fuzzy_demo.py [--seed 1] [--pruned]
python experiments/analysis/fuzzy_paper_rules.py                 # paper rules 1/47/98 vs our expert consequents

# E5-lite: kill every topology root at t = 300 s (B1 vs B3, load 0.25, 3 seeds)
python experiments/e5_lite.py                                    # -> results/tables/e5_lite.csv
```

Earlier analysis and calibration scripts:

```bash
# draw the physical network and the V-SDN partition for one seed
python experiments/analysis/plot_topology.py --seed 1            # -> results/figures/topology.png
python experiments/analysis/plot_topology.py --set placement.random.r_placement=uniform --out results/figures/topology_uniform.png

# E0: connectivity analysis of the disjoint partition (uniform vs region placement, 30 seeds)
python experiments/analysis/e0_connectivity.py                   # -> results/tables/e0_connectivity*.csv, results/figures/e0_topology_uniform_vs_region.png

# MAC AIFS calibration: saturated three-sender sweep; writes the best AIFS pair to configs/default.yaml
python experiments/calibration/mac_aifs.py                       # -> results/tables/mac_aifs_calibration.csv (--no-write: table only)
```

Not yet implemented: `experiments/build_training_data.py` and `experiments/train.py` (M8, training of NF-MTAR).

Outputs: one row per run in `results/tables/runs.csv`, figures in `results/figures/`, trained models in `results/models/`.

## Reproduction Experiments

| ID | Paper reference | What is varied | Methods | Seeds |
|---|---|---|---|---|
| E1 | Fig. 4a, 5a, Table 4 | load multiplier in {0.25, 0.5, 1} (M6 calibration); per-class throughput, PDR, latency | B1-plain, B1, B2, B3, NF-MTAR | 30 |
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

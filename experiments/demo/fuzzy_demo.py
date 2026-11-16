"""Bulanık motor demosu (CLAUDE.md §5.14, §5.15): örnek düğümler için skor (f_norm), güven ve en baskın kuralın metni.

İki bölüm:
1. Elle seçilmiş düğüm profilleri (en iyi / orta / en kötü ve tek girdisi bozuk profiller): her girdinin skora etkisi.
2. Gerçek ağ (seed 1, config varsayılanları): topoloji başına adayların §5.14 girdileri (`extract_features`;
   nominal trafik hızları, kuyruklar boş, enerji dolu — simülasyon dışı anlık görüntü) ve B3 skorlarına göre en iyi 3
   ve en kötü 1 aday. Rank (§5.17) bu demoda YOK; burada yalnızca f_norm sıralaması gösterilir.

Kullanım:
    python experiments/demo/fuzzy_demo.py [--seed 1] [--pruned]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # bu çalışma kopyasının src'si (kurulu paketten önce)

import numpy as np  # noqa: E402

from nfmtar.config import load_config  # noqa: E402
from nfmtar.engine.base import NodeFeatures  # noqa: E402
from nfmtar.engine.baselines import b3_strategy  # noqa: E402
from nfmtar.engine.fuzzy import FuzzyEngine, extract_features, nominal_rates_pps  # noqa: E402
from nfmtar.network.graph import TOPOLOGY_ORDER  # noqa: E402
from nfmtar.network.partition import build_partitioned_network  # noqa: E402
from nfmtar.network.radio import RadioModel  # noqa: E402
from nfmtar.router import Router  # noqa: E402

PROFILES: list[tuple[str, tuple[float, float, float, float, float]]] = [
    ("best (high demand, idle, fresh, good links, well connected)", (0.9, 0.05, 0.05, 0.95, 0.9)),
    ("medium (all inputs mid-range)",                               (0.5, 0.5, 0.5, 0.5, 0.5)),
    ("worst (low demand, full queue, depleted, poor links, isolated)", (0.1, 0.95, 0.95, 0.1, 0.1)),
    ("best but depleted (x3 = 0.9)",                                (0.9, 0.05, 0.9, 0.95, 0.9)),
    ("best but congested (x2 = 0.9)",                               (0.9, 0.9, 0.05, 0.95, 0.9)),
    ("best but poor links (x4 = 0.2)",                              (0.9, 0.05, 0.05, 0.2, 0.9)),
]


def show(engine: FuzzyEngine, rows: list[tuple[str, NodeFeatures]]) -> None:
    out = engine.evaluate_batch([f for _, f in rows])
    print(f"{'node':<64} {'x1':>5} {'x2':>5} {'x3':>5} {'x4':>5} {'x5':>5} {'H':>3} | {'score':>6} {'conf':>6}")
    for b, (label, f) in enumerate(rows):
        h = "-" if f.hops_to_edge is None else str(f.hops_to_edge)
        print(f"{label:<64} {f.x1:5.2f} {f.x2:5.2f} {f.x3:5.2f} {f.x4:5.2f} {f.x5:5.2f} {h:>3} | "
              f"{out.score[b]:6.3f} {out.confidence[b]:6.3f}")
        print(f"    top rule {engine.rule_to_text(int(out.top_rule_index[b]))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pruned", action="store_true", help="don't-care birlestirilmis kural tabani (§5.14)")
    args = parser.parse_args(argv)

    cfg = load_config()
    strategy = b3_strategy(cfg)
    engine = FuzzyEngine.from_config(cfg, pruned=args.pruned)
    print(f"B3 fuzzy engine: {engine.n_rules} rules, pruned={engine.pruned}; "
          f"sigma per input = {[round(float(s[0]), 3) for s in engine.params.sigmas]}")
    print(f"strategy: {strategy!r}, dynamic_root={strategy.dynamic_root()}\n")

    print("1) hand-made node profiles")
    show(engine, [(name, NodeFeatures(i, *x)) for i, (name, x) in enumerate(PROFILES)])

    print(f"\n2) real network, seed {args.seed} (nominal rates, empty queues, full energy)")
    pn = build_partitioned_network(cfg, args.seed)
    radio = RadioModel.from_config(cfg)
    router = Router.for_method(cfg, radio, "B3")
    rates = nominal_rates_pps(cfg, pn.net)
    for t in TOPOLOGY_ORDER:
        reach = router.reach_from_edge(pn.net, pn.part, t)
        feats = extract_features(engine.spec, pn.net, pn.part, t, rates, radio.prr, hops_to_edge=reach.hops)
        cands = [feats[v] for v in sorted(reach.candidates)]
        scores = engine.evaluate_batch(cands).score
        order = np.argsort(-scores, kind="stable")
        picks = [*order[:3], order[-1]]
        print(f"\ntopology {t.value}: {len(cands)} candidates; score min / median / max = "
              f"{scores.min():.3f} / {np.median(scores):.3f} / {scores.max():.3f}")
        show(engine, [(f"{t.value} node {cands[k].node_id} (rank {r + 1 if r < 3 else len(cands)} by score)", cands[k])
                      for r, k in enumerate(picks)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

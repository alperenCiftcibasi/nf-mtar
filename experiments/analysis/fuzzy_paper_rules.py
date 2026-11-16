"""Makaledeki örnek kurallar için kalibrasyon raporu: bizim uzman y_k'lerimiz vs makale (CLAUDE.md §5.14).

Makalenin "Examples of selected rules" bölümündeki üç kural (terimler config fuzzy.calibration_check.paper_rules'da):
  Kural 1:  x1 VeryHigh, x2 High,     x3 Low,      x4 High, x5 High   => 0,95
  Kural 47: x1 High,     x2 Medium,   x3 Medium,   x4 High, x5 Medium => 0,78
  Kural 98: x1 Low,      x2 VeryHigh, x3 VeryHigh, x4 Low,  x5 Low    => 0,15
(makalede x4 Good/Poor = bizim High/Low.) Her kural için iki değer raporlanır:
- y_ours: bizim tam kural tabanımızda aynı terim birleşimine sahip kuralın uzman sonucu y_k (§5.14 uzman fonksiyonu);
- f_at_centers / f_norm_at_centers: girdiler bu terimlerin merkezindeyken motorun çıkışı (komşu kurallar da ateşlenir).
Fark tanı amaçlıdır: makale kendi kural sonuçlarını nasıl ürettiğini vermez ve x2'de kendi içinde tutarsızdır
(Kural 1'de yüksek kullanım "Excellent", Kural 98'de çok yüksek kullanım "Poor"; CLAUDE.md §5.14). Test başarısız olmaz.

Çıktı: results/tables/fuzzy_paper_rules.csv

Kullanım:
    python experiments/analysis/fuzzy_paper_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfmtar.config import load_config  # noqa: E402
from nfmtar.engine.fuzzy import FuzzyEngine, paper_rule_report  # noqa: E402

OUT = PROJECT_ROOT / "results" / "tables" / "fuzzy_paper_rules.csv"


def main() -> int:
    cfg = load_config()
    engine = FuzzyEngine.from_config(cfg)
    rows = paper_rule_report(engine, cfg.fuzzy["calibration_check"]["paper_rules"])
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(df.drop(columns=["our_rule_text"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for r in rows:
        print(f"paper rule {r['paper_rule']}: {r['our_rule_text']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

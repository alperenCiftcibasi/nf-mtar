"""E5-lite — kök arıza enjeksiyonu (CLAUDE.md §5.19 E5'in kısaltılmış hali).

t = 300 s'de her topolojinin o anki root'u zorla öldürülür (enerji = 0). Yöntemler: B1 (sabit kural, §5.12: ölümde aynı
kuralla yeni root) ve B3 (§5.18 root sağlığı tetikleyicisi: r* hemen). Yük 0,25; 3 seed; koşu 360 s (ısınma 60 s).
Topoloji başına ölçütler:
- recovery_s: arızadan sonra OLUŞTURULAN paketlerden ilk başarılı teslimin arıza anına uzaklığı
- window_pdr / window_lost: 300–330 s penceresinde (oluşturma zamanına göre) PDR ve kayıp paket sayısı
- pre_pdr: arıza öncesi 270–300 s PDR (karşılaştırma için)
Çıktı: results/tables/e5_lite.csv

Kullanım: python experiments/e5_lite.py [--methods B1 B3] [--seeds 1 2 3] [--workers 6]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

T_FAIL, WINDOW_S, DURATION_S, LOAD = 300.0, 30.0, 360.0, 0.25
OUT = PROJECT_ROOT / "results" / "tables" / "e5_lite.csv"


def run_one(task: tuple[str, int]) -> list[dict]:
    from nfmtar.config import load_config
    from nfmtar.simulation import run_simulation

    method, seed = task
    cfg = load_config(overrides={"simulation.duration_s": DURATION_S, "traffic.load_multiplier": LOAD})
    res = run_simulation(cfg, method, seed, kill_roots_at_s=T_FAIL)
    rows = []
    for t in ("E", "L", "R"):
        pk = [ps for ps in res.packets if ps.topology.value == t]
        after = sorted(ps.end_s for ps in pk if ps.delivered and ps.packet.created_s >= T_FAIL)
        win = [ps for ps in pk if T_FAIL <= ps.packet.created_s < T_FAIL + WINDOW_S]
        pre = [ps for ps in pk if T_FAIL - WINDOW_S <= ps.packet.created_s < T_FAIL]
        change = [rc for rc in res.root_changes if rc["topology"] == t and rc["time_s"] == T_FAIL]
        rows.append({
            "method": method, "seed": seed, "topology": t,
            "old_root": change[0]["old"] if change else None, "new_root": change[0]["new"] if change else None,
            "recovery_s": (after[0] - T_FAIL) if after else float("nan"),
            "window_generated": len(win), "window_lost": sum(not ps.delivered for ps in win),
            "window_pdr": sum(ps.delivered for ps in win) / len(win) if win else float("nan"),
            "pre_pdr": sum(ps.delivered for ps in pre) / len(pre) if pre else float("nan"),
            "root_changes_total": len(res.root_changes),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["B1", "B3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    tasks = [(m, s) for m in args.methods for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for rs in pool.map(run_one, tasks) for r in rs]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary = df.groupby(["method", "topology"])[["recovery_s", "window_pdr", "window_lost", "pre_pdr"]].mean()
    print("\nmean over seeds:\n" + summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

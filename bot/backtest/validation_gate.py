"""Tier 0 validation gate. Must pass before any real money."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from bot.research.evaluate import run_evaluation
from bot.research.strategy import STARTING_BANKROLL

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 9999, 1111, 5555]

# Gate thresholds
MIN_AVG_WIN_RATE = 0.95
MAX_AVG_DRAWDOWN = 0.15
MIN_PROFITABLE_SEEDS = 8
MIN_AVG_RETURN = 0.05


def run_gate():
    results = []

    for seed in SEEDS:
        trades, final_bankroll, max_dd = run_evaluation(seed)
        num_trades = len(trades)
        wins = sum(1 for t in trades if t["won"])
        win_rate = wins / num_trades if num_trades > 0 else 0.0
        total_return = (final_bankroll - STARTING_BANKROLL) / STARTING_BANKROLL

        results.append({
            "seed": seed,
            "trades": num_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "final_bankroll": final_bankroll,
        })

    # Print header
    print("=" * 60)
    print("TIER 0 VALIDATION GATE")
    print("=" * 60)

    for r in results:
        print(
            f"  Seed {r['seed']}: {r['trades']} trades, "
            f"{r['win_rate']:.0%} WR, "
            f"{r['total_return']:.1%} return, "
            f"{r['max_drawdown']:.1%} DD"
        )

    # Compute aggregates
    avg_wr = sum(r["win_rate"] for r in results) / len(results)
    avg_dd = sum(r["max_drawdown"] for r in results) / len(results)
    profitable_count = sum(1 for r in results if r["total_return"] > 0)
    avg_return = sum(r["total_return"] for r in results) / len(results)

    # Check criteria
    criteria = [
        (avg_wr >= MIN_AVG_WIN_RATE, f"Win rate >= {MIN_AVG_WIN_RATE:.0%}", f"{avg_wr:.1%}"),
        (avg_dd < MAX_AVG_DRAWDOWN, f"Max drawdown < {MAX_AVG_DRAWDOWN:.0%}", f"{avg_dd:.1%}"),
        (profitable_count >= MIN_PROFITABLE_SEEDS, f"Profitable on {MIN_PROFITABLE_SEEDS}+/{len(SEEDS)} seeds", f"{profitable_count}/{len(SEEDS)}"),
        (avg_return > MIN_AVG_RETURN, f"Monthly return > {MIN_AVG_RETURN:.0%}", f"{avg_return:.1%}"),
    ]

    print()
    print("--- GATE CRITERIA ---")
    all_passed = True
    for passed, label, actual in criteria:
        tag = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{tag}] {label} (actual: {actual})")

    print()
    print("=" * 60)
    if all_passed:
        print("GATE: PASSED. Strategy is validated for live deployment.")
    else:
        print("GATE: FAILED. Run autoresearch to optimize, or adjust strategy.py.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_gate())

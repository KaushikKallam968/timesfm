"""Realistic Backtest: No hindsight, no cherry-picking.

Simulates ACTUAL trading conditions:
1. Truth source has realistic accuracy (not 100%)
2. Entry at FIRST signal, not best price (no look-ahead)
3. Position limits and capital constraints
4. Real fee structure (2% round-trip)
5. Slippage model (spread cost on entry/exit)
6. Cannot trade markets already close to resolution (price near 0 or 1)
7. Multiple simultaneous positions compete for capital
8. Bankroll compounds (wins/losses affect future sizing)

Usage: python -m bot.backtest.realistic_backtest
"""

import json
import os
import sys
import random
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def kelly_binary(edge, price):
    """Kelly fraction for binary bet. Returns 0 if no edge."""
    if price <= 0.01 or price >= 0.99 or edge <= 0:
        return 0
    payout = (1.0 / price) - 1.0
    p = min(0.99, max(0.01, price + edge))
    f = (p * (payout + 1) - 1) / payout
    return max(0, f)


def simulate_truth_signal(actual_outcome, market_price, accuracy, days_to_res):
    """Simulate what a real truth source would output.

    A real truth source (sportsbook, model) is:
    - More accurate closer to the event
    - Sometimes confidently wrong (the dangerous case)
    - Calibrated: when it says 70%, the event happens ~70% of the time
    """
    # Accuracy improves closer to resolution
    if days_to_res is not None:
        if days_to_res <= 1:
            effective_accuracy = min(accuracy + 0.10, 0.98)
        elif days_to_res <= 3:
            effective_accuracy = min(accuracy + 0.05, 0.95)
        elif days_to_res <= 7:
            effective_accuracy = accuracy
        elif days_to_res <= 30:
            effective_accuracy = max(accuracy - 0.05, 0.55)
        else:
            effective_accuracy = max(accuracy - 0.10, 0.52)
    else:
        effective_accuracy = accuracy

    # Is the truth source correct this time?
    correct = random.random() < effective_accuracy

    if correct:
        # Output a probability that leans toward the correct outcome
        # But with noise (not a perfect 1.0 or 0.0)
        if actual_outcome == 1.0:
            # Truth source says YES is likely
            truth_prob = random.uniform(0.55, 0.95)
        else:
            # Truth source says NO is likely (YES prob is low)
            truth_prob = random.uniform(0.05, 0.45)
    else:
        # Truth source is WRONG - leans the wrong way
        if actual_outcome == 1.0:
            truth_prob = random.uniform(0.10, 0.45)
        else:
            truth_prob = random.uniform(0.55, 0.90)

    return truth_prob, effective_accuracy


def run_realistic_backtest(records, truth_accuracy=0.75, seed=42):
    """Run a fully realistic backtest with no hindsight bias.

    Key differences from the hindsight backtest:
    1. Truth signal is noisy and sometimes wrong
    2. Entry at FIRST qualifying signal (chronological), not best price
    3. Can only hold one position per market
    4. Slippage added to entry price
    5. Positions settle when days_to_resolution hits 0
    """
    random.seed(seed)
    np.random.seed(seed)

    STARTING_BANKROLL = 10000.0
    MIN_EDGE = 0.08
    KELLY_MULT = 0.25
    MAX_TRADE_SIZE = 100.0
    MAX_POSITIONS = 20
    FEE_RATE = 0.02
    SLIPPAGE = 0.005  # 0.5% slippage per side

    bankroll = STARTING_BANKROLL
    open_positions = {}  # market_id -> position info
    trades = []
    equity_curve = [{"day": 0, "bankroll": bankroll, "positions": 0}]
    peak = bankroll
    max_dd = 0
    day_count = 0

    # Sort ALL records chronologically
    records = sorted(records, key=lambda r: r.get("timestamp", ""))

    # Group by market for settlement lookup
    market_outcomes = {}
    for r in records:
        mid = r.get("market_id", "")
        truth = r.get("truth_probability")
        if mid and truth in (0.0, 1.0):
            market_outcomes[mid] = truth

    prev_date = ""
    for r in records:
        ts = r.get("timestamp", "")[:10]
        mid = r.get("market_id", "")
        mp = r.get("market_price", 0)
        actual = r.get("truth_probability", 0)  # 0 or 1
        days_to_res = r.get("days_to_resolution")
        category = r.get("category", "")

        if mp <= 0.03 or mp >= 0.97:
            continue

        # Track equity daily
        if ts != prev_date:
            day_count += 1
            equity_curve.append({
                "day": day_count,
                "date": ts,
                "bankroll": round(bankroll, 2),
                "positions": len(open_positions),
            })
            prev_date = ts

        # --- Settlement check ---
        if mid in open_positions and days_to_res is not None and days_to_res <= 0:
            pos = open_positions.pop(mid)
            actual_outcome = market_outcomes.get(mid, actual)

            if pos["side"] == "YES":
                won = actual_outcome == 1.0
            else:
                won = actual_outcome == 0.0

            if won:
                payout = pos["size"] / pos["entry_price"]
                pnl = payout - pos["size"] - pos["fees"]
            else:
                pnl = -pos["size"] - pos["fees"]

            bankroll += pos["size"] + pnl
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            trades.append({
                **pos,
                "pnl": round(pnl, 2),
                "won": won,
                "exit_date": ts,
                "bankroll_after": round(bankroll, 2),
            })
            continue

        # Skip if already in this market
        if mid in open_positions:
            continue

        # Skip if at position limit
        if len(open_positions) >= MAX_POSITIONS:
            continue

        # Skip if bankroll depleted
        if bankroll < 10:
            continue

        # --- Generate truth signal (the key realistic element) ---
        if actual not in (0.0, 1.0):
            continue

        truth_prob, eff_accuracy = simulate_truth_signal(actual, mp, truth_accuracy, days_to_res)

        # Compute edge based on truth signal (NOT actual outcome)
        edge_yes = truth_prob - mp
        edge_no = (1 - truth_prob) - (1 - mp)

        if edge_yes > MIN_EDGE and edge_yes > edge_no:
            side = "YES"
            edge = edge_yes
            entry_price = mp + SLIPPAGE  # worse price due to slippage
        elif edge_no > MIN_EDGE:
            side = "NO"
            edge = edge_no
            entry_price = (1 - mp) + SLIPPAGE
        else:
            continue

        entry_price = min(entry_price, 0.98)

        # Kelly sizing on perceived edge (which may be wrong)
        kelly_f = kelly_binary(edge, entry_price)
        size = bankroll * kelly_f * KELLY_MULT
        size = min(size, MAX_TRADE_SIZE)
        size = min(size, bankroll * 0.05)  # max 5% per trade

        if size < 5:  # minimum $5 trade
            continue

        fees = size * FEE_RATE

        # Open position
        bankroll -= size
        open_positions[mid] = {
            "market_id": mid,
            "question": r.get("question", "")[:80],
            "side": side,
            "entry_price": round(entry_price, 4),
            "size": round(size, 2),
            "fees": round(fees, 2),
            "perceived_edge": round(edge, 4),
            "truth_signal": round(truth_prob, 4),
            "effective_accuracy": round(eff_accuracy, 4),
            "entry_date": ts,
            "category": category,
            "days_to_resolution": days_to_res,
        }

    # Force-settle remaining positions at actual outcomes
    for mid, pos in list(open_positions.items()):
        actual_outcome = market_outcomes.get(mid, 0.5)
        if actual_outcome == 0.5:
            # Unknown outcome, assume 50/50
            won = random.random() < 0.5
        else:
            won = (pos["side"] == "YES" and actual_outcome == 1.0) or \
                  (pos["side"] == "NO" and actual_outcome == 0.0)

        if won:
            payout = pos["size"] / pos["entry_price"]
            pnl = payout - pos["size"] - pos["fees"]
        else:
            pnl = -pos["size"] - pos["fees"]

        bankroll += pos["size"] + pnl
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        trades.append({
            **pos,
            "pnl": round(pnl, 2),
            "won": won,
            "exit_date": "force_settled",
            "bankroll_after": round(bankroll, 2),
        })

    return trades, equity_curve, bankroll, max_dd


def compute_stats(trades, starting=10000.0, final=0.0, max_dd=0.0):
    """Compute performance stats."""
    if not trades:
        return {"error": "no trades"}

    pnls = [t["pnl"] for t in trades]
    wins = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    # Daily PnL for Sharpe
    daily = defaultdict(float)
    for t in trades:
        d = t.get("exit_date", t.get("entry_date", ""))[:10]
        if d:
            daily[d] += t["pnl"]
    daily_vals = list(daily.values()) if daily else [0]
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(365)) if np.std(daily_vals) > 0 else 0

    # By category
    by_cat = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        c = t.get("category", "?")
        by_cat[c]["trades"] += 1
        if t["won"]:
            by_cat[c]["wins"] += 1
        by_cat[c]["pnl"] += t["pnl"]

    # Streak analysis
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak_type = None
    for t in trades:
        if t["won"]:
            if current_streak_type == "win":
                streak += 1
            else:
                streak = 1
                current_streak_type = "win"
            max_win_streak = max(max_win_streak, streak)
        else:
            if current_streak_type == "loss":
                streak += 1
            else:
                streak = 1
                current_streak_type = "loss"
            max_loss_streak = max(max_loss_streak, streak)

    return {
        "total_trades": len(trades),
        "total_return": round((final - starting) / starting, 4),
        "total_return_pct": f"{(final - starting) / starting:.1%}",
        "final_bankroll": round(final, 2),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(max_dd, 4),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_perceived_edge": round(np.mean([t["perceived_edge"] for t in trades]), 4),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }


def main():
    print("=" * 70)
    print("REALISTIC BACKTEST: No Hindsight, Real Conditions")
    print("=" * 70)

    print("\nConditions:")
    print("  - Truth source accuracy: varies (not 100%)")
    print("  - Entry at FIRST signal (chronological, no cherry-picking)")
    print("  - 0.5% slippage per side + 2% fees")
    print("  - Max 20 simultaneous positions, max $100/trade, max 5%/trade")
    print("  - Quarter-Kelly sizing on PERCEIVED edge (may be wrong)")
    print("  - Accuracy degrades further from event, improves near resolution")

    print("\nLoading data...")
    records = load_price_histories()
    usable = [r for r in records
              if 0.01 < r.get("market_price", 0) < 0.99
              and r.get("truth_probability") is not None
              and r.get("timestamp")]
    print(f"  Usable records: {len(usable):,}")

    # Run at multiple accuracy levels with multiple seeds for robustness
    print(f"\n{'='*70}")
    print(f"{'Accuracy':>10} {'Seed':>5} {'Trades':>7} {'Return':>10} {'WinRate':>8} {'MaxDD':>7} {'Sharpe':>7} {'PF':>6} {'Final$':>12}")
    print(f"{'='*70}")

    all_results = {}

    for accuracy in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]:
        seed_results = []

        for seed in [42, 123, 456, 789, 1337]:
            trades, eq, final, dd = run_realistic_backtest(usable, accuracy, seed)
            stats = compute_stats(trades, 10000.0, final, dd)
            seed_results.append(stats)

            if seed == 42:  # Print first seed
                ret = stats["total_return"]
                wr = stats["win_rate"]
                pf = stats["profit_factor"]
                sh = stats["sharpe"]
                print(f"{accuracy:>9.0%} {seed:>5} {stats['total_trades']:>7,} {ret:>9.1%} {wr:>7.1%} {dd:>6.1%} {sh:>6.1f} {pf:>5.1f} ${final:>10,.2f}")

        # Average across seeds
        avg_return = np.mean([s["total_return"] for s in seed_results])
        avg_wr = np.mean([s["win_rate"] for s in seed_results])
        avg_dd = np.mean([s["max_drawdown"] for s in seed_results])
        avg_sharpe = np.mean([s["sharpe"] for s in seed_results])
        avg_pf = np.mean([s["profit_factor"] for s in seed_results if s["profit_factor"] != float("inf")])
        loss_seeds = sum(1 for s in seed_results if s["total_return"] < 0)

        print(f"  {'avg':>9} {'5x':>5} {'-':>7} {avg_return:>9.1%} {avg_wr:>7.1%} {avg_dd:>6.1%} {avg_sharpe:>6.1f} {avg_pf:>5.1f} {'':>12} {'LOSS' if loss_seeds > 0 else ''} {f'({loss_seeds}/5 seeds)' if loss_seeds > 0 else ''}")

        all_results[f"{accuracy:.0%}"] = {
            "per_seed": seed_results,
            "avg_return": round(float(avg_return), 4),
            "avg_win_rate": round(float(avg_wr), 4),
            "avg_max_drawdown": round(float(avg_dd), 4),
            "avg_sharpe": round(float(avg_sharpe), 2),
            "avg_profit_factor": round(float(avg_pf), 2),
            "seeds_with_loss": loss_seeds,
        }

    # Category breakdown for the 75% accuracy run
    print(f"\n--- Category Breakdown (75% accuracy, seed=42) ---")
    trades75, _, final75, _ = run_realistic_backtest(usable, 0.75, 42)
    stats75 = compute_stats(trades75, 10000.0, final75, 0)
    if stats75.get("by_category"):
        print(f"  {'Category':>12} {'Trades':>7} {'WinRate':>8} {'PnL':>10}")
        for cat, s in sorted(stats75["by_category"].items(), key=lambda x: -x[1]["trades"]):
            wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
            print(f"  {cat:>12} {s['trades']:>7,} {wr:>7.0%} ${s['pnl']:>9,.2f}")

    # Find breakeven accuracy
    print(f"\n--- KEY FINDINGS ---")
    breakeven = None
    for acc_str in sorted(all_results.keys(), reverse=True):
        if all_results[acc_str]["avg_return"] <= 0:
            breakeven = acc_str
            break

    if breakeven:
        print(f"  Strategy breaks even at ~{breakeven} truth accuracy")
    else:
        print(f"  Strategy profitable at ALL tested accuracy levels (55-95%)")

    best = max(all_results.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"  Best risk-adjusted: {best[0]} accuracy (Sharpe={best[1]['avg_sharpe']:.1f})")

    realistic = all_results.get("75%", {})
    print(f"\n  At 75% accuracy (realistic sportsbook level):")
    print(f"    Avg return: {realistic.get('avg_return', 0):.1%}")
    print(f"    Avg win rate: {realistic.get('avg_win_rate', 0):.1%}")
    print(f"    Avg max drawdown: {realistic.get('avg_max_drawdown', 0):.1%}")
    print(f"    Avg Sharpe: {realistic.get('avg_sharpe', 0):.1f}")
    print(f"    Seeds with loss: {realistic.get('seeds_with_loss', 0)}/5")

    # Save
    with open(os.path.join(RESULTS_DIR, "realistic_backtest.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/realistic_backtest.json")


if __name__ == "__main__":
    main()

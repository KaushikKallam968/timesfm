"""Phase 1: Simple Replay Backtest (no optimization).

Event-driven replay through historical data in chronological order.
Uses quarter-Kelly sizing with fixed parameters.

Usage: python -m bot.backtest.proper_backtest
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories, load_sportsbook_matched

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed parameters (NOT optimized)
PARAMS = {
    "min_edge_threshold": 0.08,
    "kelly_multiplier": 0.25,
    "max_trade_size": 100.0,
    "starting_bankroll": 10000.0,
    "fee_rate": 0.02,
    "max_positions": 20,
}


def kelly_fraction_binary(edge, market_price):
    """Kelly criterion for binary outcome markets.

    edge = truth_prob - market_price (for YES bets)
    payout = (1 / market_price) - 1 (what you get per dollar if YES wins)
    """
    if market_price <= 0 or market_price >= 1:
        return 0
    payout = (1.0 / market_price) - 1.0
    truth_prob = market_price + edge
    truth_prob = max(0.01, min(0.99, truth_prob))

    # Kelly: f* = (p * (b+1) - 1) / b where p=truth_prob, b=payout
    f = (truth_prob * (payout + 1) - 1) / payout
    return max(0, f)


def run_backtest(records, params=None):
    """Run the full replay backtest.

    Approach: group records by market. For each market, find the best
    entry point (highest edge within the tradeable window), then settle
    at the known outcome. This models buying a YES/NO share and holding
    to resolution.
    """
    if params is None:
        params = PARAMS

    min_edge = params["min_edge_threshold"]
    kelly_mult = params["kelly_multiplier"]
    max_size = params["max_trade_size"]
    bankroll = params["starting_bankroll"]
    fee_rate = params["fee_rate"]

    # Group records by market_id
    by_market = defaultdict(list)
    for r in records:
        mid = r.get("market_id", "")
        if mid:
            by_market[mid].append(r)

    # For each market: find the best entry point, compute trade
    trades = []
    equity_curve = [{"timestamp": "", "bankroll": bankroll}]
    peak_bankroll = bankroll
    max_drawdown = 0

    for mid, market_records in by_market.items():
        # Sort by timestamp
        market_records.sort(key=lambda r: r.get("timestamp", ""))

        # Get the outcome (same for all records in this market)
        truth = market_records[0].get("truth_probability", 0)
        if truth not in (0.0, 1.0):
            continue  # Skip markets without clear binary outcome

        # Find the record with the best edge
        best = None
        best_edge = 0
        for r in market_records:
            mp = r.get("market_price", 0)
            if mp <= 0.02 or mp >= 0.98:
                continue

            if truth == 1.0:
                edge = 1.0 - mp  # YES is underpriced
                if edge > best_edge:
                    best_edge = edge
                    best = r
            else:
                edge = mp  # NO is underpriced (YES is overpriced)
                if edge > best_edge:
                    best_edge = edge
                    best = r

        if best is None or best_edge < min_edge:
            continue

        mp = best["market_price"]
        ts = best.get("timestamp", "")
        category = best.get("category", "")
        days_to_res = best.get("days_to_resolution")

        # Determine side and sizing
        if truth == 1.0:
            side = "YES"
            entry_price = mp
        else:
            side = "NO"
            entry_price = 1.0 - mp

        kelly_f = kelly_fraction_binary(best_edge, entry_price)
        position_size = bankroll * kelly_f * kelly_mult
        position_size = min(position_size, max_size)
        position_size = min(position_size, bankroll * 0.05)  # max 5% per trade
        position_size = max(position_size, 0)

        if position_size < 1 or bankroll < position_size:
            continue

        fees = position_size * fee_rate

        # Execute: buy at entry_price, settle at 1.0 (we picked the winning side)
        payout = position_size / entry_price  # shares bought
        pnl = payout - position_size - fees  # profit = (shares * $1) - cost - fees

        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        trades.append({
            "market_id": mid,
            "question": best.get("question", "")[:100],
            "side": side,
            "entry_price": round(entry_price, 4),
            "size": round(position_size, 2),
            "fees": round(fees, 2),
            "edge": round(best_edge, 4),
            "pnl": round(pnl, 2),
            "entry_timestamp": ts,
            "category": category,
            "days_to_resolution": days_to_res,
            "bankroll_after": round(bankroll, 2),
        })

        equity_curve.append({"timestamp": ts, "bankroll": round(bankroll, 2)})

    equity_curve.append({"timestamp": "end", "bankroll": round(bankroll, 2)})
    return trades, equity_curve, bankroll, max_drawdown


def compute_metrics(trades, starting_bankroll, final_bankroll, max_drawdown):
    """Compute backtest performance metrics."""
    if not trades:
        return {"error": "no trades"}

    pnls = [t["pnl"] for t in trades if "pnl" in t]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (final_bankroll - starting_bankroll) / starting_bankroll
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    avg_edge = np.mean([t["edge"] for t in trades if "edge" in t])

    # Sharpe-equivalent (daily PnL based)
    daily_pnls = defaultdict(float)
    for t in trades:
        ts = t.get("entry_timestamp", "")[:10]
        if ts:
            daily_pnls[ts] += t.get("pnl", 0)
    daily_vals = list(daily_pnls.values())
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(365)) if daily_vals and np.std(daily_vals) > 0 else 0

    # By category
    cat_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0})
    for t in trades:
        cat = t.get("category", "unknown")
        cat_stats[cat]["trades"] += 1
        if t.get("pnl", 0) > 0:
            cat_stats[cat]["wins"] += 1
        cat_stats[cat]["total_pnl"] += t.get("pnl", 0)

    return {
        "total_trades": len(trades),
        "total_return": round(total_return, 4),
        "total_return_pct": f"{total_return:.1%}",
        "final_bankroll": round(final_bankroll, 2),
        "win_rate": round(win_rate, 4),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(float(profit_factor), 2),
        "sharpe_equivalent": round(float(sharpe), 2),
        "max_drawdown": round(max_drawdown, 4),
        "avg_edge_at_entry": round(float(avg_edge), 4),
        "by_category": {k: dict(v) for k, v in cat_stats.items()},
    }


def run_backtest_with_accuracy(records, truth_accuracy, params=None):
    """Run backtest where the truth source is correct only X% of the time.

    Simulates a real (imperfect) truth source like sportsbook odds.
    When the truth source is wrong, we pick the LOSING side.
    """
    import random
    random.seed(42)

    if params is None:
        params = PARAMS

    min_edge = params["min_edge_threshold"]
    kelly_mult = params["kelly_multiplier"]
    max_size = params["max_trade_size"]
    bankroll = params["starting_bankroll"]
    fee_rate = params["fee_rate"]

    by_market = defaultdict(list)
    for r in records:
        mid = r.get("market_id", "")
        if mid:
            by_market[mid].append(r)

    trades = []
    peak_bankroll = bankroll
    max_drawdown = 0

    for mid, market_records in by_market.items():
        market_records.sort(key=lambda r: r.get("timestamp", ""))
        truth = market_records[0].get("truth_probability", 0)
        if truth not in (0.0, 1.0):
            continue

        # Simulate imperfect truth: correct X% of the time
        truth_source_correct = random.random() < truth_accuracy
        perceived_truth = truth if truth_source_correct else (1.0 - truth)

        best = None
        best_edge = 0
        for r in market_records:
            mp = r.get("market_price", 0)
            if mp <= 0.02 or mp >= 0.98:
                continue
            if perceived_truth == 1.0:
                edge = 1.0 - mp
            else:
                edge = mp
            if edge > best_edge:
                best_edge = edge
                best = r

        if best is None or best_edge < min_edge:
            continue

        mp = best["market_price"]
        if perceived_truth == 1.0:
            side = "YES"
            entry_price = mp
        else:
            side = "NO"
            entry_price = 1.0 - mp

        kelly_f = kelly_fraction_binary(best_edge, entry_price)
        position_size = bankroll * kelly_f * kelly_mult
        position_size = min(position_size, max_size, bankroll * 0.05)
        if position_size < 1 or bankroll < position_size:
            continue

        fees = position_size * fee_rate

        # Settle against ACTUAL truth (not perceived)
        won = (side == "YES" and truth == 1.0) or (side == "NO" and truth == 0.0)
        if won:
            payout = position_size / entry_price
            pnl = payout - position_size - fees
        else:
            pnl = -position_size - fees

        bankroll += pnl
        bankroll = max(bankroll, 0)
        peak_bankroll = max(peak_bankroll, bankroll)
        dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        trades.append({
            "market_id": mid,
            "side": side,
            "entry_price": round(entry_price, 4),
            "size": round(position_size, 2),
            "edge": round(best_edge, 4),
            "pnl": round(pnl, 2),
            "won": won,
            "truth_correct": truth_source_correct,
            "category": best.get("category", ""),
            "entry_timestamp": best.get("timestamp", ""),
            "bankroll_after": round(bankroll, 2),
        })

        if bankroll <= 0:
            break

    return trades, bankroll, max_drawdown


def main():
    print("=" * 60)
    print("PHASE 1: SIMPLE REPLAY BACKTEST")
    print("=" * 60)

    print(f"\nParameters (FIXED, not optimized):")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    print("\nLoading data...")
    records = load_price_histories()
    print(f"  Records: {len(records):,}")

    usable = [r for r in records
              if 0.01 < r.get("market_price", 0) < 0.99
              and r.get("truth_probability") is not None
              and r.get("timestamp")]
    print(f"  Usable: {len(usable):,}")

    # --- Hindsight (perfect truth) backtest ---
    print("\n--- HINDSIGHT BACKTEST (100% accurate truth) ---")
    trades, equity_curve, final_bankroll, max_dd = run_backtest(usable)
    metrics = compute_metrics(trades, PARAMS["starting_bankroll"], final_bankroll, max_dd)
    print(f"  Trades: {metrics.get('total_trades', 0):,}")
    print(f"  Return: {metrics.get('total_return_pct', 'N/A')}")
    print(f"  Win rate: {metrics.get('win_rate', 0):.1%}")
    print(f"  This is the CEILING. No real system achieves 100% truth accuracy.")

    # --- Degraded accuracy backtests ---
    print(f"\n--- REALISTIC ACCURACY BACKTESTS ---")
    print(f"  {'Accuracy':>10} {'Trades':>7} {'Return':>10} {'Win Rate':>10} {'Max DD':>8} {'Final $':>12}")

    accuracy_results = {}
    for accuracy in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
        trades_a, final_a, dd_a = run_backtest_with_accuracy(usable, accuracy)
        pnls = [t["pnl"] for t in trades_a]
        wins = sum(1 for p in pnls if p > 0)
        total_return = (final_a - PARAMS["starting_bankroll"]) / PARAMS["starting_bankroll"]
        wr = wins / len(pnls) if pnls else 0

        print(f"  {accuracy:>9.0%} {len(trades_a):>7,} {total_return:>9.1%} {wr:>9.1%} {dd_a:>7.1%} ${final_a:>10,.2f}")
        accuracy_results[f"{accuracy:.0%}"] = {
            "trades": len(trades_a),
            "total_return": round(total_return, 4),
            "win_rate": round(wr, 4),
            "max_drawdown": round(dd_a, 4),
            "final_bankroll": round(final_a, 2),
        }

    # Save all results
    result = {
        "params": PARAMS,
        "hindsight": metrics,
        "accuracy_sweep": accuracy_results,
        "equity_curve": equity_curve,
    }
    with open(os.path.join(RESULTS_DIR, "backtest_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to results/backtest_result.json")

    # Key insight
    print(f"\n--- KEY INSIGHT ---")
    breakeven = None
    for acc_str, res in sorted(accuracy_results.items(), reverse=True):
        if res["total_return"] <= 0:
            breakeven = acc_str
            break
    if breakeven:
        print(f"  Strategy breaks even at ~{breakeven} truth accuracy.")
        print(f"  Sportsbooks are typically 70-85% accurate.")
    else:
        print(f"  Strategy is profitable at all tested accuracy levels.")


if __name__ == "__main__":
    main()

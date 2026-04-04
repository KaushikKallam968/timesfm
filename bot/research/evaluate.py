"""Fixed evaluation harness for autoresearch. READ-ONLY after creation."""

import random
import math
import sys
import os
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtest", "data"))
from loader import load_price_histories
from bot.research.strategy import (
    STARTING_BANKROLL, FEE_RATE, SLIPPAGE, DIR_TRUTH_ACCURACY,
    should_trade_market, compute_edge, kelly_size,
)


def simulate_truth(actual_outcome, accuracy, days_to_res):
    """Simulate a noisy truth source. Accuracy improves closer to resolution."""
    if days_to_res <= 1:
        adj_acc = min(accuracy + 0.10, 0.98)
    elif days_to_res <= 3:
        adj_acc = min(accuracy + 0.05, 0.95)
    elif days_to_res <= 7:
        adj_acc = accuracy
    elif days_to_res <= 30:
        adj_acc = max(accuracy - 0.05, 0.55)
    else:
        adj_acc = max(accuracy - 0.10, 0.52)

    is_yes = actual_outcome == "Yes" or actual_outcome == 1.0
    is_correct = random.random() < adj_acc

    if is_correct:
        if is_yes:
            return random.uniform(0.55, 0.95)
        else:
            return random.uniform(0.05, 0.45)
    else:
        if is_yes:
            return random.uniform(0.05, 0.45)
        else:
            return random.uniform(0.55, 0.95)


def run_evaluation(seed=42):
    """Run full backtest evaluation for a single seed."""
    random.seed(seed)
    np.random.seed(seed)

    data = load_price_histories()

    # Group by market_id, pick one representative record per market
    markets = {}
    for rec in data:
        mid = rec["market_id"]
        mapped = {
            "price": rec["market_price"],
            "volume": rec.get("volume", 0),
            "category": rec.get("category", ""),
            "days_to_resolution": rec.get("days_to_resolution", float("inf")),
            "actual_outcome": rec.get("actual_outcome", "No"),
            "truth_probability": rec.get("truth_probability", 0.0),
            "market_id": mid,
            "timestamp": rec.get("timestamp", ""),
        }
        if mid not in markets:
            markets[mid] = mapped
        else:
            # keep the record closest to resolution (smallest days)
            if mapped["days_to_resolution"] < markets[mid]["days_to_resolution"]:
                markets[mid] = mapped

    # Filter with should_trade_market
    eligible = [m for m in markets.values() if should_trade_market(m)]

    bankroll = STARTING_BANKROLL
    max_bankroll = bankroll
    max_drawdown = 0.0
    trades = []

    for market in eligible:
        actual = market["actual_outcome"]
        days = market["days_to_resolution"]

        truth_prob = simulate_truth(actual, DIR_TRUTH_ACCURACY, days)

        side, edge = compute_edge(market, truth_prob)
        if side is None or edge <= 0:
            continue

        if side == "YES":
            entry_price = market["price"] + SLIPPAGE
        else:
            entry_price = (1 - market["price"]) + SLIPPAGE

        entry_price = max(0.01, min(entry_price, 0.99))

        size = kelly_size(edge, entry_price, bankroll)
        if size <= 0:
            continue

        # Determine win/loss
        is_yes_outcome = actual == "Yes" or market["truth_probability"] == 1.0
        won = (side == "YES" and is_yes_outcome) or (side == "NO" and not is_yes_outcome)

        if won:
            payout = size * ((1 - entry_price) / entry_price)
            pnl = payout - (size * FEE_RATE)
        else:
            pnl = -size - (size * FEE_RATE)

        bankroll += pnl
        if bankroll > max_bankroll:
            max_bankroll = bankroll
        dd = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        trades.append({
            "market_id": market["market_id"],
            "side": side,
            "edge": edge,
            "size": size,
            "entry_price": entry_price,
            "won": won,
            "pnl": pnl,
            "bankroll_after": bankroll,
            "timestamp": market["timestamp"],
        })

        if bankroll <= 0:
            break

    return trades, bankroll, max_drawdown


def compute_sharpe(trades):
    """Compute annualized Sharpe from trade PnLs grouped by day."""
    if not trades:
        return 0.0

    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["timestamp"]] += t["pnl"]

    pnls = list(daily_pnl.values())
    if len(pnls) < 2:
        return 0.0

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1)
    if std_pnl == 0:
        return 0.0

    return (mean_pnl / std_pnl) * math.sqrt(365)


def main():
    seeds = [42, 123, 456, 789, 1337]
    all_win_rates = []
    all_sharpes = []
    all_returns = []
    all_drawdowns = []
    all_num_trades = []
    seeds_profitable = 0
    profit_factor = 0.0

    for i, seed in enumerate(seeds):
        trades, final_bankroll, max_dd = run_evaluation(seed)

        num_trades = len(trades)
        wins = sum(1 for t in trades if t["won"])
        wr = wins / num_trades if num_trades > 0 else 0.0
        total_return = (final_bankroll - STARTING_BANKROLL) / STARTING_BANKROLL
        sharpe = compute_sharpe(trades)

        all_win_rates.append(wr)
        all_sharpes.append(sharpe)
        all_returns.append(total_return)
        all_drawdowns.append(max_dd)
        all_num_trades.append(num_trades)

        if final_bankroll > STARTING_BANKROLL:
            seeds_profitable += 1

        # Compute profit_factor from seed=42 only
        if seed == 42:
            gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gross_loss = sum(t["pnl"] for t in trades if t["pnl"] < 0)
            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    avg_wr = np.mean(all_win_rates)
    avg_sharpe = np.mean(all_sharpes)
    avg_return = np.mean(all_returns)
    avg_dd = np.mean(all_drawdowns)
    avg_trades = np.mean(all_num_trades)
    worst_wr = min(all_win_rates)
    gate = "YES" if avg_wr >= 0.95 else "NO"

    print("---")
    print(f"win_rate:          {avg_wr:.6f}")
    print(f"sharpe:            {avg_sharpe:.6f}")
    print(f"total_return:      {avg_return:.6f}")
    print(f"max_drawdown:      {avg_dd:.6f}")
    print(f"num_trades:        {avg_trades:.1f}")
    print(f"profit_factor:     {profit_factor:.6f}")
    print(f"worst_seed_wr:     {worst_wr:.6f}")
    print(f"seeds_profitable:  {seeds_profitable}")
    print(f"gate_passed:       {gate}")


if __name__ == "__main__":
    main()

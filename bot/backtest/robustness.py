"""Robustness tests for prediction market backtest.

Three tests:
A. Bootstrap resampling (1000 iterations) — trade-level PnL distribution
B. Monte Carlo parameter perturbation (500 iterations) — return distribution under param noise
C. Transaction cost sensitivity — find breakeven fee rate

Usage: python -m bot.backtest.robustness
"""

import json
import os
import sys
import random
import numpy as np

from bot.backtest.proper_backtest import run_backtest, compute_metrics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS = {
    "min_edge_threshold": 0.08,
    "kelly_multiplier": 0.25,
    "max_trade_size": 100.0,
    "starting_bankroll": 10000.0,
    "fee_rate": 0.02,
    "max_positions": 20,
}

BOOTSTRAP_ITERATIONS = 1000
MONTE_CARLO_ITERATIONS = 500
FEE_RATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]


def run_bootstrap(trades, n_iterations=BOOTSTRAP_ITERATIONS):
    """Resample trades with replacement, compute PnL distribution."""
    if not trades:
        return None

    pnls = [t["pnl"] for t in trades if "pnl" in t]
    if not pnls:
        return None

    n_trades = len(pnls)
    resampled_returns = []

    for _ in range(n_iterations):
        sample = random.choices(pnls, k=n_trades)
        total_pnl = sum(sample)
        resampled_returns.append(total_pnl)

    resampled_returns = np.array(resampled_returns)

    return {
        "iterations": n_iterations,
        "original_trades": n_trades,
        "original_total_pnl": round(sum(pnls), 2),
        "p5": round(float(np.percentile(resampled_returns, 5)), 2),
        "p25": round(float(np.percentile(resampled_returns, 25)), 2),
        "median": round(float(np.median(resampled_returns)), 2),
        "p75": round(float(np.percentile(resampled_returns, 75)), 2),
        "p95": round(float(np.percentile(resampled_returns, 95)), 2),
        "mean": round(float(np.mean(resampled_returns)), 2),
        "std": round(float(np.std(resampled_returns)), 2),
        "prob_loss": round(float(np.mean(resampled_returns < 0)), 4),
        "prob_loss_pct": f"{float(np.mean(resampled_returns < 0)):.1%}",
    }


def run_monte_carlo(records, base_params, n_iterations=MONTE_CARLO_ITERATIONS):
    """Randomly perturb params by +/-20%, run full backtest each time."""
    returns = []
    trade_counts = []
    max_drawdowns = []
    win_rates = []
    param_log = []

    for i in range(n_iterations):
        perturbed = {**base_params}

        # Perturb min_edge_threshold, kelly_multiplier, max_trade_size by +/-20%
        for key in ["min_edge_threshold", "kelly_multiplier", "max_trade_size"]:
            factor = 1.0 + random.uniform(-0.20, 0.20)
            perturbed[key] = base_params[key] * factor

        trades, _, final_bankroll, max_dd = run_backtest(records, perturbed)

        total_return = (final_bankroll - base_params["starting_bankroll"]) / base_params["starting_bankroll"]
        returns.append(total_return)
        trade_counts.append(len(trades))
        max_drawdowns.append(max_dd)

        if trades:
            pnls = [t["pnl"] for t in trades if "pnl" in t]
            wins = [p for p in pnls if p > 0]
            wr = len(wins) / len(pnls) if pnls else 0
            win_rates.append(wr)
        else:
            win_rates.append(0)

        if i < 10:  # Log first 10 for inspection
            param_log.append({
                "min_edge_threshold": round(perturbed["min_edge_threshold"], 4),
                "kelly_multiplier": round(perturbed["kelly_multiplier"], 4),
                "max_trade_size": round(perturbed["max_trade_size"], 2),
                "total_return": round(total_return, 4),
                "trades": len(trades),
            })

    returns = np.array(returns)
    trade_counts = np.array(trade_counts)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "iterations": n_iterations,
        "perturbation_range": "+/-20%",
        "perturbed_params": ["min_edge_threshold", "kelly_multiplier", "max_trade_size"],
        "return_distribution": {
            "p5": round(float(np.percentile(returns, 5)), 4),
            "p25": round(float(np.percentile(returns, 25)), 4),
            "median": round(float(np.median(returns)), 4),
            "p75": round(float(np.percentile(returns, 75)), 4),
            "p95": round(float(np.percentile(returns, 95)), 4),
            "mean": round(float(np.mean(returns)), 4),
            "std": round(float(np.std(returns)), 4),
            "prob_loss": round(float(np.mean(returns < 0)), 4),
            "prob_loss_pct": f"{float(np.mean(returns < 0)):.1%}",
        },
        "trade_count_distribution": {
            "min": int(np.min(trade_counts)),
            "median": int(np.median(trade_counts)),
            "max": int(np.max(trade_counts)),
        },
        "max_drawdown_distribution": {
            "p5": round(float(np.percentile(max_drawdowns, 5)), 4),
            "median": round(float(np.median(max_drawdowns)), 4),
            "p95": round(float(np.percentile(max_drawdowns, 95)), 4),
        },
        "sample_runs": param_log,
    }


def run_fee_sensitivity(records, base_params, fee_rates=FEE_RATES):
    """Run backtest across different fee rates, find breakeven."""
    results = []
    breakeven_fee = None
    prev_return = None

    for fee in fee_rates:
        params = {**base_params, "fee_rate": fee}
        trades, _, final_bankroll, max_dd = run_backtest(records, params)
        total_return = (final_bankroll - base_params["starting_bankroll"]) / base_params["starting_bankroll"]

        results.append({
            "fee_rate": fee,
            "fee_rate_pct": f"{fee:.1%}",
            "total_return": round(total_return, 4),
            "total_return_pct": f"{total_return:.1%}",
            "total_trades": len(trades),
            "final_bankroll": round(final_bankroll, 2),
        })

        # Find breakeven via linear interpolation
        if prev_return is not None and prev_return > 0 and total_return <= 0:
            prev_fee = fee_rates[fee_rates.index(fee) - 1]
            # Linear interpolation: find fee where return = 0
            breakeven_fee = prev_fee + (fee - prev_fee) * (prev_return / (prev_return - total_return))
            breakeven_fee = round(breakeven_fee, 4)

        prev_return = total_return

    # If all returns are positive, breakeven is above our max tested fee
    if breakeven_fee is None and all(r["total_return"] > 0 for r in results):
        breakeven_fee = f">{fee_rates[-1]}"
    elif breakeven_fee is None and all(r["total_return"] <= 0 for r in results):
        breakeven_fee = f"<{fee_rates[0]}"

    return {
        "fee_rates_tested": fee_rates,
        "results": results,
        "breakeven_fee_rate": breakeven_fee,
    }


def main():
    print("=" * 60)
    print("ROBUSTNESS TESTS")
    print("=" * 60)

    print(f"\nBase parameters:")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    print("\nLoading data...")
    records = load_price_histories()
    usable = [r for r in records
              if 0.01 < r.get("market_price", 0) < 0.99
              and r.get("truth_probability") is not None
              and r.get("timestamp")]
    print(f"  Total records: {len(records):,}")
    print(f"  Usable records: {len(usable):,}")

    # Baseline run
    print("\nRunning baseline backtest...")
    baseline_trades, _, baseline_final, baseline_max_dd = run_backtest(usable, PARAMS)
    baseline_metrics = compute_metrics(baseline_trades, PARAMS["starting_bankroll"], baseline_final, baseline_max_dd)
    print(f"  Baseline: {baseline_metrics.get('total_return_pct', 'N/A')} return, "
          f"{baseline_metrics.get('total_trades', 0)} trades")

    # A. Bootstrap
    print(f"\n{'=' * 60}")
    print(f"A. BOOTSTRAP RESAMPLING ({BOOTSTRAP_ITERATIONS} iterations)")
    print(f"{'=' * 60}")
    bootstrap_result = run_bootstrap(baseline_trades)
    if bootstrap_result:
        print(f"  Original total PnL: ${bootstrap_result['original_total_pnl']:,.2f}")
        print(f"  Resampled PnL distribution:")
        print(f"    5th percentile:  ${bootstrap_result['p5']:>10,.2f}")
        print(f"    25th percentile: ${bootstrap_result['p25']:>10,.2f}")
        print(f"    Median:          ${bootstrap_result['median']:>10,.2f}")
        print(f"    75th percentile: ${bootstrap_result['p75']:>10,.2f}")
        print(f"    95th percentile: ${bootstrap_result['p95']:>10,.2f}")
        print(f"  Mean: ${bootstrap_result['mean']:,.2f}  Std: ${bootstrap_result['std']:,.2f}")
        print(f"  Probability of loss: {bootstrap_result['prob_loss_pct']}")
    else:
        print("  No trades to bootstrap")

    # B. Monte Carlo
    print(f"\n{'=' * 60}")
    print(f"B. MONTE CARLO PARAMETER PERTURBATION ({MONTE_CARLO_ITERATIONS} iterations)")
    print(f"{'=' * 60}")
    print(f"  Perturbing: min_edge_threshold, kelly_multiplier, max_trade_size by +/-20%")
    mc_result = run_monte_carlo(usable, PARAMS)
    rd = mc_result["return_distribution"]
    print(f"  Return distribution:")
    print(f"    5th percentile:  {rd['p5']:>8.2%}")
    print(f"    25th percentile: {rd['p25']:>8.2%}")
    print(f"    Median:          {rd['median']:>8.2%}")
    print(f"    75th percentile: {rd['p75']:>8.2%}")
    print(f"    95th percentile: {rd['p95']:>8.2%}")
    print(f"  Mean: {rd['mean']:.2%}  Std: {rd['std']:.2%}")
    print(f"  Probability of loss: {rd['prob_loss_pct']}")
    td = mc_result["trade_count_distribution"]
    print(f"  Trade count: min={td['min']}, median={td['median']}, max={td['max']}")
    dd = mc_result["max_drawdown_distribution"]
    print(f"  Max drawdown: p5={dd['p5']:.2%}, median={dd['median']:.2%}, p95={dd['p95']:.2%}")

    # C. Fee Sensitivity
    print(f"\n{'=' * 60}")
    print(f"C. TRANSACTION COST SENSITIVITY")
    print(f"{'=' * 60}")
    fee_result = run_fee_sensitivity(usable, PARAMS)
    print(f"  {'Fee Rate':>10} {'Return':>10} {'Trades':>8} {'Final Bankroll':>16}")
    print(f"  {'-' * 48}")
    for r in fee_result["results"]:
        print(f"  {r['fee_rate_pct']:>10} {r['total_return_pct']:>10} {r['total_trades']:>8} "
              f"${r['final_bankroll']:>14,.2f}")
    print(f"\n  Breakeven fee rate: {fee_result['breakeven_fee_rate']}")

    # Overall summary
    print(f"\n{'=' * 60}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'=' * 60}")

    if bootstrap_result:
        print(f"\n  Bootstrap:")
        if bootstrap_result["prob_loss"] < 0.05:
            print(f"    STRONG — {bootstrap_result['prob_loss_pct']} chance of loss")
        elif bootstrap_result["prob_loss"] < 0.20:
            print(f"    MODERATE — {bootstrap_result['prob_loss_pct']} chance of loss")
        else:
            print(f"    WEAK — {bootstrap_result['prob_loss_pct']} chance of loss")

    print(f"\n  Monte Carlo:")
    if rd["prob_loss"] < 0.05:
        print(f"    ROBUST — {rd['prob_loss_pct']} loss probability under param perturbation")
    elif rd["prob_loss"] < 0.20:
        print(f"    MODERATE — {rd['prob_loss_pct']} loss probability under param perturbation")
    else:
        print(f"    FRAGILE — {rd['prob_loss_pct']} loss probability under param perturbation")

    print(f"\n  Fee sensitivity:")
    bfr = fee_result["breakeven_fee_rate"]
    if isinstance(bfr, str) and bfr.startswith(">"):
        print(f"    ROBUST — profitable even at {FEE_RATES[-1]:.0%} fees")
    elif isinstance(bfr, (int, float)) and bfr > 0.03:
        print(f"    GOOD — breakeven at {bfr:.2%} fees")
    elif isinstance(bfr, (int, float)):
        print(f"    THIN MARGIN — breakeven at {bfr:.2%} fees")
    else:
        print(f"    UNPROFITABLE — negative return at all tested fee rates")

    # Save results
    result = {
        "base_params": PARAMS,
        "baseline_metrics": baseline_metrics,
        "bootstrap": bootstrap_result,
        "monte_carlo": mc_result,
        "fee_sensitivity": fee_result,
    }

    output_path = os.path.join(RESULTS_DIR, "robustness.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

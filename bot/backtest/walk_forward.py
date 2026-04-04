"""Walk-forward validation for prediction market backtest.

Splits data into rolling 6-month train / 1-month test windows.
For each window, sweeps min_edge_threshold on training data,
applies best params to test period. Compares IS vs OOS performance.
Also runs fixed-params baseline.

Usage: python -m bot.backtest.walk_forward
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from datetime import datetime

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

EDGE_SWEEP = [round(0.03 + i * 0.01, 2) for i in range(18)]  # 0.03 to 0.20


def split_into_months(records):
    """Group records by year-month, return sorted list of (month_key, records)."""
    by_month = defaultdict(list)
    for r in records:
        ts = r.get("timestamp", "")
        if not ts:
            continue
        month_key = ts[:7]  # "YYYY-MM"
        by_month[month_key].append(r)

    return sorted(by_month.items(), key=lambda x: x[0])


def sweep_edge_threshold(train_records, base_params):
    """Sweep min_edge_threshold on training data, return best value by total return."""
    best_edge = base_params["min_edge_threshold"]
    best_return = -float("inf")
    results = []

    for edge in EDGE_SWEEP:
        params = {**base_params, "min_edge_threshold": edge}
        trades, _, final_bankroll, max_dd = run_backtest(train_records, params)

        if not trades:
            results.append({"edge": edge, "total_return": 0, "trades": 0})
            continue

        total_return = (final_bankroll - params["starting_bankroll"]) / params["starting_bankroll"]
        results.append({"edge": edge, "total_return": round(total_return, 4), "trades": len(trades)})

        if total_return > best_return:
            best_return = total_return
            best_edge = edge

    return best_edge, best_return, results


def run_walk_forward(records):
    """Run walk-forward validation with 6-month train / 1-month test windows."""
    monthly = split_into_months(records)
    month_keys = [m[0] for m in monthly]
    month_data = {m[0]: m[1] for m in monthly}

    if len(month_keys) < 7:
        print(f"Not enough months ({len(month_keys)}) for walk-forward. Need at least 7.")
        return []

    windows = []
    train_size = 6
    test_size = 1

    for i in range(train_size, len(month_keys) - test_size + 1):
        train_months = month_keys[i - train_size:i]
        test_months = month_keys[i:i + test_size]

        train_records = []
        for m in train_months:
            train_records.extend(month_data[m])

        test_records = []
        for m in test_months:
            test_records.extend(month_data[m])

        if not train_records or not test_records:
            continue

        # Sweep on training data
        best_edge, best_is_return, sweep_results = sweep_edge_threshold(train_records, PARAMS)

        # Apply best params to test period
        best_params = {**PARAMS, "min_edge_threshold": best_edge}
        test_trades, _, test_final, test_max_dd = run_backtest(test_records, best_params)
        test_metrics = compute_metrics(test_trades, PARAMS["starting_bankroll"], test_final, test_max_dd)

        # Also run fixed params on test period for baseline
        fixed_trades, _, fixed_final, fixed_max_dd = run_backtest(test_records, PARAMS)
        fixed_metrics = compute_metrics(fixed_trades, PARAMS["starting_bankroll"], fixed_final, fixed_max_dd)

        # IS metrics with best edge on training data
        is_trades, _, is_final, is_max_dd = run_backtest(train_records, best_params)
        is_metrics = compute_metrics(is_trades, PARAMS["starting_bankroll"], is_final, is_max_dd)

        window_result = {
            "train_months": train_months,
            "test_months": test_months,
            "best_edge_threshold": best_edge,
            "in_sample": {
                "total_return": is_metrics.get("total_return", 0),
                "total_return_pct": is_metrics.get("total_return_pct", "N/A"),
                "total_trades": is_metrics.get("total_trades", 0),
                "win_rate": is_metrics.get("win_rate", 0),
                "sharpe_equivalent": is_metrics.get("sharpe_equivalent", 0),
                "max_drawdown": is_metrics.get("max_drawdown", 0),
            },
            "out_of_sample": {
                "total_return": test_metrics.get("total_return", 0),
                "total_return_pct": test_metrics.get("total_return_pct", "N/A"),
                "total_trades": test_metrics.get("total_trades", 0),
                "win_rate": test_metrics.get("win_rate", 0),
                "sharpe_equivalent": test_metrics.get("sharpe_equivalent", 0),
                "max_drawdown": test_metrics.get("max_drawdown", 0),
            },
            "fixed_baseline_oos": {
                "total_return": fixed_metrics.get("total_return", 0),
                "total_return_pct": fixed_metrics.get("total_return_pct", "N/A"),
                "total_trades": fixed_metrics.get("total_trades", 0),
                "win_rate": fixed_metrics.get("win_rate", 0),
            },
        }
        windows.append(window_result)

    return windows


def main():
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    print(f"\nBase parameters:")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")
    print(f"\nEdge sweep range: {EDGE_SWEEP[0]} to {EDGE_SWEEP[-1]} (step 0.01)")
    print(f"Training window: 6 months, Test window: 1 month, Roll: 1 month")

    print("\nLoading data...")
    records = load_price_histories()
    usable = [r for r in records
              if 0.01 < r.get("market_price", 0) < 0.99
              and r.get("truth_probability") is not None
              and r.get("timestamp")]
    print(f"  Total records: {len(records):,}")
    print(f"  Usable records: {len(usable):,}")

    # Sort by timestamp
    usable = sorted(usable, key=lambda r: r.get("timestamp", ""))

    monthly = split_into_months(usable)
    print(f"  Months spanned: {len(monthly)} ({monthly[0][0]} to {monthly[-1][0]})")

    print("\nRunning walk-forward analysis...")
    windows = run_walk_forward(usable)

    if not windows:
        print("No windows produced. Not enough data.")
        return

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(windows)} windows")
    print(f"{'=' * 60}")

    is_returns = []
    oos_returns = []
    baseline_returns = []

    for i, w in enumerate(windows):
        is_ret = w["in_sample"]["total_return"]
        oos_ret = w["out_of_sample"]["total_return"]
        base_ret = w["fixed_baseline_oos"]["total_return"]
        is_returns.append(is_ret)
        oos_returns.append(oos_ret)
        baseline_returns.append(base_ret)

        print(f"\nWindow {i + 1}: Train {w['train_months'][0]}..{w['train_months'][-1]} | "
              f"Test {w['test_months'][0]}")
        print(f"  Best edge: {w['best_edge_threshold']:.2f}")
        print(f"  IS return: {w['in_sample']['total_return_pct']:>8} "
              f"({w['in_sample']['total_trades']} trades, "
              f"WR {w['in_sample']['win_rate']:.0%})")
        print(f"  OOS return: {w['out_of_sample']['total_return_pct']:>8} "
              f"({w['out_of_sample']['total_trades']} trades, "
              f"WR {w['out_of_sample']['win_rate']:.0%})")
        print(f"  Baseline OOS: {w['fixed_baseline_oos']['total_return_pct']:>8} "
              f"({w['fixed_baseline_oos']['total_trades']} trades)")

    # Aggregate stats
    print(f"\n{'=' * 60}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 60}")

    is_arr = np.array(is_returns)
    oos_arr = np.array(oos_returns)
    base_arr = np.array(baseline_returns)

    print(f"\nIn-Sample returns:")
    print(f"  Mean: {np.mean(is_arr):.2%}  Median: {np.median(is_arr):.2%}  "
          f"Std: {np.std(is_arr):.2%}")
    print(f"  Min: {np.min(is_arr):.2%}  Max: {np.max(is_arr):.2%}")

    print(f"\nOut-of-Sample returns:")
    print(f"  Mean: {np.mean(oos_arr):.2%}  Median: {np.median(oos_arr):.2%}  "
          f"Std: {np.std(oos_arr):.2%}")
    print(f"  Min: {np.min(oos_arr):.2%}  Max: {np.max(oos_arr):.2%}")
    print(f"  % profitable windows: {np.mean(oos_arr > 0):.0%}")

    print(f"\nFixed-params baseline OOS:")
    print(f"  Mean: {np.mean(base_arr):.2%}  Median: {np.median(base_arr):.2%}")
    print(f"  % profitable windows: {np.mean(base_arr > 0):.0%}")

    # IS vs OOS degradation
    degradation = np.mean(is_arr) - np.mean(oos_arr)
    print(f"\nIS vs OOS degradation: {degradation:.2%}")
    if degradation > 0.05:
        print("  WARNING: Significant IS/OOS gap suggests overfitting")
    elif degradation > 0.02:
        print("  CAUTION: Moderate IS/OOS gap")
    else:
        print("  OK: IS/OOS gap is small")

    # Optimized vs baseline
    opt_vs_base = np.mean(oos_arr) - np.mean(base_arr)
    print(f"\nOptimized vs baseline OOS: {opt_vs_base:+.2%}")
    if opt_vs_base > 0:
        print("  Walk-forward optimization adds value over fixed params")
    else:
        print("  Fixed params perform as well or better (optimization not helping)")

    # Save results
    result = {
        "config": {
            "train_window_months": 6,
            "test_window_months": 1,
            "edge_sweep_range": [EDGE_SWEEP[0], EDGE_SWEEP[-1]],
            "edge_sweep_step": 0.01,
            "base_params": PARAMS,
        },
        "windows": windows,
        "aggregate": {
            "num_windows": len(windows),
            "is_mean_return": round(float(np.mean(is_arr)), 4),
            "is_median_return": round(float(np.median(is_arr)), 4),
            "oos_mean_return": round(float(np.mean(oos_arr)), 4),
            "oos_median_return": round(float(np.median(oos_arr)), 4),
            "oos_pct_profitable": round(float(np.mean(oos_arr > 0)), 4),
            "baseline_mean_return": round(float(np.mean(base_arr)), 4),
            "is_oos_degradation": round(float(degradation), 4),
            "optimized_vs_baseline": round(float(opt_vs_base), 4),
        },
    }

    output_path = os.path.join(RESULTS_DIR, "walk_forward.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

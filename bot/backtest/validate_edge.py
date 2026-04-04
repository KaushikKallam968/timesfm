"""Phase 0: Edge Validation via Brier Skill Score.

Must pass before any backtesting. Proves (or disproves) that a
tradeable edge exists in our data after accounting for fees.

Usage: python -m bot.backtest.validate_edge
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories, load_sportsbook_matched

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def brier_score(predictions, outcomes):
    """Mean squared error between predicted probabilities and binary outcomes."""
    preds = np.array(predictions, dtype=float)
    outs = np.array(outcomes, dtype=float)
    return float(np.mean((preds - outs) ** 2))


def calibration_curve(predictions, outcomes, n_bins=10):
    """Compute calibration: for each probability bucket, actual frequency."""
    preds = np.array(predictions, dtype=float)
    outs = np.array(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (preds >= bins[i]) & (preds < bins[i + 1])
        if i == n_bins - 1:
            mask = (preds >= bins[i]) & (preds <= bins[i + 1])
        if mask.sum() > 0:
            result.append({
                "bin_low": round(float(bins[i]), 2),
                "bin_high": round(float(bins[i + 1]), 2),
                "mean_predicted": round(float(preds[mask].mean()), 4),
                "actual_frequency": round(float(outs[mask].mean()), 4),
                "count": int(mask.sum()),
            })
    return result


def analyze_edges(records):
    """Compute edge statistics from price history records."""
    edges = []
    for r in records:
        mp = r.get("market_price", 0)
        truth = r.get("truth_probability", 0)
        edge = abs(truth - mp)
        edges.append(edge)
    edges = np.array(edges)
    return {
        "count": len(edges),
        "mean_edge": round(float(edges.mean()), 4),
        "median_edge": round(float(np.median(edges)), 4),
        "std_edge": round(float(edges.std()), 4),
        "pct_above_2pct": round(float((edges > 0.02).mean()), 4),
        "pct_above_5pct": round(float((edges > 0.05).mean()), 4),
        "pct_above_8pct": round(float((edges > 0.08).mean()), 4),
        "pct_above_10pct": round(float((edges > 0.10).mean()), 4),
        "pct_above_15pct": round(float((edges > 0.15).mean()), 4),
    }


def main():
    print("=" * 60)
    print("PHASE 0: EDGE VALIDATION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    price_histories = load_price_histories()
    sportsbook = load_sportsbook_matched()
    print(f"  Price histories: {len(price_histories):,}")
    print(f"  Sportsbook matched: {len(sportsbook):,}")

    report = {}

    # --- 1. Polymarket Calibration (how good is the market?) ---
    print("\n--- Polymarket Calibration ---")
    market_prices = [r["market_price"] for r in price_histories if 0 < r.get("market_price", 0) < 1]
    outcomes = [r["truth_probability"] for r in price_histories if 0 < r.get("market_price", 0) < 1]

    poly_brier = brier_score(market_prices, outcomes)
    naive_brier = brier_score([0.5] * len(outcomes), outcomes)  # Always predict 50%
    bss_vs_naive = 1 - (poly_brier / naive_brier) if naive_brier > 0 else 0

    print(f"  Polymarket Brier score: {poly_brier:.4f}")
    print(f"  Naive (always 50%) Brier: {naive_brier:.4f}")
    print(f"  Brier Skill Score vs naive: {bss_vs_naive:.4f}")

    if bss_vs_naive > 0:
        print(f"  -> Polymarket IS better than coin flip (BSS={bss_vs_naive:.3f})")
    else:
        print(f"  -> Polymarket is NOT better than coin flip!")

    cal = calibration_curve(market_prices, outcomes)
    print("\n  Calibration curve:")
    print(f"  {'Bucket':>12} {'Predicted':>10} {'Actual':>8} {'Count':>7}")
    for b in cal:
        print(f"  {b['bin_low']:.0%}-{b['bin_high']:.0%}     {b['mean_predicted']:>8.1%}   {b['actual_frequency']:>6.1%}   {b['count']:>5,}")

    report["polymarket_calibration"] = {
        "brier_score": poly_brier,
        "naive_brier": naive_brier,
        "bss_vs_naive": bss_vs_naive,
        "calibration": cal,
    }

    # --- 2. Edge by Days to Resolution ---
    print("\n--- Edge by Days to Resolution ---")
    by_days = defaultdict(list)
    for r in price_histories:
        d = r.get("days_to_resolution")
        mp = r.get("market_price", 0)
        truth = r.get("truth_probability", 0)
        if d is not None and 0 < mp < 1:
            edge = abs(truth - mp)
            if d <= 1:
                by_days["0-1 days"].append(edge)
            elif d <= 3:
                by_days["2-3 days"].append(edge)
            elif d <= 7:
                by_days["4-7 days"].append(edge)
            elif d <= 14:
                by_days["8-14 days"].append(edge)
            elif d <= 30:
                by_days["15-30 days"].append(edge)
            elif d <= 90:
                by_days["31-90 days"].append(edge)
            else:
                by_days["90+ days"].append(edge)

    print(f"  {'Period':>12} {'Avg Edge':>10} {'Median':>8} {'>8%':>6} {'Count':>7}")
    days_report = {}
    for period in ["0-1 days", "2-3 days", "4-7 days", "8-14 days", "15-30 days", "31-90 days", "90+ days"]:
        edges = by_days.get(period, [])
        if edges:
            arr = np.array(edges)
            avg = float(arr.mean())
            med = float(np.median(arr))
            pct8 = float((arr > 0.08).mean())
            print(f"  {period:>12} {avg:>9.1%} {med:>7.1%} {pct8:>5.0%} {len(edges):>7,}")
            days_report[period] = {"avg_edge": avg, "median": med, "pct_above_8pct": pct8, "count": len(edges)}
    report["edge_by_days"] = days_report

    # --- 3. Edge by Category ---
    print("\n--- Edge by Category ---")
    by_cat = defaultdict(list)
    for r in price_histories:
        cat = r.get("category", "unknown")
        mp = r.get("market_price", 0)
        truth = r.get("truth_probability", 0)
        if 0 < mp < 1:
            by_cat[cat].append(abs(truth - mp))

    print(f"  {'Category':>15} {'Avg Edge':>10} {'>8%':>6} {'Count':>7}")
    cat_report = {}
    for cat, edges in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        arr = np.array(edges)
        avg = float(arr.mean())
        pct8 = float((arr > 0.08).mean())
        print(f"  {cat:>15} {avg:>9.1%} {pct8:>5.0%} {len(edges):>7,}")
        cat_report[cat] = {"avg_edge": avg, "pct_above_8pct": pct8, "count": len(edges)}
    report["edge_by_category"] = cat_report

    # --- 4. Fee Viability Check ---
    print("\n--- Fee Viability ---")
    all_edges = [abs(r["truth_probability"] - r["market_price"])
                 for r in price_histories if 0 < r.get("market_price", 0) < 1]
    all_edges = np.array(all_edges)

    fee_levels = [0.01, 0.02, 0.03, 0.05]
    print(f"  {'Fee Rate':>10} {'Trades >fee':>12} {'Net Avg Edge':>13}")
    fee_report = {}
    for fee in fee_levels:
        profitable = (all_edges > fee).mean()
        net_edge = float((all_edges - fee)[all_edges > fee].mean()) if (all_edges > fee).any() else 0
        print(f"  {fee:>9.0%} {profitable:>11.0%} {net_edge:>12.1%}")
        fee_report[f"{fee:.0%}"] = {"profitable_pct": float(profitable), "net_avg_edge": net_edge}
    report["fee_viability"] = fee_report

    # --- 5. Overall Edge Stats ---
    print("\n--- Overall Edge Distribution ---")
    stats = analyze_edges(price_histories)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    report["edge_stats"] = stats

    # --- 6. Sportsbook Data Analysis ---
    if sportsbook:
        print("\n--- Sportsbook Data ---")
        sb_prices = [r["market_price"] for r in sportsbook if 0 < r.get("market_price", 0) < 1]
        sb_outcomes = [1.0 if r.get("actual_outcome") == "YES" else 0.0
                       for r in sportsbook if 0 < r.get("market_price", 0) < 1]
        sb_brier = brier_score(sb_prices, sb_outcomes)
        print(f"  Sportsbook Brier score: {sb_brier:.4f}")
        print(f"  Records: {len(sb_prices):,}")
        report["sportsbook"] = {"brier_score": sb_brier, "count": len(sb_prices)}

    # --- VERDICT ---
    print("\n" + "=" * 60)
    tradeable_pct = float((all_edges > 0.08).mean())
    avg_net = float((all_edges - 0.02)[all_edges > 0.08].mean()) if (all_edges > 0.08).any() else 0

    if tradeable_pct > 0.05 and avg_net > 0.02:
        verdict = "GO"
        print(f"VERDICT: {verdict}")
        print(f"  {tradeable_pct:.0%} of observations have >8% edge")
        print(f"  Average net edge (after 2% fees): {avg_net:.1%}")
        print(f"  Sufficient data for backtesting. Proceed to Phase 1.")
    else:
        verdict = "MARGINAL"
        print(f"VERDICT: {verdict}")
        print(f"  {tradeable_pct:.0%} of observations have >8% edge")
        print(f"  Average net edge (after 2% fees): {avg_net:.1%}")
        print(f"  Edge exists but thin. Proceed with caution.")

    report["verdict"] = verdict
    print("=" * 60)

    # Save report
    report_path = os.path.join(RESULTS_DIR, "edge_validation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()

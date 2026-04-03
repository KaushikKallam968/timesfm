"""Run full backtest pipeline: load data → load model → run strategies → report."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data.downloader import get_ohlcv
from backtest.strategies import run_all_strategies

import timesfm

MODEL_DIR = "/home/user/timesfm/poc/model_cache/pytorch"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
print("Loading TimesFM (zero-shot baseline)...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    MODEL_DIR, torch_compile=False, local_files_only=True
)
model.compile(timesfm.ForecastConfig(
    max_context=512, max_horizon=128,
    normalize_inputs=True, use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
))
print("Model ready.\n")

# Load data
symbols = ["BTC/USD", "ETH/USD", "SPY"]
all_results = {}

for symbol in symbols:
    df = get_ohlcv(symbol)
    prices = df["close"].values.flatten()

    results = run_all_strategies(model, prices, symbol)
    all_results[symbol] = results

# Plot equity curves
fig, axes = plt.subplots(len(symbols), 1, figsize=(14, 5 * len(symbols)))
if len(symbols) == 1:
    axes = [axes]

for idx, (symbol, results) in enumerate(all_results.items()):
    ax = axes[idx]
    for name, result in results.items():
        ax.plot(result.equity_curve.values, label=f"{name} (S={result.metrics['sharpe']:.2f})")
    ax.set_title(f"{symbol} — Strategy Comparison")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Trading Days")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/backtest_equity_curves.png", dpi=150)
print(f"\nEquity curves saved: {RESULTS_DIR}/backtest_equity_curves.png")

# Summary
print("\n" + "=" * 70)
print("BACKTEST SUMMARY")
print("=" * 70)

for symbol, results in all_results.items():
    print(f"\n{symbol}:")
    best_name = max(results, key=lambda k: results[k].metrics['sharpe'] if k != 'buy_and_hold' else -999)
    best = results[best_name]
    bh = results['buy_and_hold']

    print(f"  Best strategy: {best_name}")
    print(f"    Sharpe: {best.metrics['sharpe']:.2f} (vs B&H: {bh.metrics['sharpe']:.2f})")
    print(f"    Return: {best.metrics['total_return']:.1%} (vs B&H: {bh.metrics['total_return']:.1%})")
    print(f"    Max DD: {best.metrics['max_drawdown']:.1%}")
    print(f"    Win Rate: {best.metrics['win_rate']:.1%}")

# Decision
print("\n" + "=" * 70)
all_sharpes = []
for sym, results in all_results.items():
    for name, r in results.items():
        if name != 'buy_and_hold':
            all_sharpes.append(r.metrics['sharpe'])

avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
print(f"Average strategy Sharpe: {avg_sharpe:.2f}")

if avg_sharpe > 0.5:
    print("VERDICT: Zero-shot TimesFM shows promise. Fine-tuning likely to improve further.")
    print("  → Proceed to fine-tuning phase.")
elif avg_sharpe > 0:
    print("VERDICT: Marginal edge detected. Fine-tuning is mandatory for viability.")
    print("  → Fine-tune on more data before committing to trading bot.")
else:
    print("VERDICT: Zero-shot TimesFM is not profitable for trading (as expected).")
    print("  → Fine-tuning is the critical next step. If that fails, consider Kronos.")

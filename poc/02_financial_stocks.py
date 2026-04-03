"""POC 2: Financial/Stock Forecasting — validate TimesFM on known weak domain."""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv

import timesfm

MODEL_DIR = "/home/user/timesfm/poc/model_cache/pytorch"
RESULTS_DIR = "/home/user/timesfm/poc/results"
DATA_DIR = "/home/user/timesfm/poc/data"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load model
print("Loading TimesFM...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    MODEL_DIR, torch_compile=False, local_files_only=True
)
model.compile(timesfm.ForecastConfig(
    max_context=512, max_horizon=128,
    normalize_inputs=True, use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
))
print("Model ready.\n")


def fetch_stock(ticker, period="2y"):
    """Fetch stock data from yfinance, cache locally."""
    cache_file = f"{DATA_DIR}/{ticker.replace('-', '_')}.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"  {ticker}: loaded from cache ({len(df)} days)")
        return df['Close'].values

    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        if len(df) < 100:
            raise ValueError(f"Only {len(df)} data points")
        df.to_csv(cache_file)
        print(f"  {ticker}: downloaded ({len(df)} days)")
        return df['Close'].values
    except Exception as e:
        print(f"  {ticker}: download failed ({e}), using synthetic data")
        return None


def make_synthetic_stock(name, days=500):
    """Generate synthetic stock-like data as fallback."""
    np.random.seed(hash(name) % 2**32)
    returns = np.random.randn(days) * 0.02
    if 'volatile' in name:
        returns *= 2
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


def sma_forecast(history, horizon, window=20):
    """Simple moving average forecast."""
    sma = np.mean(history[-window:])
    return np.full(horizon, sma)


def compute_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs(actual - predicted) / np.abs(actual + 1e-8)) * 100
    return mae, mape


def directional_accuracy(actual, predicted, history_last):
    """% of days where predicted direction matches actual direction."""
    actual_dir = np.sign(np.diff(np.concatenate([[history_last], actual])))
    pred_dir = np.sign(np.diff(np.concatenate([[history_last], predicted])))
    return np.mean(actual_dir == pred_dir) * 100


# --- Fetch data ---
print("Fetching stock data...")
tickers = {
    "AAPL": "stable large-cap",
    "TSLA": "volatile growth",
    "BTC-USD": "crypto",
    "SPY": "index ETF",
}

stock_data = {}
for ticker, desc in tickers.items():
    prices = fetch_stock(ticker)
    if prices is None:
        prices = make_synthetic_stock(desc)
        print(f"  {ticker}: using synthetic ({len(prices)} days)")
    stock_data[ticker] = prices

# --- Run forecasts ---
HOLDOUT = 30  # 30 trading days ahead

print(f"\n{'Ticker':<12} {'Transform':<8} {'MAE':>8} {'MAPE%':>8} {'DirAcc%':>8} | {'Naive':>8} {'SMA20':>8} {'Lift%':>8}")
print("-" * 90)

results = []
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (ticker, prices) in enumerate(stock_data.items()):
    prices = np.array(prices, dtype=float).flatten()
    if len(prices) <= HOLDOUT + 50:
        print(f"  {ticker}: not enough data, skipping")
        continue

    history = prices[:-HOLDOUT]
    actual = prices[-HOLDOUT:]

    for transform in ["raw", "log"]:
        if transform == "log":
            h_input = np.log(history)
        else:
            h_input = history

        t0 = time.time()
        point, quantiles = model.forecast(horizon=HOLDOUT, inputs=[h_input])
        elapsed = time.time() - t0

        predicted = point[0, :HOLDOUT]
        if transform == "log":
            predicted = np.exp(predicted)
            q_low = np.exp(quantiles[0, :HOLDOUT, 1])
            q_high = np.exp(quantiles[0, :HOLDOUT, -2])
        else:
            q_low = quantiles[0, :HOLDOUT, 1]
            q_high = quantiles[0, :HOLDOUT, -2]

        mae, mape = compute_metrics(actual, predicted)
        dir_acc = directional_accuracy(actual, predicted, history[-1])

        naive_pred = np.full(HOLDOUT, history[-1])
        naive_mae, _ = compute_metrics(actual, naive_pred)

        sma_pred = sma_forecast(history, HOLDOUT)
        sma_mae, _ = compute_metrics(actual, sma_pred)

        lift = (1 - mae / naive_mae) * 100 if naive_mae > 0 else 0

        results.append({
            "ticker": ticker, "transform": transform,
            "mae": round(mae, 2), "mape": round(mape, 2),
            "directional_accuracy": round(dir_acc, 1),
            "naive_mae": round(naive_mae, 2), "sma20_mae": round(sma_mae, 2),
            "lift_pct": round(lift, 1), "inference_time": round(elapsed, 2),
        })

        print(f"{ticker:<12} {transform:<8} {mae:>8.2f} {mape:>7.1f}% {dir_acc:>7.1f}% | {naive_mae:>8.2f} {sma_mae:>8.2f} {lift:>7.1f}%")

    # Plot (use raw transform)
    raw_result = [r for r in results if r['ticker'] == ticker and r['transform'] == 'raw'][-1]
    point_raw, quantiles_raw = model.forecast(horizon=HOLDOUT, inputs=[history])
    pred_raw = point_raw[0, :HOLDOUT]
    q_lo = quantiles_raw[0, :HOLDOUT, 1]
    q_hi = quantiles_raw[0, :HOLDOUT, -2]

    ax = axes[idx]
    t_hist = np.arange(len(history))
    t_fore = np.arange(len(history), len(history) + HOLDOUT)
    ax.plot(t_hist[-60:], history[-60:], 'b-', label='History', linewidth=1)
    ax.plot(t_fore, actual, 'g-', label='Actual', linewidth=2)
    ax.plot(t_fore, pred_raw, 'r--', label='TimesFM', linewidth=2)
    ax.fill_between(t_fore, q_lo, q_hi, alpha=0.2, color='red', label='P10-P90')
    ax.axhline(y=history[-1], color='k', linestyle=':', alpha=0.3, label='Last price')
    ax.set_title(f"{ticker} (MAE={raw_result['mae']:.1f}, Lift={raw_result['lift_pct']:.0f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_financial_stocks.png", dpi=150)
print(f"\nChart saved: {RESULTS_DIR}/02_financial_stocks.png")

# Save CSV
with open(f"{RESULTS_DIR}/02_financial_stocks.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved: {RESULTS_DIR}/02_financial_stocks.csv")

# Summary
raw_results = [r for r in results if r['transform'] == 'raw']
avg_lift = np.mean([r['lift_pct'] for r in raw_results])
avg_dir = np.mean([r['directional_accuracy'] for r in raw_results])
print(f"\nRaw transform — Avg lift vs naive: {avg_lift:.1f}%, Avg directional accuracy: {avg_dir:.1f}%")

log_results = [r for r in results if r['transform'] == 'log']
avg_lift_log = np.mean([r['lift_pct'] for r in log_results])
print(f"Log transform — Avg lift vs naive: {avg_lift_log:.1f}%")

if avg_lift < 5 and avg_dir < 55:
    print("\nVERDICT: TimesFM is NOT competitive for financial forecasting (as expected).")
    print("  → Exclude financial use cases from v1 product.")
elif avg_lift > 10:
    print("\nVERDICT: TimesFM shows surprising promise for financial forecasting.")
    print("  → Worth investigating further, but high risk.")
else:
    print("\nVERDICT: TimesFM is marginally useful for financial data.")
    print("  → Not a primary use case, but could be a secondary feature.")

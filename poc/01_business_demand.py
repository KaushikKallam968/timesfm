"""POC 1: Business/Demand Forecasting — validate TimesFM on its strongest domain."""
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
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# --- Generate test datasets ---
np.random.seed(42)

def make_daily_seasonal(n=365, trend=0.5, noise=0.1):
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 7)  # weekly cycle
    return 100 + trend * t + seasonal + noise * np.random.randn(n)

def make_monthly_seasonal(n=144):
    t = np.arange(n)
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    trend = 0.3 * t
    return 50 + trend + seasonal + 2 * np.random.randn(n)

# Air Passengers (classic 144-month dataset)
AIR_PASSENGERS = [
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432,
]

datasets = {
    "daily_trending": make_daily_seasonal(365, trend=0.5, noise=5),
    "monthly_seasonal": make_monthly_seasonal(144),
    "air_passengers": np.array(AIR_PASSENGERS, dtype=float),
    "noisy_demand": make_daily_seasonal(365, trend=0.2, noise=15),
}


def compute_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs(actual - predicted) / np.abs(actual + 1e-8)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, mape, rmse


def naive_forecast(history, horizon):
    return np.full(horizon, history[-1])


# --- Run forecasts ---
print(f"{'Series':<25} {'MAE':>8} {'MAPE%':>8} {'RMSE':>8} | {'Naive MAE':>10} {'Lift%':>8}")
print("-" * 80)

results = []
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, data) in enumerate(datasets.items()):
    holdout_pct = 0.2
    holdout_len = max(int(len(data) * holdout_pct), 10)
    holdout_len = min(holdout_len, 128)  # max horizon is 128

    history = data[:-holdout_len]
    actual = data[-holdout_len:]

    t0 = time.time()
    point, quantiles = model.forecast(horizon=holdout_len, inputs=[history])
    elapsed = time.time() - t0

    predicted = point[0, :holdout_len]
    q_low = quantiles[0, :holdout_len, 1]   # ~10th percentile
    q_high = quantiles[0, :holdout_len, -2]  # ~90th percentile

    mae, mape, rmse = compute_metrics(actual, predicted)
    naive_pred = naive_forecast(history, holdout_len)
    naive_mae, _, _ = compute_metrics(actual, naive_pred)
    lift = (1 - mae / naive_mae) * 100 if naive_mae > 0 else 0

    results.append({
        "series": name, "mae": round(mae, 2), "mape": round(mape, 2),
        "rmse": round(rmse, 2), "naive_mae": round(naive_mae, 2),
        "lift_pct": round(lift, 1), "inference_time": round(elapsed, 2),
    })

    print(f"{name:<25} {mae:>8.2f} {mape:>7.1f}% {rmse:>8.2f} | {naive_mae:>10.2f} {lift:>7.1f}%")

    # Plot
    ax = axes[idx]
    t_hist = np.arange(len(history))
    t_fore = np.arange(len(history), len(history) + holdout_len)
    ax.plot(t_hist[-60:], history[-60:], 'b-', label='History', linewidth=1)
    ax.plot(t_fore, actual, 'g-', label='Actual', linewidth=2)
    ax.plot(t_fore, predicted, 'r--', label='TimesFM', linewidth=2)
    ax.fill_between(t_fore, q_low, q_high, alpha=0.2, color='red', label='P10-P90')
    ax.plot(t_fore, naive_pred, 'k:', label='Naive', linewidth=1, alpha=0.5)
    ax.set_title(f"{name} (MAE={mae:.1f}, Lift={lift:.0f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_business_demand.png", dpi=150)
print(f"\nChart saved: {RESULTS_DIR}/01_business_demand.png")

# Save CSV
with open(f"{RESULTS_DIR}/01_business_demand.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved: {RESULTS_DIR}/01_business_demand.csv")

# Summary
avg_lift = np.mean([r['lift_pct'] for r in results])
print(f"\nAverage lift vs naive: {avg_lift:.1f}%")
if avg_lift > 20:
    print("VERDICT: TimesFM significantly outperforms naive baseline on business data.")
else:
    print("VERDICT: TimesFM shows marginal improvement over naive baseline.")

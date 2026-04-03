"""POC 3: Quantile Coverage/Calibration — are TimesFM's uncertainty estimates reliable?"""
import time
import numpy as np
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

# The quantile head outputs 10 quantiles. Check which quantile levels they are.
# TimesFM 2.5 default quantiles: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

np.random.seed(42)

# --- Test datasets ---
def make_daily_seasonal(n=365, trend=0.5, noise=5):
    t = np.arange(n)
    return 100 + trend * t + 10 * np.sin(2 * np.pi * t / 7) + noise * np.random.randn(n)

def make_monthly_seasonal(n=144):
    t = np.arange(n)
    return 50 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 12) + 2 * np.random.randn(n)

def make_random_walk(n=500):
    returns = np.random.randn(n) * 0.02
    return 100 * np.exp(np.cumsum(returns))

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
    "daily_trending": make_daily_seasonal(365),
    "monthly_seasonal": make_monthly_seasonal(144),
    "air_passengers": np.array(AIR_PASSENGERS, dtype=float),
    "random_walk": make_random_walk(500),
    "low_noise": make_daily_seasonal(365, noise=1),
    "high_noise": make_daily_seasonal(365, noise=20),
}

# --- Compute calibration ---
print(f"{'Series':<20}", end="")
for q in QUANTILE_LEVELS:
    print(f" q{q:.2f}", end="")
print(f" {'CalErr':>8}")
print("-" * 120)

all_calibration = {}
results = []

for name, data in datasets.items():
    holdout_len = min(int(len(data) * 0.2), 128)
    holdout_len = max(holdout_len, 10)

    history = data[:-holdout_len]
    actual = data[-holdout_len:]

    point, quantiles = model.forecast(horizon=holdout_len, inputs=[history])
    q_forecasts = quantiles[0, :holdout_len, :]  # [holdout, 10]

    # Compute empirical coverage for each quantile level
    coverages = []
    for qi, q_level in enumerate(QUANTILE_LEVELS):
        # What fraction of actual values fall below the q-th quantile forecast?
        coverage = np.mean(actual <= q_forecasts[:, qi])
        coverages.append(coverage)

    # Calibration error = mean absolute difference between expected and observed
    cal_errors = [abs(expected - observed) for expected, observed in zip(QUANTILE_LEVELS, coverages)]
    mean_cal_error = np.mean(cal_errors)

    all_calibration[name] = coverages

    row = {"series": name}
    for qi, q_level in enumerate(QUANTILE_LEVELS):
        row[f"q{q_level:.2f}_coverage"] = round(coverages[qi], 3)
    row["mean_calibration_error"] = round(mean_cal_error, 3)
    results.append(row)

    print(f"{name:<20}", end="")
    for cov in coverages:
        print(f" {cov:>5.2f}", end="")
    print(f" {mean_cal_error:>8.3f}")

# --- Overall calibration ---
print("\n" + "=" * 80)
avg_coverages = np.mean([list(all_calibration[k]) for k in all_calibration], axis=0)
avg_errors = [abs(expected - observed) for expected, observed in zip(QUANTILE_LEVELS, avg_coverages)]
overall_cal = np.mean(avg_errors)

print(f"\nOverall avg calibration error: {overall_cal:.3f}")
print("(0 = perfect, <0.05 = excellent, <0.10 = good, >0.15 = poor)")

# --- Plot calibration curve ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: calibration plot (expected vs observed)
ax = axes[0]
for name, coverages in all_calibration.items():
    ax.plot(QUANTILE_LEVELS, coverages, 'o-', label=name, markersize=4, alpha=0.7)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
ax.set_xlabel("Expected coverage (quantile level)")
ax.set_ylabel("Observed coverage")
ax.set_title(f"Quantile Calibration (avg error = {overall_cal:.3f})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Right: bar chart of calibration errors by dataset
ax = axes[1]
names = list(all_calibration.keys())
cal_errors_per_ds = [results[i]["mean_calibration_error"] for i in range(len(results))]
colors = ['green' if e < 0.05 else 'orange' if e < 0.10 else 'red' for e in cal_errors_per_ds]
ax.barh(names, cal_errors_per_ds, color=colors)
ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.05)')
ax.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5, label='Good (<0.10)')
ax.set_xlabel("Mean Calibration Error")
ax.set_title("Calibration Quality by Dataset")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_quantile_calibration.png", dpi=150)
print(f"\nChart saved: {RESULTS_DIR}/03_quantile_calibration.png")

# Save CSV
with open(f"{RESULTS_DIR}/03_quantile_calibration.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved: {RESULTS_DIR}/03_quantile_calibration.csv")

# Verdict
business_ds = ["daily_trending", "monthly_seasonal", "air_passengers", "low_noise", "high_noise"]
biz_cal = np.mean([r["mean_calibration_error"] for r in results if r["series"] in business_ds])
fin_cal = np.mean([r["mean_calibration_error"] for r in results if r["series"] == "random_walk"])

print(f"\nBusiness data avg calibration error: {biz_cal:.3f}")
print(f"Financial data avg calibration error: {fin_cal:.3f}")

if biz_cal < 0.10:
    print("\nVERDICT: Quantiles are well-calibrated on business data.")
    print("  → Uncertainty estimates are RELIABLE for demand planning.")
    print("  → 'Order enough for P80 demand' actually works.")
else:
    print("\nVERDICT: Quantiles need improvement on business data.")
    print("  → May need post-hoc calibration (conformal prediction).")

"""Smoke test: load TimesFM from local weights, forecast a sine wave, verify output."""
import time
import numpy as np

print("=" * 60)
print("TimesFM Smoke Test")
print("=" * 60)

# Load model from local checkpoint
print("\n[1/3] Loading model from local safetensors...")
t0 = time.time()
import timesfm

MODEL_DIR = "/home/user/timesfm/poc/model_cache/pytorch"
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    MODEL_DIR, torch_compile=False, local_files_only=True
)
load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s")

# Compile forecast config
print("\n[2/3] Compiling forecast config...")
t0 = time.time()
model.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=128,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True,
))
compile_time = time.time() - t0
print(f"  Compiled in {compile_time:.1f}s")

# Forecast sine wave
print("\n[3/3] Forecasting sine wave (256 points -> 128 ahead)...")
x = np.sin(np.linspace(0, 8 * np.pi, 256))
t0 = time.time()
point, quantiles = model.forecast(horizon=128, inputs=[x])
forecast_time = time.time() - t0

print(f"  Forecast in {forecast_time:.1f}s")
print(f"  Point shape: {point.shape}")
print(f"  Quantile shape: {quantiles.shape if quantiles is not None else 'None'}")
print(f"  Point forecast first 5: {point[0, :5].round(4)}")
print(f"  Any NaN in point: {np.any(np.isnan(point))}")

# Sanity checks
actual_future = np.sin(np.linspace(8 * np.pi, 12 * np.pi, 128))
mae = np.mean(np.abs(point[0] - actual_future))
in_range = np.all(np.abs(point) < 5)
no_nan = not np.any(np.isnan(point))

print(f"\n  Sine wave MAE: {mae:.4f}")
print(f"  Values in sane range: {in_range}")

print("\n" + "=" * 60)
if no_nan and in_range:
    print(f"SMOKE TEST PASSED (MAE={mae:.4f})")
else:
    print("SMOKE TEST FAILED")
print(f"Total time: {load_time + compile_time + forecast_time:.1f}s")
print("=" * 60)

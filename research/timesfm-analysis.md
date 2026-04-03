# TimesFM Deep Analysis

## What It Is

TimesFM is a **decoder-only transformer foundation model** for time series forecasting, built by Google Research. It performs **zero-shot forecasting** — you give it historical data, it predicts the future without any training on your specific dataset.

- **License:** Apache 2.0 (code AND weights) — fully commercial, no royalties, no permission needed
- **Latest Model:** `google/timesfm-2.5-200m-pytorch` on HuggingFace
- **Parameters:** 200M (v2.5), smaller and better than the 500M v2.0
- **Pre-trained on:** 307 billion timepoints from 205 million time series

## Architecture

### Core Design

| Component | Specification |
|-----------|--------------|
| Type | Decoder-only transformer |
| Layers | 20 stacked transformer layers |
| Attention Heads | 16 multi-head causal attention |
| Model Dimension | 1,280 |
| Head Dimension | 80 (1280/16) |
| Input Patch Size | 32 timepoints → 1 token |
| Output Patch Size | 128 timepoints per output token |
| Max Context | 16,384 timepoints (v2.5) |
| Max Forecast Horizon | 1,024 steps (with quantile head) |
| Attention Norm | RMS |
| Position Encoding | Rotary (RoPE) |
| FF Activation | Swish |
| QKV Fusion | Yes (Oct 2025 optimization) |

### How Forecasting Works

1. **Patching:** Raw time series split into non-overlapping 32-timepoint patches
2. **Tokenization:** Each patch → tokenizer block (64-dim input → 1,280-dim embedding)
3. **Transformer:** All patches processed through 20 layers with causal attention (can only see past, not future)
4. **Decoding:** Each output position produces 128 timepoints + 10 values (1 mean + 9 quantiles)
5. **Autoregressive Extension:** For horizons > 128, takes last 4 output patches (512 points), feeds back as input, repeats

### Training Data

- **307 billion timepoints** from 205.3 million time series
- **Sources:** Google Trends, Wikipedia pageviews, M4, electricity load, traffic data
- **Synthetic:** ~50% of training mix from 3M ARMA-generated series
- **Granularities:** Hourly, daily, weekly, monthly (balanced)
- **Minimum series length:** 256 timepoints

### Loss Function

Hybrid MSE + Quantile (Pinball) Loss:
```
Loss = MSE(mean_output, target) + Σ QuantileLoss_i(quantile_i, target)
```

## Model Variants

| Feature | v1.0 (200M) | v2.0 (500M) | v2.5 (200M) |
|---------|------------|------------|------------|
| Max Context | 512 | 2,048 | 16,384 |
| Max Forecast | 512 | 2,048 | 1,024 (quantile) |
| Quantile Heads | None | 10 (uncalibrated) | 10 (calibrated) + 30M head |
| Frequency Indicator | Required | Required | Removed |
| Covariate Support | No | No | Yes (XReg) |
| HuggingFace ID | `google/timesfm-1.0-200m` | `google/timesfm-2.0-500m-pytorch` | `google/timesfm-2.5-200m-pytorch` |

**Why v2.5 is better with fewer params:** QKV fusion optimization, better calibrated quantiles, 8x context window, simplified design.

## Complete API Surface

### Loading & Setup

```python
import timesfm
import numpy as np

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(timesfm.ForecastConfig(
    max_context=1024,          # Must be multiple of 32
    max_horizon=256,           # Must be multiple of 128
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
))
```

### ForecastConfig — All Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_context` | 0 | Max input length (multiple of 32) |
| `max_horizon` | 0 | Max forecast length (multiple of 128) |
| `normalize_inputs` | False | Per-series normalization |
| `per_core_batch_size` | 1 | Batch size per device |
| `use_continuous_quantile_head` | False | 30M quantile head (horizon ≤ 1024) |
| `force_flip_invariance` | True | Ensure f(-x) = -f(x) symmetry |
| `infer_is_positive` | True | Floor output to 0 if input ≥ 0 |
| `fix_quantile_crossing` | False | Monotonically enforce quantile order |
| `return_backcast` | False | Return fitted values (needed for XReg) |

### forecast() — Basic Forecasting

```python
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[np.array([1, 2, 3, ...]), np.sin(np.linspace(0, 20, 67))]
)
# point_forecast: (batch_size, horizon) — median forecast
# quantile_forecast: (batch_size, horizon, 10) — all quantiles
```

- Strips leading NaNs automatically
- Linear interpolation for NaN gaps
- Auto-pads short series, truncates long ones
- Variable lengths in same batch supported

### forecast_with_covariates() — External Regressors

```python
point_forecasts, quantile_forecasts = model.forecast_with_covariates(
    inputs=[series1, series2],
    dynamic_numerical_covariates={
        "temperature": [[15, 16, 17, 18, 19, 20], [20, 21, 22, 23, 24, 25]]
    },
    dynamic_categorical_covariates=None,
    static_numerical_covariates=None,
    static_categorical_covariates=None,
    xreg_mode="xreg + timesfm",  # or "timesfm + xreg"
    normalize_xreg_target_per_input=True,
    ridge=0.0,
    max_rows_per_col=0,
    force_on_cpu=False,
)
```

**Modes:**
- `"xreg + timesfm"`: Fit linear model on raw targets, forecast residuals with TimesFM
- `"timesfm + xreg"`: Forecast with TimesFM, fit linear model on residuals

**Important:** Dynamic covariates must span BOTH context AND forecast periods. Requires `return_backcast=True`.

## Infrastructure Requirements

### GPU Memory (v2.5, 200M params)

| Scenario | VRAM Needed |
|----------|-------------|
| Model weights (float32) | ~800 MB |
| Batch 1, context 512, horizon 256 | ~2.5 GB |
| Batch 8, context 2048, horizon 1024 | ~15-18 GB |
| Minimum viable | 4 GB |
| High-throughput batch | 16+ GB |

### Inference Speed (Estimated)

| Config | GPU (A100) | CPU |
|--------|-----------|-----|
| Single forecast (512 ctx, 128 horizon) | ~50-200 ms | ~2-5 seconds |
| Batch of 32 | ~500-1000 ms | ~30-60 seconds |

### System Requirements
- Python 3.10 or 3.11 (NOT 3.12+)
- 32+ GB RAM minimum
- PyTorch backend recommended (JAX/Flax also available)
- Model cold-start: 10-30 seconds

## Strengths

1. **Zero-shot generalization** — works on data it's never seen
2. **Probabilistic outputs** — calibrated P10-P90 quantiles for uncertainty
3. **Huge context window** — 16K timepoints captures long-term patterns
4. **Efficient** — 200M params, runs on modest hardware
5. **Apache 2.0** — fully commercial, no strings attached
6. **Google BigQuery integration** — enterprise-ready path
7. **Covariate support** — incorporate external factors

## Limitations & Known Issues

### Critical Limitations

1. **Financial forecasting is POOR zero-shot** — requires fine-tuning on 100M+ financial timepoints to reach acceptable performance. Stock prices have too low signal-to-noise for zero-shot.

2. **Univariate only** — doesn't natively model cross-series dependencies. XReg support is a linear regression post-hoc layer, not deep learning integration.

3. **v2.5 is ~15x slower than v2.0** (GitHub issue #313) — performance regression.

4. **Known bugs:**
   - NaN outputs on certain inputs (#321)
   - Flatlining on H100 GPUs (#328)
   - Covariate test data leakage (#338) — methodological flaw
   - Inconsistent results across batch sizes (#274)
   - Python 3.12 incompatible (#354)
   - Windows DLL loading failures (#330)

5. **Cloud/seasonal data** — naive seasonal models beat TimesFM by 2x on cloud workloads

6. **No cached decoding** — can't speed up autoregressive generation

7. **Dependency complexity** — requires lingvo (ARM/M1-incompatible), specific package versions

## Key Takeaway for Monetization

TimesFM is a strong general-purpose forecaster that works well zero-shot on business data (demand, traffic, energy) but **requires fine-tuning for financial data**. The probabilistic outputs (quantile forecasts) are the key differentiator — uncertainty quantification is what makes forecasts actionable for business decisions. The Apache 2.0 license means we can build anything commercial on top of it.

**Best domains for immediate commercial value:** Retail demand planning, energy forecasting, SaaS metrics prediction, anomaly detection.

**Worst domain without work:** Raw stock price prediction (needs fine-tuning + careful framing).

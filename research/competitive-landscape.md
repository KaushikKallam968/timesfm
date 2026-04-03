# Competitive Landscape — Time Series Foundation Models

## Model Comparison

### Scorecard

| Dimension | TimesFM 2.5 | Chronos-2 | Lag-Llama | Moirai 2.0 | N-HiTS |
|-----------|:-----------:|:---------:|:---------:|:----------:|:------:|
| Zero-Shot Accuracy | ★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★ |
| Multivariate Support | ★★ | ★★★★★ | ★ | ★★★★★ | ★★★ |
| Commercial License | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ Apache 2.0 | ❌ CC-BY-NC | ✅ MIT |
| Cloud Integration | BigQuery | AWS Deep | Emerging | Salesforce | None |
| Inference Efficiency | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| Fine-Tuning | ★★★ | ★★★ | ★★★★★ | ★★★ | ★★ |
| Community/Ecosystem | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★ |

### Commercially Viable Models (Apache 2.0 / MIT)

**1. Amazon Chronos-2 (Oct 2025)** — THE main competitor
- 120M params, encoder-only transformer
- **Outperforms TimesFM 2.5** on comprehensive benchmarks (fev-bench, GIFT-Eval)
- Native multivariate + covariate support (biggest advantage)
- 300+ forecasts/second on A10G GPU
- Deep AWS integration (SageMaker, AutoGluon)
- Millions of HuggingFace downloads
- **Risk:** Amazon can subsidize pricing, undercut any startup

**2. Lag-Llama (Feb 2024)** — Best for fine-tuning
- Decoder-only transformer, probabilistic outputs
- First open-source TS foundation model
- **Best fine-tuning workflows** — well-documented, Colab demos
- Univariate focus (like TimesFM)
- Strong community
- Good for: building fine-tuned vertical models

**3. N-BEATS / N-HiTS** — Proven, interpretable
- Not foundation models (require per-dataset training)
- MIT license
- Proven in financial applications
- Interpretable (decomposed trend/seasonality)
- Good for: specific domains where you have training data

### NOT Commercially Viable

**Salesforce Moirai 2.0** — CC-BY-NC 4.0, research only. Cannot use in commercial products. Eliminated.

### Other Notable Models

| Model | License | Notes |
|-------|---------|-------|
| PatchTST | MIT | Patches + channel-independent, 21% MSE reduction |
| iTransformer | MIT | Inverted attention, great on high-dimensional data |
| MOMENT | MIT | Multi-task (forecast + classify + anomaly + impute) |
| Datadog Toto | Apache 2.0 | New, limited adoption data |

## Existing Commercial Products

### Cloud Platform Offerings

| Product | Status | Pricing | Notes |
|---------|--------|---------|-------|
| Google BigQuery + TimesFM | Active | Query pricing | Built-in TimesFM integration |
| Amazon Forecast | **Dead** (no new customers) | Legacy | Replaced by Chronos approach |
| AWS SageMaker + Chronos-2 | Active | Compute-based | Direct deployment |
| Azure Time Series Insights | Active | IoT-focused | Visualization, not cutting-edge forecasting |
| IBM Watson Time Series API | Active (Feb 2025) | Usage-based | IoT/stock data |

### SaaS Forecasting Tools

| Product | Target | Pricing | Gap |
|---------|--------|---------|-----|
| Nixtla | Enterprise demand planning | Enterprise pricing | Statistical models, not foundation models |
| PredictHQ | Event-driven demand | API-based | Niche (event correlation) |
| Forecast.app | Professional services | Custom quote | PSA platform, not pure forecasting |
| Baremetrics Forecast+ | SaaS companies | Subscription | Revenue/MRR only |

### Market Gap Analysis

**What's missing in the market:**

1. **Self-serve foundation model forecasting** — No simple "upload CSV, get forecast" product using modern foundation models for SMBs
2. **Data integration layer** — Enterprises need 15-20 data sources connected; nobody makes this easy
3. **Domain-specific fine-tuned models** — Generic models exist, but fine-tuned vertical models (energy, retail, finance) are not productized
4. **Interpretability layer** — Foundation models are black boxes; enterprises want explanations
5. **Affordable alternative** — BigQuery/AWS solutions are expensive for small teams

## Financial Trading ML — Reality Check

### What Works
- Short-term pattern detection in high-frequency data
- Sentiment analysis integration improves signals
- XGBoost achieves 67-87% directional accuracy in research
- Best on crypto/derivatives (lower liquidity = patterns persist)

### What Doesn't
- Cannot consistently beat efficient markets long-term
- Overfitting and lookback bias are serious risks
- Live performance typically 40-60% of backtest results
- Foundation models NOT trained specifically for finance

### Regulatory Reality
- SEC/FINRA: "Technology-neutral" — existing rules apply to AI
- 4 SEC enforcement actions in past year for AI misrepresentation
- Cannot promise returns or overstate accuracy
- Registration requirements if offering "investment advice"
- **Safe framing:** "Research tool" or "quantitative analysis assistant"
- **Cost to properly set up:** $100K-$1M+ in legal/compliance

### Honest Assessment
Trading signals are a **high-risk, moderate-reward** play:
- Target: sophisticated investors (hedge funds, prop traders)
- NOT retail investors (regulatory minefield)
- Realistic ARR if successful: $1-5M (niche market)
- Need fine-tuning + significant compliance investment

## TimesFM's Competitive Position

### Where TimesFM Wins
1. **Google ecosystem** — BigQuery, Vertex AI integration
2. **Retail demand planning** — beats Chronos-2 on 43% of retail SKUs (Decathlon benchmark)
3. **Lightweight inference** — 200M params, edge/mobile viable
4. **In-context fine-tuning (ICF)** — no-gradient adaptation at inference time

### Where TimesFM Loses
1. **Overall accuracy** — Chronos-2 now outperforms on aggregate benchmarks
2. **Multivariate** — Chronos-2 and Moirai are far superior
3. **Ecosystem** — AWS has deeper integration and larger community
4. **Speed** — v2.5 has a 15x regression vs v2.0

### Bottom Line

TimesFM is a **top-3 commercially viable** time series foundation model (alongside Chronos-2 and Lag-Llama). Its Apache 2.0 license, Google backing, and BigQuery integration make it a strong choice. But Chronos-2 is currently the performance leader with better multivariate support.

**The monetization play is NOT "best model wins."** It's:
- Best **product** around the model wins
- Best **data integration** wins
- Best **vertical specialization** wins
- Best **user experience** wins

The model is table stakes. The product is the moat.

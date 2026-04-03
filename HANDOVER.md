# Session Handover — TimesFM Prediction Market Bot

## What This Project Is

We're building a **prediction market trading bot** using Google's TimesFM. The ideology is **profit over all**. The user is a solo bootstrapped developer — I (Claude) am the team.

## What's Been Done

### Session 1: Research (Complete)
Three research docs on branch `claude/timesfm-monetization-exploration-tNx9l`:
- `research/timesfm-analysis.md` — Deep technical analysis of TimesFM 2.5
- `research/competitive-landscape.md` — Competitor analysis (Chronos-2, Moirai)
- `research/monetization-strategies.md` — 6 strategies ranked by profit potential

### Session 2: POC Validation + Trading Bot Framework (Complete)

**POC Results (validated with real TimesFM model):**
- Business demand forecasting: **55.9% avg lift** over naive (model works great)
- Financial price prediction: **-3.4% vs naive** zero-shot (model fails, as expected)
- Quantile calibration: Upper quantiles good (P80/P90), lower quantiles too tight
- All in `poc/` directory with charts and CSVs

**Model Setup (critical — HuggingFace blocked by proxy):**
- Downloaded Flax checkpoint from GCS: `gs://vertex-model-garden-public-us/timesfm/timesfm-2.5-200m-flax`
- Converted Flax→PyTorch safetensors via tensorstore (820MB download, all 232 params matched)
- Conversion script: `scripts/convert_flax_to_pytorch.py`
- Model path: `poc/model_cache/pytorch/model.safetensors`
- Session start hook auto-installs deps + downloads model: `.claude/hooks/session-start.sh`

**Trading Bot Framework (built, not yet profitable):**
- `bot/data/downloader.py` — OHLCV data pipeline with synthetic fallback
- `bot/model/finetune.py` — Fine-tuning scaffold (freeze backbone, train heads)
- `bot/backtest/engine.py` — Vectorized backtester with fee modeling
- `bot/backtest/strategies.py` — Momentum + quantile-volatility strategies
- Zero-shot backtest: BTC momentum_biweekly Sharpe 1.14 (best), avg -0.10 (expected)

### Pivot: Prediction Market Arbitrage Bot (Planned, Not Built)

Research shows prediction market bots are extremely profitable:
- $40M extracted from Polymarket in 1 year by arb bots
- 14/20 most profitable Polymarket wallets are bots
- Weather bot: $1K→$24K documented
- Cross-platform arb: 12-20% monthly returns

**4 strategies identified (ranked by viability):**

1. **Weather Forecast Arbitrage** — Compare free GFS ensemble (Open-Meteo) to Polymarket odds. When forecast says 90% and market says 72%, buy. $500-2K/month potential.

2. **Cross-Platform Arbitrage** — Buy YES on Kalshi + NO on Polymarket when combined < $0.975. Risk-free. Need 2.5¢+ spread after fees. Polymarket: 0% on standard markets. Kalshi: ceil(0.07 × P × (1-P) × 100)/100.

3. **Market Making** — Provide liquidity both sides, earn spread + Polymarket maker rebates (25-50% of taker fees, paid daily USDC). Steady 0.5-2%/month.

4. **News/Information Latency** — 30sec-5min window after breaking news before odds adjust.

**Key 2026 changes:**
- Polymarket removed 500ms taker delay (Feb 2026) → pure latency arb on crypto markets is dead
- Dynamic taker fees up to 3.15% on 15-min crypto markets
- Standard markets remain 0% fee
- Maker rebate program expanded to nearly all new markets

## What's Next — Use Sequential Thinking

**MCP servers installed (available next session):**
- `sequential-thinking` — Step-by-step reasoning with branching
- `sqlite` — Local database at `/home/user/timesfm/bot/data/bot.db`
- `fetch` — HTTP requests

**The plan is written at:** `/root/.claude/plans/eager-tumbling-flamingo.md`

**Next steps:**
1. Use sequential-thinking MCP to walk through the plan step-by-step
2. Validate fee calculations with live Polymarket/Kalshi data
3. Build Phase 1: Weather arbitrage bot
4. Build Phase 2: Cross-platform arb scanner
5. Build Phase 3: TimesFM odds forecasting enhancement

## Key Dependencies (installed via session-start hook)
```
timesfm (from GitHub), torch, pandas, numpy, matplotlib
tensorstore, orbax-checkpoint, gcsfs, safetensors
yfinance (works locally, blocked by proxy in cloud)
py-clob-client (Polymarket — NOT YET INSTALLED)
web3 (Ethereum signing — NOT YET INSTALLED)
```

## Git State
- **Branch:** `claude/continue-handover-BXPnI`
- **Remote:** `kaushikkallam968/timesfm`
- **Files:**
  ```
  CLAUDE.md, HANDOVER.md, README.md, requirements.txt
  research/*.md
  poc/{00-03}_*.py, poc/results/, poc/FINDINGS.md
  scripts/convert_flax_to_pytorch.py
  bot/data/downloader.py, bot/model/finetune.py
  bot/backtest/{engine,strategies}.py
  bot/run_backtest.py, bot/results/
  .claude/hooks/session-start.sh, .claude/settings.json
  ```

## User Preferences
- Profit over all — every decision optimizes for revenue
- Solo bootstrapped — lean, scrappy, no over-engineering
- Wants deep research before committing to a direction
- JS-first for product layer, Python for ML/trading backend
- Conventional commits, no direct push to main

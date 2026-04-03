# Session Handover — TimesFM Monetization Project

## What This Project Is

We're exploring every possible way to make money using Google's TimesFM (open-source time series foundation model, Apache 2.0). The ideology is **profit over all**. The user is a solo bootstrapped developer — I (Claude) am the team.

## What's Been Done

### Research (Complete)
Three comprehensive research documents have been written and pushed to branch `claude/timesfm-monetization-exploration-tNx9l`:

1. **`research/timesfm-analysis.md`** — Deep technical analysis of TimesFM
   - Architecture: 200M param decoder-only transformer, 20 layers, 16 heads, 32-pt input patches
   - API: `forecast()`, `forecast_on_df()`, `forecast_with_covariates()`
   - Best model: `google/timesfm-2.5-200m-pytorch` (v2.5, 16K context, calibrated quantiles)
   - Key limitation: **Financial forecasting is POOR zero-shot — needs fine-tuning**
   - Known bugs: 15x slower than v2.0, NaN outputs, batch inconsistency

2. **`research/competitive-landscape.md`** — Competitor analysis
   - Chronos-2 (Amazon) is the main threat — better benchmarks, better multivariate
   - Moirai (Salesforce) eliminated — CC-BY-NC, no commercial use
   - Market gap: No simple self-serve foundation model forecasting for SMBs
   - **The model is NOT the moat. Product, UX, and data connectors are.**

3. **`research/monetization-strategies.md`** — 6 strategies ranked by profit potential
   - Recommended path: Consulting (quick cash) → API service → SaaS platform
   - GPU margins are ~97% at $29/mo tier. Break-even at ~15-30 customers.

### Expanded Monetization Plan (Complete)
The plan file at `/root/.claude/plans/piped-tickling-newell.md` contains 15 strategies across 4 tiers:
- **Tier 1 Quick Cash:** Freelance gigs, content/education, consulting
- **Tier 2 Safe Bets:** Google Sheets add-on, Shopify app, API service, Chrome extension, Slack bot
- **Tier 3 Bigger Bets:** SaaS platform, fine-tuned vertical models, white-label for SaaS platforms
- **Tier 4 Moonshots:** LLM+forecasting chat, creator growth tool, sports betting, DeFi yield predictor, real estate, autonomous monitoring agent

### Tools Installed (Complete)
**Plugins:**
- `superpowers@superpowers-marketplace` — Full dev workflow (TDD, brainstorming, plans, subagent dev)
- `claude-mem@thedotmack` — Persistent memory across sessions

**MCP Servers (User scope):**
- `sequential-thinking` — Step-by-step reasoning with branching
- `sqlite` — Local database for prototyping
- `filesystem` — Enhanced file access
- `fetch` — HTTP requests to any URL
- `memory` — Knowledge graph persistence

**MCP Servers (Project scope — /home/user/timesfm):**
- `context7` — Live version-specific docs
- `playwright` — Browser automation/testing
- `tavily-remote-mcp` — Web search (1K free/mo)
- `figma` — Design token extraction

## What's Next — Phase 1: Hands-On Validation

The user approved this execution order but we haven't started building yet:

1. **Install TimesFM** — `pip install timesfm[torch]`, load model, run smoke test
2. **POC scripts** — Test on real data:
   - Financial data (stocks/crypto) — validate the "needs fine-tuning" claim
   - Business data (sales/demand) — validate zero-shot accuracy
   - Probabilistic forecasting — are quantile outputs actually useful?
3. **Pick the first product to build** — Based on POC results + user preference
4. **Build it** — Python backend (FastAPI) + JS frontend (React)

## Key Decisions Made
- Apache 2.0 license confirmed — fully commercial, no restrictions
- Multi-model approach (TimesFM + Chronos-2) is smarter than betting on one
- Start with consulting/freelance for immediate cash while building product
- Financial trading is high-risk, defer until safer revenue established
- JS-first for product layer, Python only for ML backend

## Git State
- **Branch:** `claude/timesfm-monetization-exploration-tNx9l`
- **Remote:** `kaushikkallam968/timesfm`
- **Last commit:** `907355c` — "feat: add TimesFM research documents for monetization exploration"
- **Files:**
  ```
  research/timesfm-analysis.md
  research/competitive-landscape.md
  research/monetization-strategies.md
  research/poc/          (empty, ready for scripts)
  CLAUDE.md
  README.md
  ```

## User Preferences
- Profit over all — every decision optimizes for revenue
- Solo bootstrapped — lean, scrappy, no over-engineering
- No rush — willing to be thorough
- CLAUDE.md rules: JS-first, functional components, YAGNI, conventional commits
- Wants to research deeply before committing to a direction

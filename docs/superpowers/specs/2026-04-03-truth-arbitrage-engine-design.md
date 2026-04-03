# Truth Arbitrage Engine — Design Spec

## Overview

A prediction market trading bot that exploits price discrepancies between Polymarket and external "truth sources" (sportsbook odds, weather forecasts, mathematical logic, cross-platform prices). The bot does NOT predict outcomes — it identifies where Polymarket disagrees with better-informed sources and bets on the truth.

**Core formula:** `EDGE = truth_probability - market_probability → if EDGE > threshold → BUY`

## Why This Works

- **$1→$3.3M documented** (sovereign2013 on Polymarket, sports arb vs sportsbooks, 98% win rate)
- **$1K→$79K documented** (weather arb, GFS forecast vs Polymarket)
- **$40M extracted** from Polymarket in 1 year by arb bots
- **14/20 most profitable Polymarket wallets are bots**
- Only 7.6% of Polymarket wallets are profitable — bots capture most of the value

## Strategy Priority (by profit potential)

### 1. Sports Odds Arbitrage (PRIMARY)
- **Truth source:** Sportsbook consensus (DraftKings, FanDuel, Betfair — via The Odds API)
- **Mechanism:** Professional bookmakers set odds using billions in betting data. When Polymarket disagrees, sportsbooks are almost always right.
- **Example:** DraftKings has Celtics at -250 (71.4% implied), Polymarket has YES at $0.62 (62%). Edge = 9.4% → buy YES.
- **Frequency:** Dozens of events daily (NBA, NFL, MLB, NHL, soccer, MMA)
- **Documented profit:** $1→$3.3M, 98% win rate

### 2. Weather Forecast Arbitrage
- **Truth source:** GFS 31-member ensemble forecast (Open-Meteo, free)
- **Mechanism:** Count ensemble members exceeding threshold. 28/31 = 90.3% probability. Market says 65%. Edge = 25.3%.
- **Frequency:** 5-10 markets/day across 7 cities
- **Documented profit:** $1K→$79K

### 3. Correlation Arbitrage
- **Truth source:** Mathematical logic (probabilities must sum to 100%)
- **Mechanism:** Scan related markets for logical violations. "Trump wins" should be ≤ "Republican wins." When violated → trade.
- **Frequency:** Variable, 1-3 opportunities/week
- **Documented:** 70-80% win rate, 2.3 day average hold

### 4. Cross-Platform Arbitrage
- **Truth source:** Kalshi price on same event
- **Mechanism:** Buy YES on one platform + NO on other when combined < $0.975 (after fees)
- **Frequency:** Around scheduled events (Fed, CPI, elections)
- **Constraint:** Kalshi's 7% fee makes most spreads unprofitable. Only trade >5¢ spreads.

### 5. TimesFM Enhancement (Optimization Layer)
- **Not a standalone strategy** — enhances all others
- **Predict odds trajectory:** When will Polymarket-sportsbook spread widen or close?
- **Market selection:** Which markets will have biggest mispricings tomorrow?
- **Entry timing:** Buy when spread is widening, not closing

## Architecture

```
bot/
  core/
    config.py          — Settings, thresholds, API keys (env vars)
    database.py        — SQLite: positions, trades, P&L, market cache
    risk.py            — Position limits, daily loss halt, kill switch
  truth/
    base.py            — TruthEngine abstract base class
    sports.py          — Sportsbook odds consensus (The Odds API)
    weather.py         — GFS ensemble forecast (Open-Meteo)
    correlation.py     — Mathematical consistency checker
    cross_platform.py  — Kalshi price comparator
    timesfm_enhancer.py — TimesFM odds trajectory prediction
  market/
    polymarket.py      — Polymarket CLOB client (py-clob-client)
    kalshi.py          — Kalshi REST API client
    scanner.py         — Active market discovery + matching
  execution/
    edge_detector.py   — Truth vs market comparison, edge calculation
    kelly.py           — Fractional Kelly position sizing
    order_manager.py   — Place/cancel orders, handle partial fills
    settlement.py      — Monitor resolution, track actual P&L
  backtest/
    historical_data.py — Fetch/store 5yr historical odds + outcomes
    simulator.py       — Replay bot logic on historical data
    optimizer.py       — Auto-research parameter optimization loop
    reporter.py        — Performance metrics, equity curves, attribution
  monitoring/
    telegram.py        — Trade alerts, daily P&L, error notifications
    reporter.py        — Performance reports, strategy attribution
  run.py              — Main loop: APScheduler, all strategies
```

### Data Flow

Every 5 minutes:
1. Scanner discovers active Polymarket markets (sports, weather, etc.)
2. For each market, query matching truth engine
3. Edge detector computes: truth_probability - market_probability
4. Filter: edge > threshold AND risk limits OK
5. Kelly sizer computes position ($, capped)
6. Order manager places trade via py-clob-client
7. Database logs everything
8. Telegram notifies on trades + daily summary

### TruthEngine Interface

```python
class TruthEngine:
    def get_probability(self, market) -> float | None:
        """Return the 'true' probability for this market, or None if no opinion."""
        ...

    def confidence(self, market) -> float:
        """0-1 confidence in this truth estimate."""
        ...
```

All truth sources implement this. The edge detector queries all engines for each market and uses the highest-confidence response.

## Backtesting Validation (BEFORE Live)

**No real money until backtesting proves profitability.**

### Historical Data Collection (5 years)
- **Sportsbook odds:** The Odds API historical data, or scrape odds archives (oddsportal.com, betting-data.com)
- **Polymarket prices:** Polymarket historical data API / Gamma Markets data
- **Weather forecasts:** Open-Meteo historical API (free, goes back years)
- **Event outcomes:** Who won, what temperature was, which candidate was elected

### Simulation
- For every historical market: what would each truth engine have said?
- What was the Polymarket price at that time?
- Would the edge threshold have triggered a trade?
- What would Kelly sizing have been?
- What was the actual outcome? Win or lose?
- Full equity curve over 5 years

### Auto-Research Optimization (targeting 95% win rate)

Using the `0-autoresearch-skill` two-loop architecture:

**Inner loop** (rapid iteration):
1. Run backtest with current parameters
2. Check win rate — if < 95%, adjust:
   - Edge threshold (try 5%, 10%, 15%, 20%)
   - Market filters (liquidity minimums, time-to-resolution)
   - Truth source weighting (which sources are most reliable?)
   - Kelly fraction (5%, 10%, 15%, 20%)
3. Re-run backtest, compare metrics
4. Repeat until win rate ≥ 95% on out-of-sample data

**Outer loop** (strategy synthesis):
- Which truth sources contribute most to wins?
- Which market types have highest win rates?
- Time-of-day or day-of-week patterns?
- Should we filter to only the highest-confidence trades?
- How does reducing trade frequency affect total profit?

**What changes when the loop fails (win rate < 95%):**

The inner loop applies these adjustments in order until the target is hit:

1. **Raise edge threshold** — From 5% → 10% → 15% → 20%. Fewer trades, higher conviction. This is the single biggest lever.
2. **Filter by market type** — If sports arb wins 97% but weather arb only 85%, drop weather. Keep only the profitable categories.
3. **Require multi-source agreement** — Only trade when 2+ truth sources confirm the same direction (e.g., sportsbook + TimesFM both agree).
4. **Filter by liquidity** — Only trade markets with >$50K volume. Thin markets have unreliable prices.
5. **Filter by time-to-resolution** — Markets resolving in <24h are more predictable. Narrow the window.
6. **Adjust Kelly fraction** — Smaller positions = smaller losses on the losers.
7. **Add time-of-day filters** — If losses cluster at certain times, avoid those windows.

**Algorithm changes (when parameter tuning plateaus):**

8. **Truth source algorithm swap** — Try different sportsbook consensus methods:
   - Mean vs median vs weighted-by-sharp-books (Pinnacle, Betfair carry more weight)
   - Bayesian probability fusion (weight by each book's historical accuracy)
   - Remove outlier books that consistently disagree with consensus
9. **Edge calculation method** — Try alternatives to simple subtraction:
   - Log-odds ratio (better for extreme probabilities near 0 or 1)
   - Bayesian edge with prior from historical accuracy
   - Confidence-weighted edge (multiply edge by truth source confidence)
10. **Entry timing algorithm** — Instead of "trade immediately when edge found":
    - Wait for edge to be increasing (momentum confirmation)
    - Wait for volume confirmation (others entering same direction)
    - Use TimesFM to predict if spread will widen further before entering
11. **Portfolio optimization** — Instead of independent trades:
    - Kelly criterion across the full portfolio (correlated positions reduce sizing)
    - Maximum diversification (cap exposure per sport/league/market type)
    - Anti-correlation seeking (find hedged combinations)
12. **ML-based trade filtering** — Train a classifier on historical trades:
    - Features: edge size, time-to-resolution, liquidity, truth source confidence, market type
    - Label: win/loss
    - Only trade when classifier predicts >95% win probability
    - Use XGBoost or similar (fast, interpretable, works on small datasets)
13. **Ensemble of strategies** — Run multiple algorithm variants in parallel:
    - Strategy A: pure sportsbook consensus
    - Strategy B: weighted by sharp books
    - Strategy C: ML-filtered
    - Only trade when 2+ strategies agree (consensus voting)

**Data source changes (expand what we feed in):**

14. **Add more sportsbooks** — Betfair Exchange API, Pinnacle (sharpest book), Asian books. Weight sharp books higher in consensus.
15. **Add contextual data** — Injury reports, weather conditions (outdoor sports), referee/umpire tendencies, home/away records.
16. **Add social signals** — X/Twitter sentiment velocity on teams/events. Sudden sentiment shift = information the market hasn't priced.
17. **Upgrade weather models** — Add ECMWF ensemble (European, often beats GFS), HRRR, NAM. Ensemble-of-ensembles for higher accuracy.

**Market structure changes (change HOW we trade):**

18. **Switch to maker orders** — Place limit orders instead of market. 0% fee + maker rebates vs taker fee. Requires order book depth analysis.
19. **Lifecycle timing** — Trade at market open (max volatility, biggest mispricings) OR near resolution (truth is clearest). Test both.
20. **Dynamic position management** — Exit early if edge drops below threshold. Scale in if edge widens. Hedge large positions on other platforms.

**Behavioral/temporal changes:**

21. **Time-of-day filters** — Morning vs evening, weekday vs weekend. Find when markets are least efficient.
22. **Adversarial adaptation** — Track competing bots. Avoid crowded markets. Find markets where we're the only bot (highest edge).
23. **Market creation** — Create our own Polymarket markets where we know the truth. Market make on our own markets for guaranteed spread.

**Meta-optimization:**

24. **Meta-learning** — Track which levers actually improved win rate. Build a model of optimization effectiveness. The outer loop should LEARN how to optimize, not just try things in sequence.
25. **Strategy ensemble with voting** — Run multiple algorithm variants in parallel. Only trade when 2+ strategies agree (consensus voting).

If after ALL of the above the target STILL can't be hit:
- Try completely different market categories (politics, crypto, entertainment)
- Try completely different platforms (Kalshi-only, Manifold for testing)
- **Final fallback:** If no combination works → strategy is not viable → kill it, pivot to demand forecasting SaaS (already validated at 55.9% lift)

**Key insight:** 95% win rate is achieved through **selectivity**, not prediction accuracy. The $3.3M bot had 98% win rate by only trading when the edge was overwhelming. Fewer trades, higher conviction.

### Go/No-Go Criteria
| Metric | Required | Why |
|--------|----------|-----|
| Win rate | ≥ 95% | Matches documented top bots |
| Sharpe ratio | ≥ 2.0 | Strong risk-adjusted returns |
| Max drawdown | ≤ 15% | Survivable on $500 capital |
| Profit factor | ≥ 3.0 | Wins >> losses |
| Out-of-sample pass | Yes | Not overfitting to history |
| Walk-forward stable | Yes | Works across time periods |

If ANY metric fails after optimization → do NOT deploy. Pivot to demand forecasting SaaS instead.

## Risk Controls

| Control | Limit |
|---------|-------|
| Max per trade | $100 |
| Max open positions | 20 |
| Daily loss limit | $150 → halt 24h |
| Weekly loss limit | $400 → halt 1 week |
| Max drawdown | 30% of bankroll → full shutdown |
| Min edge threshold | Determined by optimization (likely 10-20%) |
| Stale price protection | Skip if data >30s old |
| Kill switch | Manual + automatic on drawdown |

## Deployment

Claude Code scheduled agent running the main loop. Falls back to VPS ($5/mo Hetzner) if scheduled agents can't maintain state/uptime.

## Dependencies

```
py-clob-client     # Polymarket CLOB API
web3               # Ethereum signing for Polymarket
requests           # Kalshi REST, The Odds API, Open-Meteo
openmeteo-requests # Weather forecast helper
pandas numpy       # Data handling
timesfm            # Odds forecasting (Phase 5)
apscheduler        # Job scheduling
python-telegram-bot # Alerts
sqlite3            # Trade database (stdlib)
```

All external APIs are free or cheap ($20/mo for The Odds API).

## Build Phases

### Phase 1: Historical Data + Backtest Engine
- Collect 5 years of sportsbook odds, Polymarket prices, weather data, outcomes
- Build simulation engine that replays bot logic
- Validate on known profitable periods

### Phase 2: Auto-Research Optimization
- Use `0-autoresearch-skill` to iterate toward 95% win rate
- Parameter sweep: edge thresholds, market filters, truth source weights
- Walk-forward validation across multiple time periods

### Phase 3: Paper Trading
- Deploy with $1-5 real trades (minimum Polymarket order size)
- Compare live results to backtested expectations
- Run for 2-4 weeks minimum

### Phase 4: Live Trading
- Scale to full Kelly sizing ($200-500 capital)
- All risk controls active
- Daily Telegram reports

### Phase 5: TimesFM Enhancement
- Fine-tune TimesFM on historical odds time series
- Add odds trajectory prediction for entry timing
- A/B test: does TimesFM improve win rate beyond pure truth comparison?

## Revenue Projections (Conservative)

Assuming 95% win rate with selective trading:
- 3-5 high-confidence trades per day
- Average profit per winning trade: $10-30
- Average loss per losing trade: $50-100
- Daily expected: (0.95 × $20 × 4) - (0.05 × $75 × 4) = $76 - $15 = $61/day
- Monthly: ~$1,800/month on $500 capital
- Even at 50% of projection: ~$900/month

## Sources

- [Claude bot $1→$3.3M on Polymarket](https://finbold.com/claude-ai-powered-trading-bot-turns-1-into-3-3-million-on-polymarket/)
- [$40M extracted by arb bots](https://finance.yahoo.com/news/arbitrage-bots-dominate-polymarket-millions-100000888.html)
- [4 strategies that work in 2026](https://medium.com/illumination/beyond-simple-arbitrage-4-polymarket-strategies-bots-actually-profit-from-in-2026-ddacc92c5b4f)
- [Weather bot $1K→$79K](https://blog.devgenius.io/found-the-weather-trading-bots-quietly-making-24-000-on-polymarket-and-built-one-myself-for-free-120bd34d6f09)
- [Correlation arbitrage 70-80% win rate](https://arxiv.org/abs/2508.03474)

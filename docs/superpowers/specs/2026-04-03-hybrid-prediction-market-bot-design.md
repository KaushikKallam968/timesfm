# Hybrid Prediction Market Bot — Design Spec

## Problem

The current TruthArbitrageEngine takes directional bets against Polymarket using noisy truth sources. Realistic backtest: 24 trades in 3 years, 9.8% return, 61% win rate. Spreads on most Polymarket markets are 2-10% which eats the edge. The bot sits idle waiting for high-confidence signals that rarely come.

We need a system that: (a) generates consistent revenue from market making, (b) adds alpha from directional bets when confidence is high, (c) auto-researches and self-optimizes to find and maintain edges, and (d) scales through capital tiers as it proves itself.

## Solution: The Hybrid Machine

Three revenue streams in one system:

1. **Market Making** (baseline income): Post two-sided quotes, collect the bid-ask spread. No directional bet needed. Revenue even when truth engine has no signal.

2. **Directional Overlay** (alpha): When truth engine confidence > 90% and edge > 15%, skew quotes toward the correct side. Only trade within 3 days of resolution where accuracy is highest. Target: 95% win rate on directional trades.

3. **Structural Arbitrage** (risk-free): Detect multi-outcome markets where probabilities don't sum to 1. Aggressively trade the mispricing. Deterministic profit.

Plus: an **autoresearch engine** that continuously experiments with parameters, discovers new edges, and adapts to market changes.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    HYBRID TRADING ENGINE                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Market Maker  │  │ Truth Engine │  │ Autoresearch     │   │
│  │ (spread mgmt) │  │ (5+ sources) │  │ (2-loop optimizer│   │
│  └──────┬───────┘  └──────┬───────┘  │  inner: params   │   │
│         │                 │          │  outer: strategy) │   │
│         v                 v          └────────┬─────────┘   │
│  ┌────────────────────────────┐               │             │
│  │       Quote Engine          │<──────────────┘             │
│  │  neutral MM + directional   │  (autoresearch feeds        │
│  │  skew + correlation arb     │   optimized params)         │
│  └─────────────┬──────────────┘                              │
│                │                                             │
│                v                                             │
│  ┌────────────────────────────┐                              │
│  │      Risk Manager           │                              │
│  │  inventory limits, drawdown │                              │
│  │  tier-based position sizing │                              │
│  └─────────────┬──────────────┘                              │
│                │                                             │
│                v                                             │
│  ┌────────────────────────────┐                              │
│  │     Execution Layer         │                              │
│  │  Polymarket CLOB orders     │                              │
│  └────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

## Module Specifications

### 1. Market Maker (`bot/mm/market_maker.py`)

Posts two-sided quotes (bid + ask) on selected markets.

**Spread calculation:**
- `base_spread` = max(0.02, volatility_estimate * 2)
- Widen when: inventory imbalanced, close to resolution, low liquidity
- Tighten when: balanced inventory, high volume, far from resolution
- Minimum spread = 2 cents (below this, fees eat the profit)

**Inventory management:**
- Track `net_inventory` per market (positive = long YES, negative = long NO)
- When |net_inventory| > `max_inventory`: widen the overweight side by 50%, tighten the underweight side by 25%
- When |net_inventory| > `panic_threshold`: place aggressive rebalancing orders on the overweight side
- At resolution: all inventory settles at 0 or 1. This is the main risk.

**Market selection criteria:**
- Daily volume > $10,000
- At least 5 active price levels on both sides
- Spread currently > 2 cents (room for us to provide tighter quotes)
- Not within 1 hour of resolution (too volatile for MM)

### 2. Truth Engine (`bot/truth/` — enhanced)

Aggregates multiple sources per category. Each source gets a calibrated accuracy weight.

**Sports:**
- Pinnacle closing line (sharpest book)
- Betfair exchange price (market-driven, no bookmaker margin)
- ESPN/FiveThirtyEight model predictions
- Consensus of 3+ sportsbooks

**Politics:**
- 538/Silver Bulletin polling averages
- Kalshi prices (cross-platform signal)
- Metaculus community predictions
- PredictIt prices (if available)

**Weather:**
- GFS ensemble (current)
- ECMWF ensemble (European model, often more accurate)
- NAM model (short-range, US only)
- Historical accuracy calibration per city

**Crypto:**
- Deribit options implied volatility
- Funding rate signals
- On-chain flow data

**Confidence scoring:**
- Each source has a tracked accuracy score (updated after each resolution)
- Consensus = weighted average of sources (weight = historical accuracy)
- Confidence = agreement level (all sources agree = 95%, split = 60%)
- Only trigger directional overlay when confidence > 90% AND edge > 15%

### 3. Quote Engine (`bot/mm/quote_engine.py`)

Merges market maker quotes with truth engine signal.

**Three modes:**
1. **Neutral** (default): symmetric bid/ask around mid-price. Pure spread collection.
2. **Skewed** (truth engine has signal): lean quotes toward the predicted correct side. Buy the underpriced side more aggressively, sell reluctantly.
3. **Aggressive** (correlation arb detected): place limit orders to capture the full mispricing. No spread — just arb the violation.

**Skew formula:**
```
If truth_prob > market_price + min_edge:
  yes_bid = market_price - (spread/4)    # tighter (attract YES buys)
  yes_ask = market_price + (spread * 1.5) # wider (discourage YES sells)
  
If truth_prob < market_price - min_edge:
  yes_bid = market_price - (spread * 1.5) # wider
  yes_ask = market_price + (spread/4)     # tighter
```

### 4. Autoresearch Engine (Karpathy-style autonomous loop)

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The agent modifies a single strategy file, runs a backtest, checks if the metric improved, keeps or discards, and repeats forever.

**Core pattern:**
```
LOOP FOREVER:
  1. Read current strategy state (strategy.py + results.tsv)
  2. Propose a modification (parameter tweak, new logic, different market selection)
  3. git commit the change
  4. Run backtest: python -m bot.backtest.evaluate > run.log 2>&1
  5. Extract metrics: grep "^win_rate:\|^sharpe:\|^return:" run.log
  6. If improved: KEEP (advance branch)
  7. If worse: DISCARD (git reset --hard to previous commit)
  8. Log result to results.tsv
  9. NEVER STOP. The human is asleep.
```

**Three files that matter (mirroring Karpathy's design):**

| File | Role | Who edits |
|------|------|-----------|
| `bot/research/program.md` | Agent instructions ("the skill") | Human |
| `bot/research/strategy.py` | The single file the agent modifies. Contains: spread width, edge thresholds, Kelly fraction, market selection rules, truth source weights, skew formula. Everything tunable is here. | Agent |
| `bot/research/evaluate.py` | Fixed evaluation harness. Loads historical data, runs the strategy from strategy.py, prints metrics in a parseable format. Read-only. | Nobody (fixed) |

**What the agent CAN modify** (in strategy.py):
- Spread width and its dynamic adjustment formula
- Edge threshold per category
- Kelly fraction and position sizing
- Market selection filters (volume, liquidity, category, days-to-resolution)
- Truth source weighting (which sources, how to combine)
- Quote skew aggressiveness
- Inventory rebalancing thresholds
- Any new logic: regime detection, time-of-day filters, correlation checks

**What the agent CANNOT modify:**
- evaluate.py (fixed evaluation harness)
- The historical data
- The backtest simulation engine (proper_backtest.py, mm_backtest.py)

**Metrics (the agent optimizes all simultaneously):**
```
win_rate:       0.9500    # PRIMARY: must be >= 95%
sharpe:         3.20      # Higher is better
total_return:   0.1850    # 18.5% return
max_drawdown:   0.0800    # 8% max drawdown
num_trades:     45        # More is better (shows the edge is broad)
profit_factor:  4.20      # Gross profit / gross loss
```

The primary gate is `win_rate >= 0.95`. Among configurations that pass the gate, the agent optimizes for highest Sharpe.

**results.tsv format:**
```
commit	win_rate	sharpe	return	drawdown	trades	status	description
a1b2c3d	0.9200	2.80	0.1200	0.1000	38	keep	baseline
b2c3d4e	0.9500	3.10	0.1500	0.0800	42	keep	tighten edge to 12%
c3d4e5f	0.8800	2.50	0.0900	0.1500	55	discard	loosen edge to 5% (too many bad trades)
d4e5f6g	0.0000	0.00	0.0000	0.0000	0	crash	divide by zero in skew formula
```

**Experiment speed:**
- Karpathy's: ~5 min per experiment (GPU training)
- Ours: ~5-30 seconds per experiment (backtest on 134k records)
- We can run **100-700 experiments per hour**
- Overnight (8 hours): **800-5,600 experiments**

**Branch management:**
- Each autoresearch session runs on `autoresearch/{tag}` branch
- Best result gets merged back to master when the human reviews
- results.tsv stays untracked (like Karpathy's design)

**State persistence:**
```
bot/research/
  program.md          # Human-written agent instructions
  strategy.py         # Agent-modified strategy (THE file)
  evaluate.py         # Fixed evaluation harness (read-only)
  results.tsv         # Experiment log (untracked by git)
  best_config.json    # Best-so-far configuration snapshot
```

**How the agent decides what to try:**
1. Start with baseline (current params)
2. Systematic sweeps: vary one parameter at a time, find the best value
3. Combination experiments: combine the best individual changes
4. Creative experiments: try new logic (time-of-day filters, volatility scaling, category-specific rules)
5. If stuck: read the evaluation logs, look for patterns in what works/fails, try more radical changes
6. Never repeat a failed experiment

### 5. Risk Manager (`bot/core/risk.py` — enhanced)

**Tier-based limits:**

| Parameter | Tier 0 (sim) | Tier 1 ($500) | Tier 2 ($2-5k) | Tier 3 ($10k+) |
|-----------|-------------|---------------|----------------|----------------|
| Max trade | $0 | $25 | $100 | $500 |
| Max positions | unlimited | 5 | 15 | 30 |
| Daily loss limit | N/A | $25 | $100 | $500 |
| Max drawdown | N/A | 15% | 20% | 25% |
| Markets made | unlimited | 1-2 | 5-10 | all liquid |
| Kelly fraction | 0.25 | 0.15 | 0.20 | 0.25 |

**Inventory risk controls:**
- Max net inventory per market: 20% of bankroll (Tier 1), 10% (Tier 2+)
- Auto-hedge: if correlated markets exist, offset inventory risk
- Pre-resolution exit: flatten inventory 1 hour before resolution if possible

## Validation Gate (Tier 0)

Must pass on historical simulation before ANY real money:

**Market Making Simulation:**
1. Load 134k price history records
2. For each market: simulate posting quotes daily at `mid +/- spread/2`
3. Estimate fills from daily price movement (if price crosses our quote level, we got filled)
4. Track inventory per market, settle at known outcome
5. Compute: net P&L per market = spread earned - inventory gain/loss - fees

**Directional Simulation:**
1. Load same data
2. Use simulated truth source at 75-85% accuracy (realistic range)
3. Only trade when confidence > 90% AND edge > 15% AND days_to_res < 3
4. Quarter-Kelly sizing with 2% fees and 0.5% slippage

**Combined Simulation:**
1. Run both together: MM revenue + directional P&L
2. Multiple random seeds (10+) for truth source noise

**Gate criteria (must ALL pass):**
- 95%+ of markets made are net profitable after settlement
- Overall win rate on directional trades > 95%
- Combined Sharpe > 2.0
- Max drawdown < 15%
- Profitable on 8+ of 10 random seeds
- Monthly return > 5% (annualized > 60%)

**If gate fails:** autoresearch inner loop runs parameter sweeps until a passing configuration is found, or concludes the strategy isn't viable for the current market regime.

## Capital Tier Progression

```
Tier 0 (Simulation) ─── pass gate ───> Tier 1 ($500)
     │                                      │
     │  fail: autoresearch                  │  30 days profitable
     │  optimizes or kills                  │
     v                                      v
  STOP                              Tier 2 ($2-5k)
                                           │
                                           │  3 months profitable
                                           │  Sharpe > 2.0
                                           v
                                    Tier 3 ($10k+)
```

Each tier upgrade requires:
- Previous tier's performance sustained for the minimum period
- Autoresearch outer loop confirms edge is stable, not degrading
- No manual intervention needed for the required period

## Files to Create

| File | Purpose |
|------|---------|
| **Autoresearch (Karpathy pattern)** | |
| `bot/research/program.md` | Human-written agent instructions (the "skill") |
| `bot/research/strategy.py` | THE file the agent modifies: all tunable strategy params + logic |
| `bot/research/evaluate.py` | Fixed evaluation harness (read-only, loads data, runs backtest, prints metrics) |
| **Market Making** | |
| `bot/mm/__init__.py` | Market making package |
| `bot/mm/market_maker.py` | Core spread posting + inventory management |
| `bot/mm/quote_engine.py` | Merge MM quotes + truth signal + correlation arb |
| `bot/mm/spread_model.py` | Dynamic spread calculation based on volatility/liquidity |
| `bot/mm/inventory.py` | Per-market inventory tracking + rebalancing |
| **Backtesting** | |
| `bot/backtest/mm_backtest.py` | Market making historical simulation |
| `bot/backtest/hybrid_backtest.py` | Combined MM + directional simulation |
| `bot/backtest/validation_gate.py` | Tier 0 gate checker (runs all simulations, pass/fail) |
| **Infrastructure** | |
| `bot/core/tiers.py` | Tier system + promotion logic |
| `bot/truth/aggregator.py` | Multi-source truth aggregation with accuracy weighting |

## Files to Modify

| File | Change |
|------|--------|
| `bot/core/config.py` | Add MM params, tier configs, autoresearch intervals |
| `bot/core/risk.py` | Add inventory limits, tier-based position sizing |
| `bot/run.py` | Integrate MM + quote engine into main loop |
| `bot/truth/sports.py` | Add multi-sportsbook aggregation |
| `bot/truth/weather.py` | Add ECMWF + NAM models |

## Existing Code to Reuse

| Existing | Reuse For |
|----------|-----------|
| `bot/backtest/proper_backtest.py` → `run_backtest()` | Directional component simulation |
| `bot/backtest/data/loader.py` | All data loading |
| `bot/truth/sports.py` → odds calculation | Consensus probability math |
| `bot/truth/weather.py` → GFS forecast | Weather truth source |
| `bot/truth/correlation.py` → sum/subset rules | Structural arb detection |
| `bot/execution/kelly.py` | Kelly sizing formula |
| `bot/core/risk.py` → loss limits | Risk management framework |
| `bot/backtest/realistic_backtest.py` → `simulate_truth_signal()` | Noisy truth source model |

## Verification

1. `python -m bot.backtest.validation_gate` — runs full Tier 0 simulation, prints pass/fail with metrics
2. `python -m bot.research.evaluate` — runs single evaluation of current strategy.py, prints metrics
3. **Autoresearch loop**: Point Claude Code at `bot/research/program.md` and let it run autonomously. It modifies strategy.py, runs evaluate.py, keeps/discards based on win_rate >= 95%. Check results.tsv in the morning.
4. All gate criteria met on historical data before any deployment discussion

**To kick off autoresearch:**
```
cd ~/timesfm
git checkout -b autoresearch/apr3
# Then tell the agent:
# "Read bot/research/program.md and kick off a new experiment session."
```

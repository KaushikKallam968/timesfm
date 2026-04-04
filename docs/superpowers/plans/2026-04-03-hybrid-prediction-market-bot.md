# Hybrid Prediction Market Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid market-making + directional trading bot for Polymarket with Karpathy-style autoresearch that self-optimizes to 95% win rate on historical data before any live deployment.

**Architecture:** Three layers: (1) strategy.py contains all tunable parameters and logic in one file, (2) evaluate.py is the fixed backtest harness that scores any strategy configuration, (3) program.md tells the autonomous agent how to iterate. On top of this: market making simulation, directional overlay, and a validation gate.

**Tech Stack:** Python 3.11+, numpy, pandas, existing `bot/backtest/data/loader.py` for data.

---

## Phase 1: Autoresearch Foundation (the loop)

Build the three Karpathy-pattern files first. This is the engine that optimizes everything else.

### Task 1: Strategy file (the ONE file the agent modifies)

**Files:**
- Create: `bot/research/strategy.py`
- Create: `bot/research/__init__.py`

- [ ] **Step 1: Create package**

```bash
mkdir -p bot/research
touch bot/research/__init__.py
```

- [ ] **Step 2: Write strategy.py with all tunable parameters and logic**

```python
"""The ONE file the autoresearch agent modifies.

Contains every tunable parameter and strategy decision.
The evaluate.py harness imports this and scores it.
"""

# === MARKET MAKING PARAMETERS ===
MM_BASE_SPREAD = 0.04          # 4 cents base spread
MM_MIN_SPREAD = 0.02           # never go below 2 cents
MM_INVENTORY_SKEW = 0.5        # how much to skew when inventory imbalanced (0-1)
MM_MAX_INVENTORY_PCT = 0.20    # max 20% of bankroll in one market
MM_PANIC_INVENTORY_PCT = 0.30  # aggressive rebalance above this

# === DIRECTIONAL PARAMETERS ===
DIR_MIN_EDGE = 0.15            # 15% minimum edge for directional bets
DIR_MIN_CONFIDENCE = 0.90      # 90% confidence required
DIR_MAX_DAYS_TO_RES = 3        # only trade within 3 days of resolution
DIR_TRUTH_ACCURACY = 0.80      # assumed truth source accuracy

# === KELLY SIZING ===
KELLY_FRACTION = 0.25          # quarter-Kelly
MAX_TRADE_SIZE = 100.0         # max $100 per trade
MAX_PCT_PER_TRADE = 0.05       # max 5% of bankroll per trade

# === FEES AND SLIPPAGE ===
FEE_RATE = 0.02                # 2% round-trip
SLIPPAGE = 0.005               # 0.5% per side

# === MARKET SELECTION ===
MIN_VOLUME = 1000              # minimum $1000 volume
EXCLUDED_CATEGORIES = []       # e.g. ["esports"] to skip categories
MAX_POSITIONS = 20             # max simultaneous positions

# === CATEGORY-SPECIFIC OVERRIDES ===
CATEGORY_EDGE_OVERRIDES = {
    # "politics": 0.12,  # lower threshold for politics (more data)
    # "crypto": 0.20,    # higher threshold for crypto (more volatile)
}

# === STARTING CAPITAL ===
STARTING_BANKROLL = 10000.0


def should_trade_market(record):
    """Filter: should we consider this market at all?

    Args:
        record: dict with keys: market_id, market_price, category, volume,
                days_to_resolution, truth_probability

    Returns:
        bool
    """
    mp = record.get("market_price", 0)
    if mp <= 0.03 or mp >= 0.97:
        return False

    vol = record.get("volume", 0)
    if vol and float(vol) < MIN_VOLUME:
        return False

    cat = record.get("category", "")
    if cat in EXCLUDED_CATEGORIES:
        return False

    return True


def compute_edge(record, truth_prob):
    """Compute the edge between our truth and the market price.

    Returns:
        (side, edge) where side is "YES" or "NO" and edge > 0
        Returns (None, 0) if no tradeable edge.
    """
    mp = record.get("market_price", 0)
    days = record.get("days_to_resolution")

    # Category-specific edge threshold
    cat = record.get("category", "")
    min_edge = CATEGORY_EDGE_OVERRIDES.get(cat, DIR_MIN_EDGE)

    # Only take directional if close to resolution
    if days is not None and days > DIR_MAX_DAYS_TO_RES:
        return None, 0

    edge_yes = truth_prob - mp
    edge_no = (1 - truth_prob) - (1 - mp)  # same magnitude, opposite direction

    if edge_yes > min_edge:
        return "YES", edge_yes
    elif edge_no > min_edge:
        return "NO", edge_no

    return None, 0


def compute_mm_quotes(mid_price, net_inventory, bankroll):
    """Compute market making bid/ask quotes.

    Args:
        mid_price: current market mid price
        net_inventory: our net position (positive = long YES)
        bankroll: current bankroll

    Returns:
        (bid, ask) prices for YES token
    """
    spread = max(MM_BASE_SPREAD, abs(net_inventory / max(bankroll, 1)) * 0.10)
    spread = max(spread, MM_MIN_SPREAD)

    half = spread / 2

    # Skew based on inventory
    inv_ratio = net_inventory / max(bankroll, 1)
    skew = inv_ratio * MM_INVENTORY_SKEW * spread

    bid = mid_price - half - skew   # if long, lower bid (buy less)
    ask = mid_price + half - skew   # if long, lower ask (sell more)

    bid = max(0.01, min(0.98, bid))
    ask = max(0.02, min(0.99, ask))

    if ask <= bid:
        ask = bid + MM_MIN_SPREAD

    return round(bid, 4), round(ask, 4)


def kelly_size(edge, entry_price, bankroll):
    """Compute position size using Kelly criterion."""
    if edge <= 0 or entry_price <= 0.01 or entry_price >= 0.99:
        return 0

    payout = (1.0 / entry_price) - 1.0
    p = min(0.99, max(0.01, entry_price + edge))
    f = (p * (payout + 1) - 1) / payout
    f = max(0, f)

    size = bankroll * f * KELLY_FRACTION
    size = min(size, MAX_TRADE_SIZE)
    size = min(size, bankroll * MAX_PCT_PER_TRADE)

    return round(size, 2)
```

- [ ] **Step 3: Commit**

```bash
git add bot/research/__init__.py bot/research/strategy.py
git commit -m "feat: add strategy.py - the single file autoresearch modifies"
```

### Task 2: Evaluation harness (fixed, read-only after creation)

**Files:**
- Create: `bot/research/evaluate.py`

- [ ] **Step 1: Write the evaluation harness**

This loads data, imports strategy.py, runs the hybrid backtest, and prints metrics in a grep-parseable format (matching Karpathy's output pattern).

```python
"""Fixed evaluation harness for autoresearch.

DO NOT MODIFY THIS FILE. The agent modifies strategy.py only.
This file loads data, runs the strategy, and prints metrics.

Usage: python -m bot.research.evaluate
Output: key:value pairs, one per line, grep-parseable.
"""

import json
import os
import sys
import random
import numpy as np
from collections import defaultdict

# Add data loader to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backtest", "data"))
from loader import load_price_histories

# Import the strategy (THE file the agent modifies)
from bot.research.strategy import (
    STARTING_BANKROLL, FEE_RATE, SLIPPAGE, MAX_POSITIONS,
    DIR_TRUTH_ACCURACY, KELLY_FRACTION, MAX_TRADE_SIZE,
    should_trade_market, compute_edge, kelly_size,
)


def simulate_truth(actual_outcome, accuracy, days_to_res):
    """Simulate a noisy truth source. Same logic as realistic_backtest.py."""
    if days_to_res is not None:
        if days_to_res <= 1:
            eff = min(accuracy + 0.10, 0.98)
        elif days_to_res <= 3:
            eff = min(accuracy + 0.05, 0.95)
        elif days_to_res <= 7:
            eff = accuracy
        elif days_to_res <= 30:
            eff = max(accuracy - 0.05, 0.55)
        else:
            eff = max(accuracy - 0.10, 0.52)
    else:
        eff = accuracy

    correct = random.random() < eff
    if correct:
        return random.uniform(0.55, 0.95) if actual_outcome == 1.0 else random.uniform(0.05, 0.45)
    else:
        return random.uniform(0.10, 0.45) if actual_outcome == 1.0 else random.uniform(0.55, 0.90)


def run_evaluation(seed=42):
    """Run the full hybrid backtest and return metrics."""
    random.seed(seed)
    np.random.seed(seed)

    records = load_price_histories()
    usable = [r for r in records
              if r.get("truth_probability") in (0.0, 1.0)
              and r.get("timestamp")
              and should_trade_market(r)]

    # Sort chronologically
    usable.sort(key=lambda r: r.get("timestamp", ""))

    # Group by market for one-trade-per-market
    by_market = defaultdict(list)
    for r in usable:
        by_market[r["market_id"]].append(r)

    bankroll = STARTING_BANKROLL
    trades = []
    peak = bankroll
    max_dd = 0

    for mid, market_records in by_market.items():
        if bankroll < 10:
            break

        actual = market_records[0]["truth_probability"]  # 0 or 1
        truth_prob = None
        best_record = None
        best_edge = 0
        best_side = None

        for r in market_records:
            tp = simulate_truth(actual, DIR_TRUTH_ACCURACY, r.get("days_to_resolution"))
            side, edge = compute_edge(r, tp)
            if side and edge > best_edge:
                best_edge = edge
                best_side = side
                best_record = r
                truth_prob = tp

        if not best_side or not best_record:
            continue

        mp = best_record["market_price"]
        entry_price = (mp + SLIPPAGE) if best_side == "YES" else ((1 - mp) + SLIPPAGE)
        entry_price = min(entry_price, 0.98)

        size = kelly_size(best_edge, entry_price, bankroll)
        if size < 5:
            continue

        fees = size * FEE_RATE
        won = (best_side == "YES" and actual == 1.0) or (best_side == "NO" and actual == 0.0)

        if won:
            pnl = (size / entry_price) - size - fees
        else:
            pnl = -size - fees

        bankroll += pnl
        bankroll = max(bankroll, 0)
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        trades.append({"pnl": pnl, "won": won, "edge": best_edge,
                        "category": best_record.get("category", ""),
                        "timestamp": best_record.get("timestamp", "")})

    return trades, bankroll, max_dd


def main():
    # Run across multiple seeds for robustness
    all_returns = []
    all_win_rates = []
    all_sharpes = []
    all_drawdowns = []
    all_trades = []

    for seed in [42, 123, 456, 789, 1337]:
        trades, final, max_dd = run_evaluation(seed)
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for t in trades if t["won"])
        ret = (final - STARTING_BANKROLL) / STARTING_BANKROLL
        wr = wins / len(trades) if trades else 0

        daily = defaultdict(float)
        for t in trades:
            daily[t["timestamp"][:10]] += t["pnl"]
        dv = list(daily.values()) or [0]
        sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(365)) if np.std(dv) > 0 else 0

        all_returns.append(ret)
        all_win_rates.append(wr)
        all_sharpes.append(float(sharpe))
        all_drawdowns.append(max_dd)
        all_trades.append(len(trades))

    # Print grep-parseable metrics (like Karpathy's val_bpb output)
    avg_wr = np.mean(all_win_rates)
    avg_sharpe = np.mean(all_sharpes)
    avg_return = np.mean(all_returns)
    avg_dd = np.mean(all_drawdowns)
    avg_trades = np.mean(all_trades)
    worst_wr = min(all_win_rates)
    seeds_profitable = sum(1 for r in all_returns if r > 0)

    gross_profit = sum(sum(t["pnl"] for t in run_evaluation(42)[0] if t["won"]) for _ in [0])
    gross_loss = abs(sum(sum(t["pnl"] for t in run_evaluation(42)[0] if not t["won"]) for _ in [0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print("---")
    print(f"win_rate:          {avg_wr:.6f}")
    print(f"sharpe:            {avg_sharpe:.6f}")
    print(f"total_return:      {avg_return:.6f}")
    print(f"max_drawdown:      {avg_dd:.6f}")
    print(f"num_trades:        {avg_trades:.1f}")
    print(f"profit_factor:     {pf:.6f}")
    print(f"worst_seed_wr:     {worst_wr:.6f}")
    print(f"seeds_profitable:  {seeds_profitable}")
    print(f"gate_passed:       {'YES' if avg_wr >= 0.95 else 'NO'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluate to get baseline metrics**

```bash
cd ~/timesfm && PYTHONIOENCODING=utf-8 python -m bot.research.evaluate
```

Expected output: metrics printed in `key: value` format. This is the baseline the agent will try to beat.

- [ ] **Step 3: Commit**

```bash
git add bot/research/evaluate.py
git commit -m "feat: add fixed evaluation harness for autoresearch"
```

### Task 3: program.md (agent instructions)

**Files:**
- Create: `bot/research/program.md`

- [ ] **Step 1: Write program.md**

```markdown
# Prediction Market Autoresearch

The idea: give an AI agent a prediction market trading strategy and let it
experiment autonomously. It modifies the strategy, runs a backtest (~10 seconds),
checks if the result improved, keeps or discards, and repeats.

## Setup

1. **Agree on a run tag**: e.g. `apr3`. Branch: `autoresearch/{tag}`.
2. **Create branch**: `git checkout -b autoresearch/{tag}`
3. **Read the in-scope files**:
   - `bot/research/strategy.py` — THE file you modify. All tunable params and logic.
   - `bot/research/evaluate.py` — Fixed evaluation harness. Read-only.
   - `bot/backtest/data/loader.py` — Data loading. Read-only.
4. **Initialize results.tsv**: Create with just the header row.
5. **Confirm and go**.

## Experimentation

Each experiment runs a backtest on 134k historical records across 5 random seeds.
Takes ~10-30 seconds. You launch: `python -m bot.research.evaluate > run.log 2>&1`

**What you CAN do:**
- Modify `bot/research/strategy.py` — everything is fair game: parameters,
  formulas, filters, new logic, different Kelly sizing, category-specific rules.

**What you CANNOT do:**
- Modify `bot/research/evaluate.py` or any other file.
- Install new packages.
- Modify the historical data.

**The goal: get win_rate >= 0.95 with the highest possible Sharpe ratio.**

The `gate_passed` line in the output tells you if win_rate >= 95%. Among configs
that pass the gate, optimize for Sharpe. If you can't pass the gate, optimize
for the highest win_rate you can achieve.

Secondary goals (in priority order):
1. Highest win_rate (PRIMARY — must be >= 0.95)
2. Highest Sharpe
3. Lowest max_drawdown
4. More num_trades (broader edge, not cherry-picked)
5. All 5 seeds profitable

## Output format

The script prints metrics like:
```
---
win_rate:          0.750000
sharpe:            9.500000
total_return:      0.098000
max_drawdown:      0.201000
num_trades:        24.0
profit_factor:     2.100000
worst_seed_wr:     0.583000
seeds_profitable:  5
gate_passed:       NO
```

Extract key metrics: `grep "^win_rate:\|^sharpe:\|^gate_passed:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	win_rate	sharpe	return	drawdown	trades	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at current strategy.py and results.tsv
2. Modify strategy.py with an experimental idea
3. `git commit -am "experiment: {description}"`
4. `python -m bot.research.evaluate > run.log 2>&1`
5. `grep "^win_rate:\|^sharpe:\|^gate_passed:" run.log`
6. If empty: crash. `tail -n 50 run.log` for stack trace. Fix or skip.
7. Record in results.tsv
8. If improved (higher win_rate, or same win_rate + higher Sharpe): KEEP
9. If worse: `git reset --hard HEAD~1`
10. **NEVER STOP.** The human is asleep.

## What to try (in order)

1. **Baseline**: run as-is to establish the starting point
2. **Edge threshold sweep**: try DIR_MIN_EDGE from 0.05 to 0.30
3. **Days-to-resolution filter**: try DIR_MAX_DAYS_TO_RES from 1 to 30
4. **Truth accuracy assumption**: try DIR_TRUTH_ACCURACY from 0.60 to 0.95
5. **Kelly fraction**: try 0.10 to 0.40
6. **Category exclusions**: try excluding esports, sports, other
7. **Category-specific edges**: different thresholds per category
8. **Volume filter**: try MIN_VOLUME from 100 to 50000
9. **Combination experiments**: combine the best individual findings
10. **Creative**: add time-of-day logic, market-age filters, volatility scaling
11. **Radical**: change compute_edge formula entirely, try non-linear scoring

If you get stuck, read the backtest data structure and think about what
information you're not using. The records have: market_price, truth_probability,
days_to_resolution, category, volume, question, timestamp.
```

- [ ] **Step 2: Add results.tsv to .gitignore**

```bash
echo "bot/research/results.tsv" >> ~/timesfm/.gitignore
echo "bot/research/run.log" >> ~/timesfm/.gitignore
```

- [ ] **Step 3: Commit**

```bash
git add bot/research/program.md .gitignore
git commit -m "feat: add program.md - autoresearch agent instructions"
```

### Task 4: Validation gate

**Files:**
- Create: `bot/backtest/validation_gate.py`

- [ ] **Step 1: Write the validation gate**

```python
"""Tier 0 Validation Gate.

Runs the full evaluation and checks all gate criteria.
Must pass before any real money is deployed.

Usage: python -m bot.backtest.validation_gate
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))


def main():
    # Import evaluate's run function
    from bot.research.evaluate import run_evaluation
    import numpy as np

    print("=" * 60)
    print("TIER 0 VALIDATION GATE")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1337, 2024, 3141, 9999, 1111, 5555]
    results = []

    for seed in seeds:
        trades, final, max_dd = run_evaluation(seed)
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for t in trades if t["won"])
        ret = (final - 10000.0) / 10000.0
        wr = wins / len(trades) if trades else 0
        results.append({"seed": seed, "return": ret, "win_rate": wr,
                         "max_dd": max_dd, "trades": len(trades), "final": final})
        print(f"  Seed {seed}: {len(trades)} trades, {wr:.0%} WR, {ret:.1%} return, {max_dd:.1%} DD")

    # Gate checks
    avg_wr = np.mean([r["win_rate"] for r in results])
    avg_sharpe = 0  # simplified
    avg_dd = np.mean([r["max_dd"] for r in results])
    seeds_profitable = sum(1 for r in results if r["return"] > 0)
    avg_return = np.mean([r["return"] for r in results])

    print(f"\n--- GATE CRITERIA ---")
    checks = {
        "Win rate >= 95%": avg_wr >= 0.95,
        "Max drawdown < 15%": avg_dd < 0.15,
        "Profitable on 8+/10 seeds": seeds_profitable >= 8,
        "Monthly return > 5%": avg_return > 0.05,
    }

    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("GATE: PASSED. Strategy is validated for Tier 1 deployment.")
    else:
        print("GATE: FAILED. Run autoresearch to optimize, or adjust strategy.py.")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the gate**

```bash
cd ~/timesfm && PYTHONIOENCODING=utf-8 python -m bot.backtest.validation_gate
```

Expected: FAIL on first run (baseline won't hit 95% win rate). This is what autoresearch is for.

- [ ] **Step 3: Commit**

```bash
git add bot/backtest/validation_gate.py
git commit -m "feat: add Tier 0 validation gate"
```

### Task 5: Smoke test the full autoresearch loop

- [ ] **Step 1: Run evaluate to get baseline**

```bash
cd ~/timesfm && PYTHONIOENCODING=utf-8 python -m bot.research.evaluate
```

Record the baseline win_rate and sharpe.

- [ ] **Step 2: Manually simulate one experiment cycle**

Edit strategy.py: change `DIR_MIN_EDGE = 0.15` to `DIR_MIN_EDGE = 0.10`. Then:

```bash
cd ~/timesfm && git commit -am "experiment: lower edge threshold to 10%"
PYTHONIOENCODING=utf-8 python -m bot.research.evaluate > bot/research/run.log 2>&1
grep "^win_rate:\|^sharpe:\|^gate_passed:" bot/research/run.log
```

If improved: keep. If worse: `git reset --hard HEAD~1`. This proves the loop works.

- [ ] **Step 3: Revert to baseline and commit clean state**

```bash
cd ~/timesfm && git checkout bot/research/strategy.py
git add -A && git commit -m "feat: autoresearch foundation complete - ready for autonomous runs"
git push origin master
```

---

## Phase 2: Run Autoresearch (autonomous)

This is NOT a coding task. This is kicking off the autonomous agent.

### Task 6: Launch autoresearch session

- [ ] **Step 1: Create branch**

```bash
cd ~/timesfm && git checkout -b autoresearch/apr3
```

- [ ] **Step 2: Initialize results.tsv**

```bash
printf "commit\twin_rate\tsharpe\treturn\tdrawdown\ttrades\tstatus\tdescription\n" > bot/research/results.tsv
```

- [ ] **Step 3: Tell the agent to start**

Open a new Claude Code session in `~/timesfm` and say:

```
Read bot/research/program.md and let's kick off a new experiment. Do the setup first.
```

The agent will: read program.md, run baseline, then loop forever modifying strategy.py and testing. Check results.tsv in the morning.

- [ ] **Step 4: After autoresearch completes (next morning)**

```bash
# Check results
cat bot/research/results.tsv
# Check best win rate achieved
sort -t$'\t' -k2 -rn bot/research/results.tsv | head -5
# Run validation gate with the optimized strategy
python -m bot.backtest.validation_gate
# If passed: merge to master
git checkout master && git merge autoresearch/apr3
git push origin master
```

---

## Phase 3: Market Making Simulation (after autoresearch finds a passing config)

### Task 7: Market making backtest engine

**Files:**
- Create: `bot/mm/__init__.py`
- Create: `bot/backtest/mm_backtest.py`

- [ ] **Step 1: Create mm package**

```bash
mkdir -p bot/mm
touch bot/mm/__init__.py
```

- [ ] **Step 2: Write MM backtest**

```python
"""Market making backtest using historical price data.

Simulates posting two-sided quotes on each market daily.
Estimates fills from price movement, tracks inventory, settles at resolution.
"""

import sys
import os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories
from bot.research.strategy import (
    MM_BASE_SPREAD, MM_MIN_SPREAD, MM_INVENTORY_SKEW,
    MM_MAX_INVENTORY_PCT, STARTING_BANKROLL, FEE_RATE,
    compute_mm_quotes,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_mm_backtest():
    records = load_price_histories()
    usable = [r for r in records
              if 0.05 < r.get("market_price", 0) < 0.95
              and r.get("truth_probability") in (0.0, 1.0)
              and r.get("timestamp")]
    usable.sort(key=lambda r: r.get("timestamp", ""))

    by_market = defaultdict(list)
    for r in usable:
        by_market[r["market_id"]].append(r)

    bankroll = STARTING_BANKROLL
    total_spread_earned = 0
    total_inventory_pnl = 0
    markets_traded = 0
    markets_profitable = 0

    for mid, market_records in by_market.items():
        if len(market_records) < 3:
            continue

        actual = market_records[0]["truth_probability"]
        net_inventory = 0  # positive = long YES
        spread_earned = 0
        inventory_cost = 0

        for i, r in enumerate(market_records):
            mp = r["market_price"]
            bid, ask = compute_mm_quotes(mp, net_inventory, bankroll)

            # Simulate fills: if price moved enough to cross our quotes
            if i > 0:
                prev_mp = market_records[i - 1]["market_price"]
                price_change = abs(mp - prev_mp)

                # If price moved more than half our spread, we likely got filled
                if price_change > (ask - bid) / 2:
                    fill_size = min(50, bankroll * 0.02)  # small fills
                    if mp > prev_mp:
                        # Price went up, our ask got hit (we sold YES)
                        spread_earned += fill_size * (ask - bid) / 2
                        net_inventory -= fill_size / ask
                    else:
                        # Price went down, our bid got hit (we bought YES)
                        spread_earned += fill_size * (ask - bid) / 2
                        net_inventory += fill_size / bid

                    spread_earned -= fill_size * FEE_RATE

        # Settle inventory at resolution
        if actual == 1.0:
            inventory_pnl = net_inventory * 1.0  # YES shares worth $1
        else:
            inventory_pnl = -abs(net_inventory) * 0.5  # rough loss estimate

        net_pnl = spread_earned + inventory_pnl
        total_spread_earned += spread_earned
        total_inventory_pnl += inventory_pnl
        markets_traded += 1
        if net_pnl > 0:
            markets_profitable += 1

    win_rate = markets_profitable / markets_traded if markets_traded > 0 else 0
    total_pnl = total_spread_earned + total_inventory_pnl

    print("--- MM BACKTEST ---")
    print(f"markets_traded:    {markets_traded}")
    print(f"markets_profitable:{markets_profitable}")
    print(f"mm_win_rate:       {win_rate:.4f}")
    print(f"spread_earned:     {total_spread_earned:.2f}")
    print(f"inventory_pnl:     {total_inventory_pnl:.2f}")
    print(f"total_pnl:         {total_pnl:.2f}")
    print(f"mm_return:         {total_pnl / STARTING_BANKROLL:.4f}")

    return {
        "markets_traded": markets_traded,
        "win_rate": win_rate,
        "spread_earned": total_spread_earned,
        "inventory_pnl": total_inventory_pnl,
        "total_pnl": total_pnl,
    }


if __name__ == "__main__":
    run_mm_backtest()
```

- [ ] **Step 3: Run MM backtest**

```bash
cd ~/timesfm && PYTHONIOENCODING=utf-8 python -m bot.backtest.mm_backtest
```

- [ ] **Step 4: Commit**

```bash
git add bot/mm/__init__.py bot/backtest/mm_backtest.py
git commit -m "feat: add market making backtest simulation"
```

### Task 8: Combined hybrid backtest

**Files:**
- Create: `bot/backtest/hybrid_backtest.py`

- [ ] **Step 1: Write hybrid backtest combining MM + directional**

```python
"""Combined hybrid backtest: market making + directional overlay.

Runs both strategies simultaneously on the same capital pool.
This is the full strategy that gets deployed in Tier 1+.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from bot.research.evaluate import run_evaluation
from bot.backtest.mm_backtest import run_mm_backtest


def main():
    print("=" * 60)
    print("HYBRID BACKTEST: Market Making + Directional")
    print("=" * 60)

    print("\n--- Directional Component ---")
    trades, final_dir, max_dd = run_evaluation(seed=42)
    dir_pnl = final_dir - 10000.0
    wins = sum(1 for t in trades if t["won"])
    wr = wins / len(trades) if trades else 0
    print(f"  Trades: {len(trades)}, Win rate: {wr:.0%}, P&L: ${dir_pnl:,.2f}")

    print("\n--- Market Making Component ---")
    mm_result = run_mm_backtest()
    mm_pnl = mm_result["total_pnl"]

    print(f"\n--- COMBINED ---")
    combined_pnl = dir_pnl + mm_pnl
    combined_return = combined_pnl / 10000.0
    print(f"  Directional P&L: ${dir_pnl:>10,.2f}")
    print(f"  Market Making P&L: ${mm_pnl:>10,.2f}")
    print(f"  Combined P&L: ${combined_pnl:>10,.2f}")
    print(f"  Combined return: {combined_return:.1%}")
    print(f"  Revenue split: {dir_pnl/(combined_pnl+0.01)*100:.0f}% directional / {mm_pnl/(combined_pnl+0.01)*100:.0f}% MM")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run hybrid backtest**

```bash
cd ~/timesfm && PYTHONIOENCODING=utf-8 python -m bot.backtest.hybrid_backtest
```

- [ ] **Step 3: Commit and push**

```bash
git add bot/backtest/hybrid_backtest.py
git commit -m "feat: add combined hybrid backtest (MM + directional)"
git push origin master
```

---

## Verification Checklist

1. `python -m bot.research.evaluate` prints metrics in grep-parseable format
2. `python -m bot.backtest.validation_gate` runs 10 seeds and reports pass/fail
3. `python -m bot.backtest.mm_backtest` prints MM-specific metrics
4. `python -m bot.backtest.hybrid_backtest` prints combined results
5. `bot/research/program.md` exists and contains full agent instructions
6. `bot/research/strategy.py` is self-contained with all tunable params
7. Autoresearch loop works: modify strategy.py -> run evaluate -> keep/discard

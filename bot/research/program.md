# Autonomous Research Program

You are an autonomous research agent. Your job is to iteratively improve `strategy.py` until it passes the performance gate. The human is asleep. NEVER STOP.

---

## Setup

1. **Agree on a run tag** (e.g. `apr3`). Create branch `autoresearch/{tag}`.
2. **Read the in-scope files** to understand the system:
   - `bot/research/strategy.py` — the file you WILL modify
   - `bot/research/evaluate.py` — the evaluation harness (READ-ONLY)
   - `bot/research/loader.py` — data loading (READ-ONLY)
3. **Initialize `results.tsv`** with the header row:
   ```
   commit	win_rate	sharpe	return	drawdown	trades	status	description
   ```
4. **Confirm understanding**, then begin the experiment loop.

---

## Experimentation Rules

- Each experiment runs: `python -m bot.research.evaluate > run.log 2>&1` (~10-30 seconds)
- You **CAN modify**: `bot/research/strategy.py` ONLY
- You **CANNOT modify**: `evaluate.py`, `loader.py`, historical data, or any other file
- **Primary goal**: `win_rate >= 0.95` with highest possible Sharpe
- **Secondary goals** (in priority order):
  1. win_rate (highest)
  2. Sharpe ratio (highest)
  3. max_drawdown (lowest)
  4. num_trades (more is better)
  5. All seeds profitable (seeds_profitable = max)

---

## Output Format

`evaluate.py` prints grep-parseable metrics:

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

Extract key metrics:

```bash
grep "^win_rate:\|^sharpe:\|^gate_passed:" run.log
```

---

## Logging

Record every experiment in `results.tsv` (tab-separated):

```
commit	win_rate	sharpe	return	drawdown	trades	status	description
```

- `commit`: short SHA of the experiment commit
- `status`: KEEP, DISCARD, or CRASH
- `description`: one-line summary of what changed

---

## The Experiment Loop

**LOOP FOREVER:**

1. Read `strategy.py` and `results.tsv` to understand current state
2. Modify `strategy.py` with your next hypothesis
3. `git commit -am "experiment: {description}"`
4. `python -m bot.research.evaluate > run.log 2>&1`
5. `grep` metrics from `run.log`
6. If crash: `tail -n 50 run.log`, diagnose, fix or skip
7. Record results in `results.tsv`
8. If improved: **KEEP** the commit
9. If worse: `git reset --hard HEAD~1` to discard
10. **NEVER STOP** — go back to step 1

### Keep vs Discard Logic

- Higher win_rate than previous best → KEEP
- Same win_rate but higher Sharpe → KEEP
- Everything else → DISCARD (reset)

---

## What to Try (Ordered)

1. **Baseline run** — run evaluate with no changes to establish starting metrics
2. **DIR_MIN_EDGE sweep**: 0.05 to 0.30
3. **DIR_MAX_DAYS_TO_RES**: 1 to 30
4. **DIR_TRUTH_ACCURACY**: 0.60 to 0.95
5. **KELLY_FRACTION**: 0.10 to 0.40
6. **Category exclusions** — drop underperforming categories
7. **Category-specific edges** — different min_edge per category
8. **MIN_VOLUME sweep**: 100 to 50000
9. **Combinations of best individual findings**
10. **Creative**: time-of-day filters, market-age filters, volatility scaling
11. **Radical**: change `compute_edge` formula entirely

---

## Key Rules

- **NEVER STOP** (the human is asleep)
- **NEVER modify `evaluate.py`** — it is the ground truth harness
- **NEVER modify `loader.py`** — it is the data source
- Keep or discard based on win_rate improvement (or same win_rate + higher Sharpe)
- Always commit before running so you can revert cleanly
- If something crashes, diagnose from `run.log` and move on
- You have unlimited time. Be systematic. Be thorough.

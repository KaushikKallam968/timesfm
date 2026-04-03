"""Vectorized backtesting engine."""
import numpy as np
import pandas as pd


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, equity_curve, trades, metrics):
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = metrics

    def __repr__(self):
        m = self.metrics
        return (
            f"BacktestResult(\n"
            f"  total_return={m['total_return']:.1%},\n"
            f"  annual_return={m['annual_return']:.1%},\n"
            f"  sharpe={m['sharpe']:.2f},\n"
            f"  max_drawdown={m['max_drawdown']:.1%},\n"
            f"  win_rate={m['win_rate']:.1%},\n"
            f"  profit_factor={m['profit_factor']:.2f},\n"
            f"  num_trades={m['num_trades']}\n"
            f")"
        )


def run_backtest(signals, prices, fee_rate=0.001, initial_capital=10000):
    """Run backtest on a signal series.

    Args:
        signals: array of position signals (-1=short, 0=flat, 1=long)
        prices: array of close prices (same length as signals)
        fee_rate: one-way trading fee (0.001 = 0.1%)
        initial_capital: starting capital

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    n = len(signals)
    positions = np.array(signals, dtype=float)
    prices = np.array(prices, dtype=float)

    # Calculate returns
    price_returns = np.diff(prices) / prices[:-1]

    # Position-weighted returns (shifted: today's signal -> tomorrow's return)
    strategy_returns = np.zeros(n)
    strategy_returns[1:] = positions[:-1] * price_returns

    # Subtract fees on position changes
    position_changes = np.abs(np.diff(positions, prepend=0))
    fees = position_changes * fee_rate
    strategy_returns -= fees

    # Build equity curve
    equity = initial_capital * np.cumprod(1 + strategy_returns)

    # Track trades
    trades = _extract_trades(positions, prices, equity)

    # Compute metrics
    metrics = _compute_metrics(strategy_returns, equity, trades)

    return BacktestResult(
        equity_curve=pd.Series(equity),
        trades=trades,
        metrics=metrics,
    )


def _extract_trades(positions, prices, equity):
    """Extract individual trades from position changes."""
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    direction = 0

    for i in range(len(positions)):
        if not in_trade and positions[i] != 0:
            # Enter trade
            in_trade = True
            entry_price = prices[i]
            entry_idx = i
            direction = positions[i]

        elif in_trade and (positions[i] != direction):
            # Exit trade
            exit_price = prices[i]
            pnl = direction * (exit_price / entry_price - 1)
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "direction": "long" if direction > 0 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl,
                "holding_period": i - entry_idx,
            })

            if positions[i] != 0:
                # Immediately enter new trade
                entry_price = prices[i]
                entry_idx = i
                direction = positions[i]
            else:
                in_trade = False

    return trades


def _compute_metrics(strategy_returns, equity, trades):
    """Compute standard trading metrics."""
    total_return = equity[-1] / equity[0] - 1
    n_days = len(strategy_returns)
    annual_factor = 365  # crypto trades 24/7

    # Annualized return
    annual_return = (1 + total_return) ** (annual_factor / n_days) - 1

    # Sharpe ratio (annualized)
    daily_mean = np.mean(strategy_returns)
    daily_std = np.std(strategy_returns)
    sharpe = (daily_mean / daily_std * np.sqrt(annual_factor)) if daily_std > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown)

    # Win rate and profit factor
    if trades:
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        win_rate = len(wins) / len(trades)
        total_wins = sum(t["pnl_pct"] for t in wins) if wins else 0
        total_losses = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1e-8
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades),
        "avg_holding_period": np.mean([t["holding_period"] for t in trades]) if trades else 0,
    }

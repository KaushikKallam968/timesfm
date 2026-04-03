"""Trading strategies using TimesFM predictions."""
import numpy as np
import time


def momentum_strategy(model, prices, context_len=512, horizon=7,
                      rebalance_every=7, threshold=0.001):
    """Momentum: predict future returns, go long if positive, short if negative.

    Args:
        model: compiled TimesFM model
        prices: array of close prices
        context_len: lookback window for model
        horizon: forecast horizon in days
        rebalance_every: days between rebalancing
        threshold: minimum predicted return to take a position

    Returns:
        signals: array of positions (-1, 0, 1) aligned with prices
    """
    n = len(prices)
    signals = np.zeros(n)
    log_prices = np.log(prices)

    for i in range(context_len, n, rebalance_every):
        context = log_prices[max(0, i - context_len):i]
        if len(context) < 64:
            continue

        point, quantiles = model.forecast(horizon=min(horizon, n - i), inputs=[context])
        predicted_log = point[0]

        # Predicted return over horizon
        current_log_price = context[-1]
        predicted_return = predicted_log[-1] - current_log_price

        # Set signal for the next `rebalance_every` days
        end_idx = min(i + rebalance_every, n)
        if predicted_return > threshold:
            signals[i:end_idx] = 1
        elif predicted_return < -threshold:
            signals[i:end_idx] = -1

    return signals


def quantile_volatility_strategy(model, prices, context_len=512, horizon=7,
                                  rebalance_every=7):
    """Use quantile spread as volatility estimate. Trade mean-reversion in low vol,
    trend-following in high vol.

    Args:
        model: compiled TimesFM model
        prices: array of close prices
        context_len: lookback window
        horizon: forecast horizon
        rebalance_every: days between rebalancing

    Returns:
        signals: array of positions (-1, 0, 1)
    """
    n = len(prices)
    signals = np.zeros(n)
    log_prices = np.log(prices)

    # Track historical volatility for regime detection
    vol_window = 30
    historical_vols = []

    for i in range(context_len, n, rebalance_every):
        context = log_prices[max(0, i - context_len):i]
        if len(context) < 64:
            continue

        point, quantiles = model.forecast(horizon=min(horizon, n - i), inputs=[context])

        # Quantile spread = predicted volatility
        q_low = quantiles[0, :, 1]   # ~10th percentile
        q_high = quantiles[0, :, -2]  # ~90th percentile
        predicted_vol = np.mean(q_high - q_low)

        # Historical realized volatility
        recent_returns = np.diff(context[-vol_window:])
        realized_vol = np.std(recent_returns) * np.sqrt(365)
        historical_vols.append(realized_vol)

        # Point forecast direction
        predicted_return = point[0, -1] - context[-1]

        # Strategy: size position inversely to predicted volatility
        # High vol → smaller position, Low vol → larger position
        median_vol = np.median(historical_vols) if historical_vols else predicted_vol
        vol_ratio = predicted_vol / (median_vol + 1e-8)

        end_idx = min(i + rebalance_every, n)

        if vol_ratio < 0.8:
            # Low vol regime: mean-reversion
            if predicted_return > 0:
                signals[i:end_idx] = 1
            elif predicted_return < 0:
                signals[i:end_idx] = -1
        elif vol_ratio > 1.5:
            # High vol regime: smaller positions, follow trend
            if predicted_return > 0:
                signals[i:end_idx] = 0.5
            elif predicted_return < 0:
                signals[i:end_idx] = -0.5
        else:
            # Normal vol: standard momentum
            if predicted_return > 0:
                signals[i:end_idx] = 0.7
            elif predicted_return < 0:
                signals[i:end_idx] = -0.7

    return signals


def buy_and_hold(prices):
    """Baseline: buy and hold."""
    return np.ones(len(prices))


def run_all_strategies(model, prices, symbol="unknown"):
    """Run all strategies and return results."""
    from .engine import run_backtest

    print(f"\n{'='*70}")
    print(f"Backtesting {symbol} ({len(prices)} bars)")
    print(f"{'='*70}")

    strategies = {
        "buy_and_hold": buy_and_hold(prices),
        "momentum_weekly": momentum_strategy(model, prices, rebalance_every=7, threshold=0.001),
        "momentum_biweekly": momentum_strategy(model, prices, rebalance_every=14, threshold=0.002),
        "quantile_vol": quantile_volatility_strategy(model, prices, rebalance_every=7),
    }

    results = {}
    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print("-" * 75)

    for name, signals in strategies.items():
        t0 = time.time()
        result = run_backtest(signals, prices, fee_rate=0.001)
        elapsed = time.time() - t0
        results[name] = result

        m = result.metrics
        print(f"{name:<25} {m['total_return']:>9.1%} {m['sharpe']:>8.2f} "
              f"{m['max_drawdown']:>7.1%} {m['win_rate']:>7.1%} {m['num_trades']:>8}")

    return results

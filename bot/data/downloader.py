"""Download and cache OHLCV data for backtesting."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(DATA_DIR, exist_ok=True)


def get_ohlcv(symbol, days=730, interval="1d"):
    """Get OHLCV data. Tries yfinance, falls back to synthetic."""
    cache_path = os.path.join(DATA_DIR, f"{symbol.replace('/', '_')}_{interval}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"  {symbol}: loaded from cache ({len(df)} bars)")
        return df

    # Try yfinance
    df = _try_yfinance(symbol, days)
    if df is not None and len(df) > 100:
        df.to_csv(cache_path)
        print(f"  {symbol}: downloaded via yfinance ({len(df)} bars)")
        return df

    # Fallback: generate realistic synthetic data
    df = _generate_synthetic(symbol, days)
    df.to_csv(cache_path)
    print(f"  {symbol}: generated synthetic ({len(df)} bars)")
    return df


def _try_yfinance(symbol, days):
    """Try downloading from Yahoo Finance."""
    try:
        import yfinance as yf
        ticker_map = {
            "BTC/USD": "BTC-USD",
            "ETH/USD": "ETH-USD",
            "SOL/USD": "SOL-USD",
            "SPY": "SPY",
            "AAPL": "AAPL",
        }
        ticker = ticker_map.get(symbol, symbol)
        period = f"{days}d" if days <= 730 else "max"
        df = yf.download(ticker, period=period, progress=False)
        if len(df) < 10:
            return None
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None


def _generate_synthetic(symbol, days):
    """Generate realistic synthetic OHLCV data based on known market properties."""
    np.random.seed(hash(symbol) % 2**31)

    # Set realistic parameters per asset type
    params = {
        "BTC/USD": {"start": 30000, "daily_vol": 0.035, "drift": 0.0003, "vol_of_vol": 0.15},
        "ETH/USD": {"start": 2000, "daily_vol": 0.045, "drift": 0.0002, "vol_of_vol": 0.18},
        "SOL/USD": {"start": 100, "daily_vol": 0.055, "drift": 0.0004, "vol_of_vol": 0.20},
        "SPY":     {"start": 450, "daily_vol": 0.012, "drift": 0.0003, "vol_of_vol": 0.08},
    }.get(symbol, {"start": 100, "daily_vol": 0.02, "drift": 0.0001, "vol_of_vol": 0.10})

    # Generate with stochastic volatility (more realistic than simple GBM)
    closes = np.zeros(days)
    closes[0] = params["start"]
    vol = params["daily_vol"]

    for i in range(1, days):
        # Stochastic vol (mean-reverting)
        vol = max(0.005, vol + params["vol_of_vol"] * (params["daily_vol"] - vol) + 0.002 * np.random.randn())
        ret = params["drift"] + vol * np.random.randn()

        # Add fat tails (occasional large moves)
        if np.random.random() < 0.02:
            ret += np.random.choice([-1, 1]) * vol * np.random.exponential(2)

        # Add weekly seasonality (crypto trades 24/7, lower weekend volume)
        day_of_week = i % 7
        if day_of_week >= 5:  # weekend
            vol *= 0.7

        closes[i] = closes[i-1] * np.exp(ret)

    # Generate OHLCV from closes
    highs = closes * (1 + np.abs(np.random.randn(days) * params["daily_vol"] * 0.5))
    lows = closes * (1 - np.abs(np.random.randn(days) * params["daily_vol"] * 0.5))
    opens = np.roll(closes, 1) * (1 + np.random.randn(days) * 0.002)
    opens[0] = closes[0]

    # Volume: correlated with absolute returns
    abs_returns = np.abs(np.diff(np.log(closes), prepend=np.log(closes[0])))
    base_volume = params["start"] * 1000
    volumes = base_volume * (1 + 5 * abs_returns) * np.random.lognormal(0, 0.3, days)

    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def get_log_returns(df, column="close"):
    """Compute log returns from price series."""
    return np.log(df[column] / df[column].shift(1)).dropna().values


if __name__ == "__main__":
    for sym in ["BTC/USD", "ETH/USD", "SPY"]:
        df = get_ohlcv(sym)
        rets = get_log_returns(df)
        print(f"  Returns: mean={rets.mean():.5f}, std={rets.std():.4f}, "
              f"skew={pd.Series(rets).skew():.2f}, kurt={pd.Series(rets).kurtosis():.2f}\n")

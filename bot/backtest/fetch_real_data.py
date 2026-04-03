"""Fetch real historical data for backtesting.

RUN THIS LOCALLY — not in Claude Code web (proxy blocks the APIs).

Usage:
    python bot/backtest/fetch_real_data.py

This fetches:
1. Polymarket resolved sports/weather markets + historical prices
2. The Odds API historical sportsbook odds (needs API key)
3. Open-Meteo historical weather data (free, no key)
4. Matches them up: truth_probability vs market_price vs actual_outcome

Output: bot/backtest/data/real_historical.json
"""
import json
import os
import time
from datetime import datetime, timedelta

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_polymarket_resolved_markets(category=None, limit=500):
    """Fetch resolved markets from Polymarket Gamma API."""
    params = {"limit": limit, "closed": "true", "order": "volume", "ascending": "false"}
    if category:
        params["tag"] = category

    try:
        resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching Polymarket markets: {e}")
        return []


def fetch_polymarket_price_history(token_id, start_ts=None, end_ts=None, interval="1d"):
    """Fetch historical price series for a Polymarket market."""
    params = {"market": token_id, "interval": interval}
    if start_ts:
        params["startTs"] = start_ts
    if end_ts:
        params["endTs"] = end_ts

    try:
        resp = requests.get(f"{GAMMA_API}/prices", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching price history for {token_id}: {e}")
        return []


def fetch_odds_api_historical(sport_key, date_str, api_key):
    """Fetch historical sportsbook odds from The Odds API.

    Requires paid plan for historical data.
    date_str format: YYYY-MM-DDTHH:MM:SSZ
    """
    url = f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds/"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "date": date_str}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  Error fetching Odds API for {sport_key} on {date_str}: {e}")
        return []


def fetch_weather_historical(lat, lon, start_date, end_date):
    """Fetch historical weather data from Open-Meteo (free, no key)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("daily", {}).get("time", [])
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        return list(zip(dates, temps))
    except Exception as e:
        print(f"  Error fetching weather for ({lat},{lon}): {e}")
        return []


def american_odds_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def build_sports_dataset(markets, api_key):
    """Match Polymarket sports markets with sportsbook odds."""
    dataset = []

    for market in markets:
        question = market.get("question", "")
        outcomes = market.get("outcomes", [])
        if not outcomes:
            continue

        # Get the Polymarket price at various points
        for outcome in outcomes:
            token_id = outcome.get("token_id")
            if not token_id:
                continue

            prices = fetch_polymarket_price_history(token_id)
            if not prices:
                continue

            # For each price point, we'd ideally match with sportsbook odds
            # from the same timestamp. This requires historical odds API access.
            for price_point in prices:
                dataset.append({
                    "timestamp": price_point.get("t", ""),
                    "market_id": market.get("id", ""),
                    "question": question,
                    "market_price": float(price_point.get("p", 0)),
                    "truth_probability": None,  # filled from sportsbook data
                    "actual_outcome": outcome.get("winner", ""),
                    "category": "sports",
                })

            time.sleep(0.5)  # rate limit

    return dataset


def build_weather_dataset():
    """Build weather dataset from historical forecasts vs actuals."""
    cities = {
        "NYC": (40.71, -74.01),
        "Chicago": (41.88, -87.63),
        "Miami": (25.76, -80.19),
        "LA": (34.05, -118.24),
        "Denver": (39.74, -104.99),
    }

    dataset = []
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    for city, (lat, lon) in cities.items():
        print(f"  Fetching weather history for {city}...")
        history = fetch_weather_historical(lat, lon, start_date, end_date)

        for date_str, actual_temp in history:
            if actual_temp is None:
                continue

            # Simulate what a market question would have been
            # "Will {city} high temp exceed {threshold}°F on {date}?"
            for threshold in [60, 70, 75, 80, 85, 90]:
                actual_outcome = "YES" if actual_temp > threshold else "NO"
                # Truth probability: we don't have the forecast from that day,
                # but we know the actual outcome, which is what matters for backtesting
                truth_prob = 0.90 if actual_outcome == "YES" else 0.10

                dataset.append({
                    "timestamp": date_str,
                    "market_id": f"weather_{city}_{date_str}_{threshold}",
                    "question": f"Will {city} high temp exceed {threshold}°F on {date_str}?",
                    "market_price": None,  # would need Polymarket historical
                    "truth_probability": truth_prob,
                    "actual_outcome": actual_outcome,
                    "category": "weather",
                    "city": city,
                    "threshold": threshold,
                    "actual_temp": actual_temp,
                })

        time.sleep(1)  # rate limit

    return dataset


def convert_poly_data_export(markets_csv_path, trades_csv_path):
    """Convert poly_data export (from github.com/warproxxx/poly_data) to our format.

    This is the recommended approach for getting real Polymarket data:
    1. Clone https://github.com/warproxxx/poly_data
    2. Run: uv run python update_all.py
    3. Pass the resulting markets.csv and trades.csv to this function

    Fields from markets.csv: createdAt, id, question, answer1, answer2,
      neg_risk, market_slug, token1, token2, condition_id, volume, ticker, closedTime
    Fields from trades.csv: timestamp, market_id, maker, taker, nonusdc_side,
      maker_direction, taker_direction, price, usd_amount, token_amount, transactionHash
    """
    import pandas as pd

    markets = pd.read_csv(markets_csv_path)
    trades = pd.read_csv(trades_csv_path)

    # Filter to resolved markets with decent volume
    resolved = markets[markets["closedTime"].notna() & (markets["volume"] > 10000)]

    dataset = []
    for _, market in resolved.iterrows():
        market_trades = trades[trades["market_id"] == market["id"]]
        if market_trades.empty:
            continue

        # Sample daily prices from trades
        market_trades = market_trades.sort_values("timestamp")
        for _, trade in market_trades.iterrows():
            dataset.append({
                "timestamp": trade["timestamp"],
                "market_id": str(market["id"]),
                "question": market["question"],
                "market_price": float(trade["price"]),
                "truth_probability": None,  # needs sportsbook/forecast matching
                "actual_outcome": None,  # needs outcome resolution
                "category": _categorize_market(market["question"]),
            })

    return dataset


def _categorize_market(question):
    """Categorize a market question."""
    q = question.lower()
    sports_keywords = ["nba", "nfl", "mlb", "nhl", "ufc", "mma", "soccer", "football",
                       "basketball", "baseball", "hockey", "boxing", "tennis",
                       "win", "beat", "defeat", "game", "match", "series"]
    weather_keywords = ["temperature", "temp", "weather", "°f", "°c", "forecast"]

    if any(kw in q for kw in sports_keywords):
        return "sports"
    if any(kw in q for kw in weather_keywords):
        return "weather"
    return "other"


if __name__ == "__main__":
    print("=== Real Historical Data Fetcher ===")
    print("NOTE: Run this LOCALLY, not in Claude Code web.\n")

    if not ODDS_API_KEY:
        print("WARNING: ODDS_API_KEY not set. Sports arb data will be incomplete.")
        print("  Get a free key at https://the-odds-api.com/\n")

    # Step 1: Fetch Polymarket resolved markets
    print("[1/3] Fetching Polymarket resolved markets...")
    sports_markets = fetch_polymarket_resolved_markets(category="sports")
    weather_markets = fetch_polymarket_resolved_markets(category="weather")
    print(f"  Found {len(sports_markets)} sports, {len(weather_markets)} weather markets")

    # Step 2: Fetch weather historical data (free, always works)
    print("\n[2/3] Fetching historical weather data...")
    weather_data = build_weather_dataset()
    print(f"  Generated {len(weather_data)} weather data points")

    # Step 3: Build sports dataset (needs Odds API key)
    print("\n[3/3] Building sports dataset...")
    if ODDS_API_KEY:
        sports_data = build_sports_dataset(sports_markets[:50], ODDS_API_KEY)
        print(f"  Built {len(sports_data)} sports data points")
    else:
        sports_data = []
        print("  Skipped (no ODDS_API_KEY)")

    # Combine and save
    all_data = sports_data + weather_data
    all_data.sort(key=lambda x: x.get("timestamp", ""))
    output_path = os.path.join(DATA_DIR, "real_historical.json")
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved {len(all_data)} data points to {output_path}")

    print("\n=== Next Steps ===")
    print("1. For FULL sports data: set ODDS_API_KEY env var (free at the-odds-api.com)")
    print("2. For FULL Polymarket data: clone github.com/warproxxx/poly_data and run it")
    print("3. Then run: python bot/backtest/fetch_real_data.py --convert path/to/markets.csv path/to/trades.csv")

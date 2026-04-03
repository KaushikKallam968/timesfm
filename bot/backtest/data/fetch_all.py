"""Fetch real historical data for backtesting.

Sources:
1. Polymarket resolved markets + price histories
2. Open-Meteo historical weather actuals

Output: bot/backtest/data/real_historical.json
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Polymarket ---

def fetch_polymarket_markets(n_pages=10, limit=100):
    """Fetch resolved (closed) Polymarket markets sorted by volume."""
    all_markets = []
    offset = 0

    for page in range(n_pages):
        url = f"https://gamma-api.polymarket.com/markets?limit={limit}&offset={offset}&closed=true&order=volume&ascending=false"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            markets = resp.json()
            if not markets:
                break
            all_markets.extend(markets)
            offset += limit
            print(f"  Polymarket page {page+1}: {len(markets)} markets (total: {len(all_markets)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error on page {page+1}: {e}")
            break

    return all_markets


def fetch_price_history(token_id, interval="1d"):
    """Fetch daily price history for a Polymarket token."""
    url = f"https://clob.polymarket.com/prices-history?market={token_id}&interval=max&fidelity=1440"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            return data
    except Exception:
        pass
    return []


def process_polymarket_data(markets):
    """Process markets into backtest records with price histories."""
    records = []
    price_data = {}
    fetched = 0

    for i, m in enumerate(markets):
        raw_tokens = m.get("clobTokenIds", "[]")
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else (raw_tokens or [])
        raw_outcomes = m.get("outcomes", "[]")
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else (raw_outcomes or [])
        raw_prices = m.get("outcomePrices", "[]")
        outcome_prices = json.loads(raw_prices) if isinstance(raw_prices, str) else (raw_prices or [])
        question = m.get("question", "")
        end_date = m.get("endDateIso") or m.get("endDate", "")
        start_date = m.get("startDateIso") or m.get("startDate", "")
        volume = m.get("volumeNum", 0) or m.get("volume", 0)

        if not token_ids or not outcomes or len(outcomes) < 2:
            continue
        if not outcome_prices or len(outcome_prices) < 2:
            continue

        # Determine winning outcome
        try:
            prices = [float(p) for p in outcome_prices]
        except (ValueError, TypeError):
            continue

        winner_idx = prices.index(max(prices))
        actual_outcome = 1.0 if winner_idx == 0 else 0.0

        # Categorize market
        q_lower = question.lower()
        if any(k in q_lower for k in ["nba", "nfl", "mlb", "nhl", "football", "basketball", "baseball", "hockey", "soccer", "tennis", "mma", "ufc", "boxing"]):
            category = "sports"
        elif any(k in q_lower for k in ["bitcoin", "ethereum", "btc", "eth", "price of", "above", "below"]):
            category = "crypto"
        elif any(k in q_lower for k in ["temperature", "weather", "rain", "snow", "hurricane"]):
            category = "weather"
        elif any(k in q_lower for k in ["election", "president", "vote", "democrat", "republican", "poll"]):
            category = "politics"
        elif any(k in q_lower for k in ["counter-strike", "dota", "league of legends", "esport", "game", "kills", "series"]):
            category = "esports"
        else:
            category = "other"

        # Fetch price history for first token (YES outcome)
        token_id = token_ids[0] if isinstance(token_ids, list) else token_ids
        if isinstance(token_id, str) and fetched < 300:
            history = fetch_price_history(token_id)
            if history:
                price_data[token_id] = history
                fetched += 1

                # Create records from price history
                for point in history:
                    ts = point.get("t", 0)
                    price = point.get("p", 0)
                    if ts and price:
                        try:
                            if isinstance(ts, (int, float)):
                                timestamp = datetime.utcfromtimestamp(ts).isoformat() + "Z"
                            else:
                                timestamp = str(ts)

                            records.append({
                                "timestamp": timestamp,
                                "market_id": m.get("conditionId", m.get("id", "")),
                                "question": question[:200],
                                "market_price": float(price),
                                "truth_probability": actual_outcome,
                                "actual_outcome": actual_outcome,
                                "category": category,
                                "volume": float(volume) if volume else 0,
                                "source": "polymarket",
                            })
                        except Exception:
                            continue

            if fetched % 20 == 0:
                print(f"  Price histories: {fetched} fetched, {len(records)} data points")
                time.sleep(0.5)

    return records, price_data


# --- Weather ---

CITIES = {
    "NYC": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "LA": (34.05, -118.24),
    "Denver": (39.74, -104.99),
    "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07),
    "Seattle": (47.61, -122.33),
    "Boston": (42.36, -71.06),
    "Atlanta": (33.75, -84.39),
}


def fetch_weather_actuals(start_date="2020-01-01", end_date="2026-04-03"):
    """Fetch historical daily max temperature from Open-Meteo."""
    all_weather = {}

    for city, (lat, lon) in CITIES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&temperature_unit=fahrenheit"
            f"&timezone=America/New_York"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_weather[city] = data
            days = len(data.get("daily", {}).get("time", []))
            print(f"  Weather: {city} - {days} days fetched")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Weather error for {city}: {e}")

    return all_weather


def process_weather_data(weather_data):
    """Convert weather data into backtest records.

    Creates binary outcome markets for temperature thresholds.
    """
    records = []
    thresholds = {
        "NYC": [32, 50, 70, 85],
        "Chicago": [32, 50, 70, 85],
        "Miami": [70, 80, 85, 90],
        "LA": [60, 70, 80, 90],
        "Denver": [32, 50, 65, 80],
        "Houston": [50, 70, 80, 90],
        "Phoenix": [70, 80, 90, 100],
        "Seattle": [40, 50, 60, 75],
        "Boston": [32, 50, 70, 85],
        "Atlanta": [40, 60, 75, 85],
    }

    for city, data in weather_data.items():
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        city_thresholds = thresholds.get(city, [50, 70, 85])

        for i, date_str in enumerate(times):
            if i >= len(temps_max) or temps_max[i] is None:
                continue

            temp_max = temps_max[i]
            temp_min = temps_min[i] if i < len(temps_min) and temps_min[i] is not None else None
            rain = precip[i] if i < len(precip) and precip[i] is not None else 0

            for threshold in city_thresholds:
                actual = 1.0 if temp_max > threshold else 0.0
                # Truth probability: using hindsight (perfect forecast baseline)
                records.append({
                    "timestamp": f"{date_str}T12:00:00Z",
                    "market_id": f"weather_{city}_{threshold}F_{date_str}",
                    "question": f"Will {city} max temp exceed {threshold}F on {date_str}?",
                    "market_price": actual,  # Perfect hindsight
                    "truth_probability": actual,
                    "actual_outcome": actual,
                    "actual_temp_max": temp_max,
                    "actual_temp_min": temp_min,
                    "precipitation_mm": rain,
                    "category": "weather",
                    "city": city,
                    "threshold_f": threshold,
                    "source": "open_meteo",
                })

    return records


# --- Main ---

def main():
    print("=" * 60)
    print("Fetching real historical data for backtesting")
    print("=" * 60)

    # 1. Polymarket
    print("\n--- Polymarket Resolved Markets ---")
    markets = fetch_polymarket_markets(n_pages=10)

    with open(os.path.join(DATA_DIR, "polymarket_markets.json"), "w") as f:
        json.dump(markets, f)
    print(f"Saved {len(markets)} markets")

    print("\n--- Polymarket Price Histories ---")
    poly_records, price_data = process_polymarket_data(markets)

    with open(os.path.join(DATA_DIR, "polymarket_prices.json"), "w") as f:
        json.dump(price_data, f)
    print(f"Saved price data for {len(price_data)} tokens, {len(poly_records)} data points")

    # 2. Weather
    print("\n--- Historical Weather Actuals ---")
    weather_data = fetch_weather_actuals(start_date="2020-01-01", end_date="2026-04-03")

    with open(os.path.join(DATA_DIR, "weather_actuals.json"), "w") as f:
        json.dump(weather_data, f)
    print(f"Saved weather for {len(weather_data)} cities")

    weather_records = process_weather_data(weather_data)
    print(f"Generated {len(weather_records)} weather backtest records")

    # 3. Combine
    all_records = poly_records + weather_records
    print(f"\n--- Unified Dataset ---")
    print(f"Total records: {len(all_records)}")

    # Stats
    categories = {}
    sources = {}
    for r in all_records:
        cat = r.get("category", "unknown")
        src = r.get("source", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1

    print("By category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,}")
    print("By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")

    # Date range
    dates = [r["timestamp"][:10] for r in all_records if r.get("timestamp")]
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")

    with open(os.path.join(DATA_DIR, "real_historical.json"), "w") as f:
        json.dump(all_records, f)
    print(f"\nSaved to real_historical.json ({len(all_records):,} records)")


if __name__ == "__main__":
    main()

"""Fetch real historical data for backtesting.

Sources:
1. Polymarket resolved markets + price histories (50k+ markets)
2. Open-Meteo historical weather actuals (10 cities, 6+ years)

Output: bot/backtest/data/real_historical.json
"""

import json
import os
import time
import requests
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Polymarket ---

def fetch_polymarket_events(n_pages=500, limit=100):
    """Fetch resolved events with embedded markets via the events API.

    The events API returns far more markets than the markets API because
    events contain multiple related sub-markets.
    """
    all_markets = []
    offset = 0

    for page in range(n_pages):
        url = (
            f"https://gamma-api.polymarket.com/events"
            f"?limit={limit}&offset={offset}&closed=true"
            f"&order=volume&ascending=false"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            events = resp.json()
            if not events:
                print(f"  Page {page+1}: empty, stopping")
                break
            for e in events:
                for m in e.get("markets", []):
                    m["event_title"] = e.get("title", "")
                    m["event_id"] = e.get("id", "")
                    all_markets.append(m)
            offset += limit
            if (page + 1) % 50 == 0:
                print(f"  Page {page+1}: {len(all_markets):,} markets from {offset:,} events")
            time.sleep(0.15)
        except Exception as e:
            print(f"  Error on page {page+1}: {e}")
            time.sleep(2)
            continue

    return all_markets


def fetch_price_history(token_id):
    """Fetch price history for a Polymarket token from CLOB API."""
    url = f"https://clob.polymarket.com/prices-history?market={token_id}&interval=max&fidelity=1440"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            history = data.get("history", [])
            if len(history) > 1:
                return history
    except Exception:
        pass
    return []


def categorize_market(question):
    """Categorize a market by its question text."""
    q = question.lower()
    if any(k in q for k in ["nba", "nfl", "mlb", "nhl", "football", "basketball",
                             "baseball", "hockey", "soccer", "tennis", "mma", "ufc",
                             "boxing", "cricket", "f1", "formula", "grand prix",
                             "olympics", "golf", "pga", "atp", "wta"]):
        return "sports"
    if any(k in q for k in ["counter-strike", "dota", "league of legends", "valorant",
                             "esport", "kills", "series:", "game 1", "game 2", "map 1",
                             "bo3", "bo5"]):
        return "esports"
    if any(k in q for k in ["bitcoin", "ethereum", "btc", "eth", "solana", "sol",
                             "crypto", "token", "price of", "above $", "below $"]):
        return "crypto"
    if any(k in q for k in ["temperature", "weather", "rain", "snow", "hurricane",
                             "tornado", "flood", "wildfire"]):
        return "weather"
    if any(k in q for k in ["election", "president", "vote", "democrat", "republican",
                             "poll", "senate", "congress", "governor", "trump", "biden",
                             "harris", "desantis", "rfk"]):
        return "politics"
    if any(k in q for k in ["fed", "interest rate", "inflation", "gdp", "unemployment",
                             "cpi", "fomc", "tariff"]):
        return "economics"
    if any(k in q for k in ["ai", "openai", "chatgpt", "google", "apple", "meta",
                             "microsoft", "tesla", "spacex", "launch"]):
        return "tech"
    return "other"


def process_polymarket(markets):
    """Process all resolved markets into backtest records.

    Two types of records:
    1. Resolution records: one per resolved market (outcome known)
    2. Price history records: daily prices for markets with CLOB history
    """
    resolution_records = []
    price_records = []
    price_fetch_count = 0
    max_price_fetches = 500  # limit API calls

    for i, m in enumerate(markets):
        question = m.get("question", "")
        raw_outcomes = m.get("outcomes", "[]")
        raw_prices = m.get("outcomePrices", "[]")
        raw_tokens = m.get("clobTokenIds", "[]")

        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else (raw_outcomes or [])
        outcome_prices = json.loads(raw_prices) if isinstance(raw_prices, str) else (raw_prices or [])
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else (raw_tokens or [])

        if len(outcomes) < 2 or len(outcome_prices) < 2:
            continue

        try:
            prices = [float(p) for p in outcome_prices]
        except (ValueError, TypeError):
            continue

        # Skip unresolved (both prices near 0.5)
        if all(abs(p - 0.5) < 0.01 for p in prices):
            continue

        winner_idx = prices.index(max(prices))
        actual_outcome = 1.0 if winner_idx == 0 else 0.0

        end_date = m.get("endDateIso") or m.get("endDate", "")
        start_date = m.get("startDateIso") or m.get("startDate") or m.get("createdAt", "")
        volume = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
        category = categorize_market(question)
        market_id = m.get("conditionId", m.get("id", ""))

        # Resolution record (every resolved market)
        resolution_records.append({
            "timestamp": end_date[:19] + "Z" if end_date else "",
            "market_id": market_id,
            "question": question[:250],
            "market_price": prices[0],
            "truth_probability": actual_outcome,
            "actual_outcome": actual_outcome,
            "winning_outcome": outcomes[winner_idx] if winner_idx < len(outcomes) else "",
            "category": category,
            "volume": volume,
            "start_date": start_date[:19] + "Z" if start_date else "",
            "source": "polymarket",
            "record_type": "resolution",
        })

        # Fetch price history for higher-volume markets
        if token_ids and price_fetch_count < max_price_fetches and volume > 100:
            tid = token_ids[0]
            history = fetch_price_history(tid)
            if history:
                price_fetch_count += 1
                for point in history:
                    ts = point.get("t", 0)
                    p = point.get("p", 0)
                    if ts and p:
                        try:
                            timestamp = datetime.utcfromtimestamp(ts).isoformat() + "Z" if isinstance(ts, (int, float)) else str(ts)
                            price_records.append({
                                "timestamp": timestamp,
                                "market_id": market_id,
                                "question": question[:250],
                                "market_price": float(p),
                                "truth_probability": actual_outcome,
                                "actual_outcome": actual_outcome,
                                "category": category,
                                "volume": volume,
                                "source": "polymarket",
                                "record_type": "price_history",
                            })
                        except Exception:
                            continue

            if price_fetch_count % 50 == 0 and price_fetch_count > 0:
                print(f"  Price histories: {price_fetch_count} tokens, {len(price_records)} points")
                time.sleep(0.5)

    return resolution_records, price_records


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
    """Fetch historical daily temperatures from Open-Meteo."""
    all_weather = {}
    for city, (lat, lon) in CITIES.items():
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&temperature_unit=fahrenheit&timezone=America/New_York"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_weather[city] = data
            days = len(data.get("daily", {}).get("time", []))
            print(f"  Weather: {city} - {days} days")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Weather error {city}: {e}")
    return all_weather


def process_weather(weather_data):
    """Convert weather into binary outcome market records."""
    records = []
    thresholds = {
        "NYC": [32, 50, 70, 85], "Chicago": [32, 50, 70, 85],
        "Miami": [70, 80, 85, 90], "LA": [60, 70, 80, 90],
        "Denver": [32, 50, 65, 80], "Houston": [50, 70, 80, 90],
        "Phoenix": [70, 80, 90, 100], "Seattle": [40, 50, 60, 75],
        "Boston": [32, 50, 70, 85], "Atlanta": [40, 60, 75, 85],
    }

    for city, data in weather_data.items():
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        for i, date_str in enumerate(times):
            if i >= len(temps) or temps[i] is None:
                continue
            temp = temps[i]
            for threshold in thresholds.get(city, [50, 70, 85]):
                actual = 1.0 if temp > threshold else 0.0
                records.append({
                    "timestamp": f"{date_str}T12:00:00Z",
                    "market_id": f"weather_{city}_{threshold}F_{date_str}",
                    "question": f"Will {city} max temp exceed {threshold}F on {date_str}?",
                    "market_price": actual,
                    "truth_probability": actual,
                    "actual_outcome": actual,
                    "actual_temp_max": temp,
                    "category": "weather",
                    "city": city,
                    "threshold_f": threshold,
                    "source": "open_meteo",
                    "record_type": "resolution",
                })
    return records


# --- Main ---

def main():
    print("=" * 60)
    print("Fetching real historical data for backtesting")
    print("=" * 60)

    # 1. Polymarket - fetch many more pages
    print("\n--- Polymarket Resolved Markets ---")
    markets = fetch_polymarket_events(n_pages=500, limit=100)
    print(f"Total: {len(markets):,} resolved markets")

    with open(os.path.join(DATA_DIR, "polymarket_markets.json"), "w") as f:
        json.dump(markets, f)

    print("\n--- Processing Polymarket Data ---")
    resolution_records, price_records = process_polymarket(markets)
    print(f"Resolution records: {len(resolution_records):,}")
    print(f"Price history records: {len(price_records):,}")

    with open(os.path.join(DATA_DIR, "polymarket_prices.json"), "w") as f:
        json.dump({"resolution_count": len(resolution_records), "price_count": len(price_records)}, f)

    # 2. Weather
    print("\n--- Historical Weather ---")
    weather_data = fetch_weather_actuals()
    with open(os.path.join(DATA_DIR, "weather_actuals.json"), "w") as f:
        json.dump(weather_data, f)

    weather_records = process_weather(weather_data)
    print(f"Weather records: {len(weather_records):,}")

    # 3. Combine
    all_records = resolution_records + price_records + weather_records

    # Stats
    print(f"\n{'=' * 60}")
    print(f"UNIFIED DATASET: {len(all_records):,} total records")
    print(f"{'=' * 60}")

    from collections import Counter
    cats = Counter(r["category"] for r in all_records)
    sources = Counter(r["source"] for r in all_records)
    types = Counter(r.get("record_type", "?") for r in all_records)

    print("By category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count:,}")
    print("By source:")
    for src, count in sources.most_common():
        print(f"  {src}: {count:,}")
    print("By record type:")
    for t, count in types.most_common():
        print(f"  {t}: {count:,}")

    dates = [r["timestamp"][:10] for r in all_records if r.get("timestamp")]
    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")

    # Polymarket-specific stats
    poly = [r for r in all_records if r["source"] == "polymarket"]
    poly_cats = Counter(r["category"] for r in poly)
    print(f"\nPolymarket breakdown ({len(poly):,} records):")
    for cat, count in poly_cats.most_common():
        print(f"  {cat}: {count:,}")

    with open(os.path.join(DATA_DIR, "real_historical.json"), "w") as f:
        json.dump(all_records, f)
    print(f"\nSaved to real_historical.json")


if __name__ == "__main__":
    main()

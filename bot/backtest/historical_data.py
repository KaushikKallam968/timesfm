"""Fetch and store historical data for backtesting."""
import json
import os
import random
from datetime import datetime, timedelta


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_synthetic_sports(n=1000, accuracy=0.88, seed=42):
    """Generate synthetic sports arb historical data.

    Simulates sportsbook truth vs Polymarket prices with a given
    accuracy rate (how often the sportsbook consensus is right).
    """
    random.seed(seed)
    data = []
    sports = ["NBA", "NFL", "MLB", "NHL", "MMA"]
    teams = ["Team_A", "Team_B", "Team_C", "Team_D"]

    start = datetime(2021, 1, 1)
    for i in range(n):
        truth_prob = random.uniform(0.30, 0.85)
        noise = random.gauss(0, 0.08)
        market_price = max(0.05, min(0.95, truth_prob - noise))

        if random.random() < accuracy:
            actual = "YES" if truth_prob > 0.5 else "NO"
        else:
            actual = "NO" if truth_prob > 0.5 else "YES"

        ts = start + timedelta(days=i * 365 * 5 // n)
        data.append({
            "timestamp": ts.strftime("%Y-%m-%d"),
            "market_id": f"sport_{i}",
            "market_price": round(market_price, 4),
            "truth_probability": round(truth_prob, 4),
            "actual_outcome": actual,
            "category": "sports",
            "sport": random.choice(sports),
            "team": random.choice(teams),
        })
    return data


def generate_synthetic_weather(n=500, accuracy=0.92, seed=123):
    """Generate synthetic weather arb historical data.

    Weather forecasts are typically more accurate than sports (GFS is
    very reliable 1-2 days out), so higher default accuracy.
    """
    random.seed(seed)
    data = []
    cities = ["NYC", "Chicago", "Miami", "LA", "Denver", "London", "Seoul"]

    start = datetime(2021, 1, 1)
    for i in range(n):
        truth_prob = random.uniform(0.40, 0.95)
        noise = random.gauss(0, 0.12)
        market_price = max(0.05, min(0.95, truth_prob - noise))

        if random.random() < accuracy:
            actual = "YES" if truth_prob > 0.5 else "NO"
        else:
            actual = "NO" if truth_prob > 0.5 else "YES"

        ts = start + timedelta(days=i * 365 * 5 // n)
        data.append({
            "timestamp": ts.strftime("%Y-%m-%d"),
            "market_id": f"weather_{i}",
            "market_price": round(market_price, 4),
            "truth_probability": round(truth_prob, 4),
            "actual_outcome": actual,
            "category": "weather",
            "city": random.choice(cities),
        })
    return data


def generate_synthetic_correlation(n=200, accuracy=0.97, seed=456):
    """Generate synthetic correlation arb data.

    Mathematical violations are almost always profitable (high accuracy)
    but occur less frequently.
    """
    random.seed(seed)
    data = []

    start = datetime(2021, 1, 1)
    for i in range(n):
        truth_prob = random.uniform(0.50, 0.95)
        noise = random.gauss(0, 0.06)
        market_price = max(0.05, min(0.95, truth_prob - noise))

        if random.random() < accuracy:
            actual = "YES" if truth_prob > 0.5 else "NO"
        else:
            actual = "NO" if truth_prob > 0.5 else "YES"

        ts = start + timedelta(days=i * 365 * 5 // n)
        data.append({
            "timestamp": ts.strftime("%Y-%m-%d"),
            "market_id": f"corr_{i}",
            "market_price": round(market_price, 4),
            "truth_probability": round(truth_prob, 4),
            "actual_outcome": actual,
            "category": "correlation",
        })
    return data


def generate_all_synthetic(sports_n=1000, weather_n=500, corr_n=200):
    """Generate combined synthetic dataset for full backtest."""
    sports = generate_synthetic_sports(sports_n)
    weather = generate_synthetic_weather(weather_n)
    corr = generate_synthetic_correlation(corr_n)

    combined = sports + weather + corr
    combined.sort(key=lambda x: x["timestamp"])
    return combined


def save_historical(data, filename="synthetic_5yr.json"):
    """Save historical data to JSON file."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_historical(filename="synthetic_5yr.json"):
    """Load historical data from JSON file."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    data = generate_all_synthetic()
    path = save_historical(data)
    print(f"Generated {len(data)} data points → {path}")

    sports = [d for d in data if d["category"] == "sports"]
    weather = [d for d in data if d["category"] == "weather"]
    corr = [d for d in data if d["category"] == "correlation"]
    print(f"  Sports: {len(sports)}, Weather: {len(weather)}, Correlation: {len(corr)}")

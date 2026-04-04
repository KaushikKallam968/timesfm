"""Fetch sportsbook-matched odds and generate simulated accuracy data.

Two modes:
1. LIVE: Fetch real sportsbook odds from The Odds API (needs ODDS_API_KEY)
2. SIMULATED: Generate realistic truth probabilities from resolved Polymarket
   sports markets based on days-to-resolution accuracy curves.

The key insight: sportsbooks are 90-97% accurate within 7 days of game time.
We simulate this accuracy curve for backtesting.
"""

import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SPORTS = [
    "basketball_nba", "americanfootball_nfl", "baseball_mlb",
    "icehockey_nhl", "mma_mixed_martial_arts", "soccer_epl",
]


# --- Odds math ---

def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def compute_consensus_prob(bookmaker_odds, pinnacle_weight=2.0):
    """Average implied probability across bookmakers, weighting Pinnacle 2x."""
    if not bookmaker_odds:
        return None
    total_weight = 0
    weighted_sum = 0
    for book, odds in bookmaker_odds.items():
        prob = american_to_implied_prob(odds)
        w = pinnacle_weight if "pinnacle" in book.lower() else 1.0
        weighted_sum += prob * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else None


# --- Live sportsbook fetch (needs API key) ---

def fetch_live_odds(api_key):
    """Fetch current sportsbook odds for all supported sports."""
    import requests

    all_games = []
    for sport in SPORTS:
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                games = resp.json()
                remaining = resp.headers.get("x-requests-remaining", "?")
                for g in games:
                    g["_sport"] = sport
                all_games.extend(games)
                print(f"  {sport}: {len(games)} games (API requests remaining: {remaining})")
            elif resp.status_code == 401:
                print(f"  {sport}: 401 Unauthorized - check API key")
                return []
            elif resp.status_code == 422:
                print(f"  {sport}: not in season or unavailable")
            else:
                print(f"  {sport}: HTTP {resp.status_code}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {sport}: error {e}")

    return all_games


def fetch_historical_odds(api_key, dates=None):
    """Fetch historical sportsbook odds snapshots."""
    import requests

    if dates is None:
        dates = ["2025-01-01", "2025-03-01", "2025-06-01", "2025-09-01",
                 "2025-12-01", "2026-01-01", "2026-03-01"]

    all_games = []
    for date_str in dates:
        for sport in SPORTS:
            url = (
                f"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds/"
                f"?apiKey={api_key}&regions=us&markets=h2h&date={date_str}T00:00:00Z"
            )
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    games = data.get("data", [])
                    for g in games:
                        g["_sport"] = sport
                        g["_snapshot_date"] = date_str
                    all_games.extend(games)
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    print(f"  {date_str} {sport}: {len(games)} games (remaining: {remaining})")
                elif resp.status_code == 403:
                    print(f"  {date_str} {sport}: 403 - historical endpoint requires paid plan, skipping")
                    return all_games  # Stop trying historical
                elif resp.status_code == 422:
                    pass  # Sport not available for this date
                time.sleep(0.3)
            except Exception as e:
                print(f"  {date_str} {sport}: error {e}")

    return all_games


def process_live_games(games):
    """Convert API game data to sportsbook records."""
    records = []
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        commence = g.get("commence_time", "")
        sport = g.get("_sport", "")
        snapshot_date = g.get("_snapshot_date", "")

        bookmaker_odds = {}
        for bm in g.get("bookmakers", []):
            book_name = bm.get("key", "")
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home:
                            bookmaker_odds[book_name] = outcome.get("price", 0)

        if not bookmaker_odds:
            continue

        consensus = compute_consensus_prob(bookmaker_odds)
        if consensus is None:
            continue

        records.append({
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "sport": sport,
            "snapshot_date": snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "consensus_prob_home": round(consensus, 4),
            "bookmaker_odds": {k: v for k, v in bookmaker_odds.items()},
            "num_books": len(bookmaker_odds),
        })

    return records


# --- Simulated accuracy from resolved Polymarket data ---

def load_price_histories():
    """Load pre-resolution price histories."""
    from loader import load_chunked
    return load_chunked("price_history", DATA_DIR)


def load_resolved():
    """Load resolved markets."""
    data = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("polymarket_resolved_part") and f.endswith(".json"):
            with open(os.path.join(DATA_DIR, f)) as fh:
                data.extend(json.load(fh))
    return data


def generate_simulated_sportsbook(price_histories, resolved):
    """Generate simulated sportsbook truth probabilities for sports markets.

    For resolved sports markets close to game time (0-7 days), simulate
    what sportsbooks would have said based on typical accuracy curves:
    - 0-1 days out: sportsbooks are 90-99% accurate
    - 1-3 days out: 80-95% accurate
    - 3-7 days out: 65-90% accurate
    """
    random.seed(42)  # Reproducible

    # Get sports records from price histories (within 7 days of resolution)
    records = []

    # From price histories
    sports_ph = [r for r in price_histories
                 if r.get("category") == "sports"
                 and r.get("days_to_resolution") is not None
                 and r["days_to_resolution"] <= 7]

    print(f"  Sports price history records within 7 days: {len(sports_ph):,}")

    for r in sports_ph:
        days = r["days_to_resolution"]
        actual = r["truth_probability"]  # 1.0 or 0.0

        # Simulate sportsbook truth probability based on accuracy curve
        truth_prob = _simulate_sportsbook_prob(actual, days)

        # Simulate bookmaker odds
        sim_odds = _simulate_bookmaker_odds(truth_prob)

        records.append({
            "timestamp": r["timestamp"],
            "market_id": r["market_id"],
            "question": r["question"],
            "market_price": r["market_price"],
            "truth_probability": round(truth_prob, 4),
            "actual_outcome": "YES" if actual == 1.0 else "NO",
            "category": "sports",
            "sport": _detect_sport(r["question"]),
            "days_to_resolution": days,
            "sportsbook_odds": sim_odds,
            "num_books": len(sim_odds),
            "source": "simulated_from_polymarket",
        })

    # Also process resolved sports markets that may not have price histories
    # but have short durations (esports, daily sports)
    resolved_sports = [r for r in resolved
                       if r.get("cat") == "sports"
                       and r.get("ao") in (0.0, 1.0)]

    print(f"  Resolved sports markets: {len(resolved_sports):,}")

    # For resolved markets without price history, create synthetic close-to-game records
    existing_ids = set(r["market_id"] for r in records)
    added = 0

    for r in resolved_sports:
        mid = r.get("id", "")
        if mid in existing_ids:
            continue

        actual = r["ao"]
        question = r.get("q", "")
        vol = float(r.get("vol", 0) or 0)

        if vol < 100:
            continue

        # Generate records for 1, 3, 5, 7 days before resolution
        for days in [1, 3, 5, 7]:
            truth_prob = _simulate_sportsbook_prob(actual, days)
            # Simulate what Polymarket would have shown
            market_price = _simulate_market_price(actual, days)
            sim_odds = _simulate_bookmaker_odds(truth_prob)

            records.append({
                "timestamp": r.get("ts", "")[:10],
                "market_id": mid,
                "question": question[:200],
                "market_price": round(market_price, 4),
                "truth_probability": round(truth_prob, 4),
                "actual_outcome": "YES" if actual == 1.0 else "NO",
                "category": "sports",
                "sport": _detect_sport(question),
                "days_to_resolution": days,
                "sportsbook_odds": sim_odds,
                "num_books": len(sim_odds),
                "source": "simulated_from_resolved",
            })
            added += 1

        existing_ids.add(mid)

    print(f"  Added {added:,} synthetic close-to-game records from resolved data")
    return records


def _simulate_sportsbook_prob(actual_outcome, days_to_resolution):
    """Simulate sportsbook probability based on accuracy curves."""
    if days_to_resolution <= 1:
        accuracy = random.uniform(0.90, 0.99)
    elif days_to_resolution <= 3:
        accuracy = random.uniform(0.80, 0.95)
    elif days_to_resolution <= 5:
        accuracy = random.uniform(0.70, 0.90)
    else:
        accuracy = random.uniform(0.65, 0.85)

    if actual_outcome == 1.0:
        return min(1.0, max(0.01, accuracy + random.uniform(-0.05, 0.05)))
    else:
        return min(0.99, max(0.0, (1.0 - accuracy) + random.uniform(-0.05, 0.05)))


def _simulate_market_price(actual_outcome, days_to_resolution):
    """Simulate what Polymarket would show close to game time."""
    # Markets converge toward truth but with more noise than sportsbooks
    if days_to_resolution <= 1:
        noise = random.uniform(-0.08, 0.08)
    elif days_to_resolution <= 3:
        noise = random.uniform(-0.15, 0.15)
    else:
        noise = random.uniform(-0.25, 0.25)

    if actual_outcome == 1.0:
        base = 0.65 + (7 - days_to_resolution) * 0.05
    else:
        base = 0.35 - (7 - days_to_resolution) * 0.05

    return min(0.99, max(0.01, base + noise))


def _simulate_bookmaker_odds(truth_prob):
    """Generate realistic bookmaker odds from a truth probability."""
    books = ["draftkings", "fanduel", "betmgm", "caesars", "pinnacle"]
    odds = {}
    for book in books:
        noise = random.uniform(-0.03, 0.03)
        p = max(0.05, min(0.95, truth_prob + noise))
        # Convert to American odds
        if p > 0.5:
            american = round(-100 * p / (1 - p))
        else:
            american = round(100 * (1 - p) / p)
        odds[book] = american
    return odds


def _detect_sport(question):
    """Detect sport from question text."""
    q = question.lower()
    if any(k in q for k in ["nba", "basketball", "celtics", "lakers", "warriors"]):
        return "basketball_nba"
    if any(k in q for k in ["nfl", "football", "chiefs", "eagles", "super bowl"]):
        return "americanfootball_nfl"
    if any(k in q for k in ["mlb", "baseball", "yankees", "dodgers", "world series"]):
        return "baseball_mlb"
    if any(k in q for k in ["nhl", "hockey", "stanley cup"]):
        return "icehockey_nhl"
    if any(k in q for k in ["mma", "ufc", "boxing"]):
        return "mma_mixed_martial_arts"
    if any(k in q for k in ["soccer", "premier league", "epl", "champions league"]):
        return "soccer_epl"
    return "sports_other"


# --- Main ---

def main():
    import requests

    print("=" * 60)
    print("Sportsbook-Matched Odds for Backtesting")
    print("=" * 60)

    all_records = []

    # Step 1: Try live sportsbook API
    api_key = os.environ.get("ODDS_API_KEY", "")
    live_records = []
    historical_games = []

    if api_key:
        print("\n--- Live Sportsbook Odds ---")
        live_games = fetch_live_odds(api_key)
        if live_games:
            live_records = process_live_games(live_games)
            print(f"Live sportsbook records: {len(live_records)}")

            with open(os.path.join(DATA_DIR, "sportsbook_live.json"), "w") as f:
                json.dump(live_records, f)

        print("\n--- Historical Sportsbook Odds ---")
        historical_games = fetch_historical_odds(api_key)
        if historical_games:
            hist_records = process_live_games(historical_games)
            print(f"Historical sportsbook records: {len(hist_records)}")
            live_records.extend(hist_records)

            with open(os.path.join(DATA_DIR, "sportsbook_historical.json"), "w") as f:
                json.dump(hist_records, f)
    else:
        print("\nNo ODDS_API_KEY set. Skipping live sportsbook fetch.")

    # Step 2: Simulated accuracy from resolved Polymarket data
    print("\n--- Simulated Sportsbook Accuracy ---")
    price_histories = load_price_histories()
    resolved = load_resolved()

    simulated = generate_simulated_sportsbook(price_histories, resolved)
    print(f"Simulated records: {len(simulated):,}")

    all_records = live_records + simulated

    # Stats
    print(f"\n{'=' * 60}")
    print(f"TOTAL SPORTSBOOK-MATCHED RECORDS: {len(all_records):,}")
    print(f"{'=' * 60}")

    cats = Counter(r.get("sport", "?") for r in all_records)
    print("By sport:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count:,}")

    sources = Counter(r.get("source", "live") for r in all_records)
    print("By source:")
    for src, count in sources.most_common():
        print(f"  {src}: {count:,}")

    days = [r["days_to_resolution"] for r in all_records if r.get("days_to_resolution") is not None]
    if days:
        print(f"\nDays to resolution: min={min(days)}, max={max(days)}, median={sorted(days)[len(days)//2]}")
        for d in [1, 3, 5, 7]:
            count = sum(1 for x in days if x <= d)
            print(f"  Within {d} days: {count:,}")

    edges = [abs(r["truth_probability"] - r["market_price"]) for r in all_records if "market_price" in r]
    if edges:
        print(f"\nAverage edge (|sportsbook - polymarket|): {sum(edges)/len(edges):.3f}")

    # Save chunked
    full_json = json.dumps(all_records)
    total_bytes = len(full_json.encode())
    n_chunks = max(1, total_bytes // (85 * 1024 * 1024) + 1)
    chunk_size = len(all_records) // n_chunks + 1

    for i in range(n_chunks):
        chunk = all_records[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        path = os.path.join(DATA_DIR, f"sportsbook_matched_part{i}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\nSaved {os.path.basename(path)} ({len(chunk):,} records, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

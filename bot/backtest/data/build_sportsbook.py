"""Build real sportsbook odds dataset from free public sources.

Sources:
1. nflverse (nfl_data_py) - NFL moneylines, spreads, totals (2020-2025)
2. aussportsbetting.com - NFL opening/closing decimal odds (2007-present)
3. football-data.co.uk - EPL odds from Bet365, Pinnacle, WHill (2020-2025)
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw_odds")


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds == 0:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def decimal_to_prob(odds):
    """Convert decimal odds to implied probability."""
    if pd.isna(odds) or odds <= 1:
        return None
    return 1.0 / odds


def decimal_to_american(odds):
    """Convert decimal odds to American odds."""
    if pd.isna(odds) or odds <= 1:
        return None
    if odds >= 2.0:
        return round((odds - 1) * 100)
    else:
        return round(-100 / (odds - 1))


# --- Source 1: nflverse ---

def process_nflverse():
    """Process NFL schedule data with real moneylines."""
    path = os.path.join(RAW_DIR, "nfl_nflverse.csv")
    if not os.path.exists(path):
        print("  nfl_nflverse.csv not found, skipping")
        return []

    df = pd.read_csv(path)
    records = []

    for _, row in df.iterrows():
        home_ml = row.get("home_moneyline")
        away_ml = row.get("away_moneyline")
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        gameday = row.get("gameday", "")
        spread = row.get("spread_line")

        if pd.isna(home_ml) or pd.isna(away_ml):
            continue
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_prob = american_to_prob(home_ml)
        away_prob = american_to_prob(away_ml)
        if home_prob is None or away_prob is None:
            continue

        # Remove vig (normalize to sum to 1)
        total = home_prob + away_prob
        home_prob_fair = home_prob / total
        away_prob_fair = away_prob / total

        # Actual outcome
        home_won = 1.0 if home_score > away_score else 0.0

        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        season = row.get("season", "")
        week = row.get("week", "")

        # Home team record
        records.append({
            "timestamp": str(gameday),
            "market_id": f"nfl_{season}_w{week}_{away_team}@{home_team}",
            "question": f"Will {home_team} beat {away_team}? (NFL {season} Week {week})",
            "market_price": round(home_prob_fair, 4),
            "truth_probability": round(home_prob_fair, 4),
            "actual_outcome": "YES" if home_won else "NO",
            "category": "sports",
            "sport": "americanfootball_nfl",
            "days_to_resolution": 0,
            "sportsbook_odds": {
                "consensus_home_ml": int(home_ml),
                "consensus_away_ml": int(away_ml),
                "spread": float(spread) if not pd.isna(spread) else None,
            },
            "num_books": 1,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "source": "nflverse",
        })

    return records


# --- Source 2: aussportsbetting.com ---

def process_aussportsbetting():
    """Process Australian sports betting historical data (real Pinnacle odds)."""
    path = os.path.join(RAW_DIR, "nfl_aus.xlsx")
    if not os.path.exists(path):
        print("  nfl_aus.xlsx not found, skipping")
        return []

    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"  Error reading nfl_aus.xlsx: {e}")
        return []

    records = []

    for _, row in df.iterrows():
        date = row.get("Date", "")
        home = row.get("Home Team", "")
        away = row.get("Away Team", "")
        home_score = row.get("Home Score")
        away_score = row.get("Away Score")
        home_open = row.get("Home Odds Open")
        home_close = row.get("Home Odds Close")
        away_open = row.get("Away Odds Open")
        away_close = row.get("Away Odds Close")

        if pd.isna(home_close) or pd.isna(away_close):
            continue
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        # Decimal odds -> implied probability (remove vig)
        home_prob = decimal_to_prob(home_close)
        away_prob = decimal_to_prob(away_close)
        if home_prob is None or away_prob is None:
            continue

        total = home_prob + away_prob
        home_prob_fair = home_prob / total

        home_won = 1.0 if float(home_score) > float(away_score) else 0.0

        # Convert date
        try:
            if isinstance(date, str):
                date_str = date[:10]
            else:
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        except Exception:
            date_str = str(date)[:10]

        odds_dict = {
            "pinnacle_home_close": decimal_to_american(home_close),
            "pinnacle_away_close": decimal_to_american(away_close),
        }
        if not pd.isna(home_open):
            odds_dict["pinnacle_home_open"] = decimal_to_american(home_open)
        if not pd.isna(away_open):
            odds_dict["pinnacle_away_open"] = decimal_to_american(away_open)

        records.append({
            "timestamp": date_str,
            "market_id": f"aus_nfl_{date_str}_{away}@{home}".replace(" ", "_"),
            "question": f"Will {home} beat {away}? (NFL)",
            "market_price": round(home_prob_fair, 4),
            "truth_probability": round(home_prob_fair, 4),
            "actual_outcome": "YES" if home_won else "NO",
            "category": "sports",
            "sport": "americanfootball_nfl",
            "days_to_resolution": 0,
            "sportsbook_odds": {k: v for k, v in odds_dict.items() if v is not None},
            "num_books": 1,
            "home_score": int(home_score) if not pd.isna(home_score) else None,
            "away_score": int(away_score) if not pd.isna(away_score) else None,
            "source": "aussportsbetting",
        })

    return records


# --- Source 3: football-data.co.uk ---

def process_football_data():
    """Process EPL data from football-data.co.uk (multiple named bookmakers)."""
    records = []

    for season in ["2021", "2122", "2223", "2324", "2425"]:
        path = os.path.join(RAW_DIR, f"epl_{season}.csv")
        if not os.path.exists(path):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        for _, row in df.iterrows():
            date = row.get("Date", "")
            home = row.get("HomeTeam", "")
            away = row.get("AwayTeam", "")
            ftr = row.get("FTR", "")  # Full Time Result: H, D, A
            fthg = row.get("FTHG")  # Full time home goals
            ftag = row.get("FTAG")  # Full time away goals

            if not ftr or pd.isna(ftr):
                continue

            # Collect odds from all bookmakers
            bookmaker_map = {
                "bet365": ("B365H", "B365D", "B365A"),
                "pinnacle": ("PSH", "PSD", "PSA"),
                "william_hill": ("WHH", "WHD", "WHA"),
                "betbrain_avg": ("AvgH", "AvgD", "AvgA"),
                "betbrain_max": ("MaxH", "MaxD", "MaxA"),
            }

            home_probs = []
            odds_dict = {}

            for book, (h_col, d_col, a_col) in bookmaker_map.items():
                h_odds = row.get(h_col)
                d_odds = row.get(d_col)
                a_odds = row.get(a_col)

                if pd.isna(h_odds) or pd.isna(d_odds) or pd.isna(a_odds):
                    continue
                if h_odds <= 1 or d_odds <= 1 or a_odds <= 1:
                    continue

                h_prob = 1.0 / h_odds
                d_prob = 1.0 / d_odds
                a_prob = 1.0 / a_odds
                total = h_prob + d_prob + a_prob
                h_fair = h_prob / total

                home_probs.append(h_fair)
                odds_dict[f"{book}_home"] = decimal_to_american(h_odds)
                odds_dict[f"{book}_draw"] = decimal_to_american(d_odds)
                odds_dict[f"{book}_away"] = decimal_to_american(a_odds)

            if not home_probs:
                continue

            avg_home_prob = sum(home_probs) / len(home_probs)

            # Outcome: for 1X2 markets, "home win" is the bet
            home_won = 1.0 if ftr == "H" else 0.0

            # Parse date
            try:
                date_str = pd.to_datetime(date, dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                date_str = str(date)[:10]

            records.append({
                "timestamp": date_str,
                "market_id": f"epl_{date_str}_{home}_v_{away}".replace(" ", "_"),
                "question": f"Will {home} beat {away}? (EPL)",
                "market_price": round(avg_home_prob, 4),
                "truth_probability": round(avg_home_prob, 4),
                "actual_outcome": "YES" if home_won else "NO",
                "full_time_result": ftr,
                "category": "sports",
                "sport": "soccer_epl",
                "days_to_resolution": 0,
                "sportsbook_odds": {k: v for k, v in odds_dict.items() if v is not None},
                "num_books": len(home_probs),
                "home_goals": int(fthg) if not pd.isna(fthg) else None,
                "away_goals": int(ftag) if not pd.isna(ftag) else None,
                "source": "football_data_uk",
            })

    return records


def main():
    print("=" * 60)
    print("Building Real Sportsbook Odds Dataset")
    print("=" * 60)

    # Process each source
    print("\n--- nflverse (NFL moneylines) ---")
    nfl_records = process_nflverse()
    print(f"  {len(nfl_records):,} records")

    print("\n--- aussportsbetting.com (NFL Pinnacle odds) ---")
    aus_records = process_aussportsbetting()
    print(f"  {len(aus_records):,} records")

    print("\n--- football-data.co.uk (EPL multi-bookmaker) ---")
    epl_records = process_football_data()
    print(f"  {len(epl_records):,} records")

    all_records = nfl_records + aus_records + epl_records

    # Deduplicate NFL (nflverse vs aus may overlap)
    seen = set()
    deduped = []
    for r in all_records:
        key = (r["timestamp"][:10], r["sport"], r["question"][:50])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"\n--- Combined ---")
    print(f"Before dedup: {len(all_records):,}")
    print(f"After dedup: {len(deduped):,}")

    from collections import Counter
    sports = Counter(r["sport"] for r in deduped)
    sources = Counter(r["source"] for r in deduped)
    print("\nBy sport:")
    for s, c in sports.most_common():
        print(f"  {s}: {c:,}")
    print("By source:")
    for s, c in sources.most_common():
        print(f"  {s}: {c:,}")

    dates = sorted(r["timestamp"][:10] for r in deduped if r.get("timestamp"))
    if dates:
        print(f"\nDate range: {dates[0]} to {dates[-1]}")
        years = Counter(d[:4] for d in dates)
        print("By year:")
        for y, c in sorted(years.items()):
            print(f"  {y}: {c:,}")

    # Replace the simulated sportsbook file
    path = os.path.join(DATA_DIR, "sportsbook_matched_part0.json")
    with open(path, "w") as f:
        json.dump(deduped, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nSaved sportsbook_matched_part0.json ({len(deduped):,} records, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

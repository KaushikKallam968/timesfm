"""Fetch pre-resolution price histories for top Polymarket markets.

Uses long-running (7+ day) resolved markets with daily price snapshots.
Each record has the market_price BEFORE resolution vs the actual outcome.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_price_history(token_id):
    """Fetch daily price history from CLOB API."""
    url = f"https://clob.polymarket.com/prices-history?market={token_id}&interval=max&fidelity=1440"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception:
        pass
    return []


def categorize(question):
    q = question.lower()
    if any(k in q for k in ["nba","nfl","mlb","nhl","football","basketball","baseball",
            "hockey","soccer","tennis","mma","ufc","boxing","cricket","f1","grand prix",
            "olympics","golf","pga","atp","wta","premier league","la liga","champions league",
            "world cup","super bowl"]):
        return "sports"
    if any(k in q for k in ["counter-strike","dota","league of legends","valorant",
            "esport","kills","bo3","bo5"]):
        return "esports"
    if any(k in q for k in ["bitcoin","ethereum","btc","eth","solana","crypto","doge",
            "above $","below $"]):
        return "crypto"
    if any(k in q for k in ["temperature","weather","rain","snow","hurricane"]):
        return "weather"
    if any(k in q for k in ["election","president","vote","democrat","republican","senate",
            "congress","governor","trump","biden","harris","cabinet","nominee","inauguration"]):
        return "politics"
    if any(k in q for k in ["fed ","interest rate","inflation","gdp","unemployment","cpi",
            "fomc","tariff"]):
        return "economics"
    if any(k in q for k in ["ai ","openai","chatgpt","google","apple","meta ","microsoft",
            "tesla","spacex","tiktok"]):
        return "tech"
    return "other"


def main():
    # Load pre-computed candidates (long-running, high-volume, resolved)
    candidates_path = os.path.join(DATA_DIR, "_candidates.json")
    if os.path.exists(candidates_path):
        with open(candidates_path) as f:
            candidates = json.load(f)
        print(f"Loaded {len(candidates):,} pre-computed candidates")
    else:
        print("No candidates file. Run the candidate finder first.")
        return

    # Fetch price histories
    all_records = []
    fetched = 0
    empty = 0
    total_points = 0

    for i, c in enumerate(candidates):
        token_id = c["token"]
        history = fetch_price_history(token_id)

        if not history or len(history) < 3:
            empty += 1
            time.sleep(0.05)
            continue

        fetched += 1

        # Determine actual outcome
        prices = c.get("prices", [1, 0])
        winner_idx = prices.index(max(prices))
        actual_outcome_val = 1.0 if winner_idx == 0 else 0.0
        outcomes = c.get("outcomes", ["Yes", "No"])
        winning_outcome = outcomes[winner_idx] if winner_idx < len(outcomes) else "Unknown"

        question = c.get("question", "")
        category = categorize(question + " " + c.get("event_title", ""))
        market_id = c.get("cid", "")
        volume = c.get("vol", 0)

        # Parse resolution date
        end_str = c.get("end", "")
        try:
            res_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except Exception:
            res_date = None

        for point in history:
            ts = point.get("t", 0)
            price = point.get("p", 0)
            if not ts or price is None:
                continue

            try:
                point_date = datetime.fromtimestamp(ts, tz=timezone.utc)

                # Days to resolution
                days_to_res = (res_date - point_date).days if res_date else None

                # Skip post-resolution prices
                if days_to_res is not None and days_to_res < 0:
                    continue

                all_records.append({
                    "timestamp": point_date.strftime("%Y-%m-%d"),
                    "market_id": market_id,
                    "question": question[:200],
                    "market_price": round(float(price), 4),
                    "truth_probability": actual_outcome_val,
                    "actual_outcome": winning_outcome,
                    "category": category,
                    "days_to_resolution": days_to_res,
                    "volume": volume,
                })
                total_points += 1
            except Exception:
                continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(candidates)}] fetched={fetched} empty={empty} records={len(all_records):,}")
            time.sleep(0.3)

    print(f"\nDone: {fetched} markets with history, {empty} empty/sparse")
    print(f"Total pre-resolution records: {len(all_records):,}")

    if not all_records:
        print("No records to save!")
        return

    # Stats
    from collections import Counter
    cats = Counter(r["category"] for r in all_records)
    print("\nBy category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count:,}")

    days = [r["days_to_resolution"] for r in all_records if r["days_to_resolution"] is not None]
    if days:
        print(f"\nDays to resolution: min={min(days)}, max={max(days)}, median={sorted(days)[len(days)//2]}")
        print(f"Records with 7+ days before resolution: {sum(1 for d in days if d >= 7):,}")
        print(f"Records with 30+ days before resolution: {sum(1 for d in days if d >= 30):,}")

    edges = [abs(r["truth_probability"] - r["market_price"]) for r in all_records]
    avg_edge = sum(edges) / len(edges)
    print(f"Average edge (|truth - market|): {avg_edge:.3f}")

    # Save chunked (<85MB per file)
    full_json = json.dumps(all_records)
    total_bytes = len(full_json.encode())
    n_chunks = max(1, total_bytes // (85 * 1024 * 1024) + 1)
    chunk_size = len(all_records) // n_chunks + 1

    for i in range(n_chunks):
        chunk = all_records[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        path = os.path.join(DATA_DIR, f"price_history_part{i}.json")
        with open(path, "w") as f:
            json.dump(chunk, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved {os.path.basename(path)} ({len(chunk):,} records, {size_mb:.1f} MB)")

    # Clean up candidates file
    os.remove(candidates_path)
    print("Removed _candidates.json")


if __name__ == "__main__":
    main()

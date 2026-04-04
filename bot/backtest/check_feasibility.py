"""Feasibility Check: Polymarket API latency, liquidity, and anti-arb measures.

Tests whether this arbitrage strategy is still viable in practice:
1. API response latency (can we get prices fast enough?)
2. Order book depth (is there liquidity to trade?)
3. Rate limiting (are we throttled?)
4. Market structure changes (has Polymarket patched arb strategies?)

Usage: python -m bot.backtest.check_feasibility
"""

import json
import os
import sys
import time
import requests
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_gamma_api_latency(n_requests=10):
    """Test latency to Polymarket's Gamma API (market data)."""
    url = "https://gamma-api.polymarket.com/markets?limit=1&closed=false&order=volume&ascending=false"
    latencies = []
    errors = 0

    for i in range(n_requests):
        try:
            start = time.monotonic()
            resp = requests.get(url, timeout=10)
            elapsed = time.monotonic() - start
            latencies.append(elapsed * 1000)  # ms

            if resp.status_code != 200:
                errors += 1
            time.sleep(0.5)
        except Exception as e:
            errors += 1
            print(f"  Request {i+1} failed: {e}")

    return {
        "endpoint": "gamma-api (market data)",
        "requests": n_requests,
        "errors": errors,
        "latencies_ms": [round(l, 1) for l in latencies],
        "min_ms": round(min(latencies), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
        "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else None,
    }


def test_clob_api_latency(n_requests=10):
    """Test latency to Polymarket's CLOB API (order book)."""
    # First get a live market token
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets?limit=1&closed=false&order=volume&ascending=false",
            timeout=10
        )
        markets = resp.json()
        if not markets:
            return {"error": "no live markets found"}

        raw_tokens = markets[0].get("clobTokenIds", "[]")
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        if not tokens:
            return {"error": "no token IDs"}
        token_id = tokens[0]
        market_q = markets[0].get("question", "")[:60]
    except Exception as e:
        return {"error": str(e)}

    # Test CLOB midpoint
    latencies_midpoint = []
    latencies_book = []

    for i in range(n_requests):
        # Midpoint price
        try:
            start = time.monotonic()
            resp = requests.get(f"https://clob.polymarket.com/midpoint?token_id={token_id}", timeout=10)
            elapsed = time.monotonic() - start
            latencies_midpoint.append(elapsed * 1000)
            time.sleep(0.3)
        except Exception:
            pass

        # Order book
        try:
            start = time.monotonic()
            resp = requests.get(f"https://clob.polymarket.com/book?token_id={token_id}", timeout=10)
            elapsed = time.monotonic() - start
            latencies_book.append(elapsed * 1000)
            time.sleep(0.3)
        except Exception:
            pass

    result = {"market": market_q, "token_id": token_id[:40] + "..."}

    if latencies_midpoint:
        result["midpoint"] = {
            "avg_ms": round(sum(latencies_midpoint) / len(latencies_midpoint), 1),
            "min_ms": round(min(latencies_midpoint), 1),
            "max_ms": round(max(latencies_midpoint), 1),
            "p50_ms": round(sorted(latencies_midpoint)[len(latencies_midpoint) // 2], 1),
        }
    if latencies_book:
        result["orderbook"] = {
            "avg_ms": round(sum(latencies_book) / len(latencies_book), 1),
            "min_ms": round(min(latencies_book), 1),
            "max_ms": round(max(latencies_book), 1),
            "p50_ms": round(sorted(latencies_book)[len(latencies_book) // 2], 1),
        }

    return result


def check_orderbook_depth(token_id=None):
    """Check order book depth for a live market."""
    if not token_id:
        try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/markets?limit=5&closed=false&order=volume&ascending=false",
                timeout=10
            )
            markets = resp.json()
            results = []
            for m in markets[:3]:
                raw_tokens = m.get("clobTokenIds", "[]")
                tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
                if tokens:
                    result = _check_single_book(tokens[0], m.get("question", ""))
                    if result:
                        results.append(result)
            return results
        except Exception as e:
            return [{"error": str(e)}]
    else:
        return [_check_single_book(token_id)]


def _check_single_book(token_id, question=""):
    """Check depth of a single order book."""
    try:
        resp = requests.get(f"https://clob.polymarket.com/book?token_id={token_id}", timeout=10)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "question": question[:60]}

        book = resp.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        bid_depth = sum(float(b.get("size", 0)) for b in bids[:10])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:10])

        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        spread = best_ask - best_bid

        return {
            "question": question[:60],
            "best_bid": round(best_bid, 4),
            "best_ask": round(best_ask, 4),
            "spread": round(spread, 4),
            "spread_pct": round(spread * 100, 2),
            "bid_depth_top10": round(bid_depth, 2),
            "ask_depth_top10": round(ask_depth, 2),
            "num_bid_levels": len(bids),
            "num_ask_levels": len(asks),
        }
    except Exception as e:
        return {"error": str(e), "question": question[:60]}


def check_rate_limits():
    """Test if Polymarket has aggressive rate limiting."""
    url = "https://gamma-api.polymarket.com/markets?limit=1"
    results = []
    blocked = False

    # Burst: 20 requests as fast as possible
    print("  Testing burst rate (20 rapid requests)...")
    for i in range(20):
        try:
            start = time.monotonic()
            resp = requests.get(url, timeout=5)
            elapsed = time.monotonic() - start
            results.append({
                "request": i + 1,
                "status": resp.status_code,
                "latency_ms": round(elapsed * 1000, 1),
            })
            if resp.status_code == 429:
                blocked = True
                print(f"    Rate limited at request {i+1}")
                break
        except Exception as e:
            results.append({"request": i + 1, "error": str(e)})

    statuses = [r.get("status") for r in results if "status" in r]
    return {
        "burst_requests": len(results),
        "blocked": blocked,
        "status_codes": dict(__import__("collections").Counter(statuses)),
        "blocked_at_request": next((r["request"] for r in results if r.get("status") == 429), None),
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in results if "latency_ms" in r) / max(len(results), 1), 1),
    }


def check_anti_arb_measures():
    """Check for known anti-arbitrage measures on Polymarket."""
    findings = []

    # Check if markets have minimum order sizes
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets?limit=3&closed=false&order=volume&ascending=false",
            timeout=10
        )
        markets = resp.json()
        for m in markets[:3]:
            min_size = m.get("orderMinSize")
            min_tick = m.get("orderPriceMinTickSize")
            fees_enabled = m.get("feesEnabled")
            fee_type = m.get("feeType")
            rfq = m.get("rfqEnabled")  # Request for Quote (institutional)
            neg_risk = m.get("negRisk")
            seconds_delay = m.get("secondsDelay")

            findings.append({
                "question": m.get("question", "")[:60],
                "order_min_size": min_size,
                "price_min_tick": min_tick,
                "fees_enabled": fees_enabled,
                "fee_type": fee_type,
                "rfq_enabled": rfq,
                "neg_risk": neg_risk,
                "seconds_delay": seconds_delay,
            })
    except Exception as e:
        findings.append({"error": str(e)})

    return findings


def main():
    print("=" * 60)
    print("FEASIBILITY CHECK: Polymarket Latency & Anti-Arb")
    print("=" * 60)

    report = {}

    # 1. Gamma API latency
    print("\n--- Gamma API Latency (market data) ---")
    gamma = test_gamma_api_latency(10)
    print(f"  Avg: {gamma.get('avg_ms')}ms, P50: {gamma.get('p50_ms')}ms, P95: {gamma.get('p95_ms')}ms")
    print(f"  Errors: {gamma.get('errors')}/{gamma.get('requests')}")
    report["gamma_api_latency"] = gamma

    # 2. CLOB API latency
    print("\n--- CLOB API Latency (prices & order book) ---")
    clob = test_clob_api_latency(10)
    if "midpoint" in clob:
        print(f"  Midpoint: avg={clob['midpoint']['avg_ms']}ms")
    if "orderbook" in clob:
        print(f"  Orderbook: avg={clob['orderbook']['avg_ms']}ms")
    report["clob_api_latency"] = clob

    # 3. Order book depth
    print("\n--- Order Book Depth (top 3 markets) ---")
    books = check_orderbook_depth()
    for b in books:
        if "error" not in b:
            print(f"  {b.get('question', '?')}")
            print(f"    Spread: {b.get('spread_pct')}% | Bid depth: ${b.get('bid_depth_top10'):,.0f} | Ask depth: ${b.get('ask_depth_top10'):,.0f}")
        else:
            print(f"  Error: {b.get('error')}")
    report["orderbook_depth"] = books

    # 4. Rate limiting
    print("\n--- Rate Limiting ---")
    rate = check_rate_limits()
    print(f"  Burst test: {rate.get('burst_requests')} requests, blocked: {rate.get('blocked')}")
    print(f"  Status codes: {rate.get('status_codes')}")
    report["rate_limiting"] = rate

    # 5. Anti-arb measures
    print("\n--- Anti-Arbitrage Measures ---")
    anti_arb = check_anti_arb_measures()
    for f in anti_arb:
        if "error" not in f:
            print(f"  {f.get('question', '?')}")
            print(f"    Min order: {f.get('order_min_size')}, Min tick: {f.get('price_min_tick')}")
            print(f"    Fees: {f.get('fees_enabled')} ({f.get('fee_type')}), Delay: {f.get('seconds_delay')}s")
            print(f"    RFQ: {f.get('rfq_enabled')}, NegRisk: {f.get('neg_risk')}")
    report["anti_arb_measures"] = anti_arb

    # Verdict
    print("\n" + "=" * 60)
    print("FEASIBILITY VERDICT")
    print("=" * 60)

    avg_latency = gamma.get("avg_ms", 0)
    blocked = rate.get("blocked", False)
    spreads = [b.get("spread_pct", 0) for b in books if "spread_pct" in b]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    delays = [f.get("seconds_delay", 0) for f in anti_arb if f.get("seconds_delay")]

    issues = []
    if avg_latency > 500:
        issues.append(f"High API latency ({avg_latency:.0f}ms avg)")
    if blocked:
        issues.append("Rate limited during burst test")
    if avg_spread > 3:
        issues.append(f"Wide spreads ({avg_spread:.1f}% avg) eat into edge")
    if any(d and d > 0 for d in delays):
        issues.append(f"Order delay detected ({max(delays)}s) - may prevent fast execution")

    if not issues:
        print("  STATUS: VIABLE")
        print("  No blocking issues detected. API is responsive, books have depth.")
    else:
        print("  STATUS: CONCERNS")
        for issue in issues:
            print(f"  - {issue}")

    report["verdict"] = {
        "status": "VIABLE" if not issues else "CONCERNS",
        "issues": issues,
        "avg_latency_ms": avg_latency,
        "rate_limited": blocked,
        "avg_spread_pct": round(avg_spread, 2),
    }

    with open(os.path.join(RESULTS_DIR, "feasibility.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to results/feasibility.json")


if __name__ == "__main__":
    main()

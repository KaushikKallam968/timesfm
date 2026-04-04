"""Market making backtest: simulates two-sided quotes on historical markets."""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from loader import load_price_histories
from bot.research.strategy import (
    MM_BASE_SPREAD,
    MM_MIN_SPREAD,
    MM_INVENTORY_SKEW,
    STARTING_BANKROLL,
    FEE_RATE,
    compute_mm_quotes,
)


def run_mm_backtest():
    """Run market making backtest on historical price data.

    Returns dict with keys: markets_traded, markets_profitable, mm_win_rate,
    spread_earned, inventory_pnl, total_pnl, mm_return.
    """
    data = load_price_histories()

    # Filter to usable records: price 0.05-0.95, truth in {0.0, 1.0}, has timestamp
    usable = [
        r for r in data
        if 0.05 <= r["market_price"] <= 0.95
        and r.get("truth_probability") in (0.0, 1.0)
        and r.get("timestamp")
    ]

    # Group by market_id
    by_market = defaultdict(list)
    for r in usable:
        by_market[r["market_id"]].append(r)

    # Sort each group by timestamp
    for mid in by_market:
        by_market[mid].sort(key=lambda r: r["timestamp"])

    bankroll = STARTING_BANKROLL
    total_spread_earned = 0.0
    total_inventory_pnl = 0.0
    markets_traded = 0
    markets_profitable = 0

    for mid, records in by_market.items():
        if len(records) < 3:
            continue

        markets_traded += 1
        net_inventory = 0.0
        market_spread_earned = 0.0
        market_fees = 0.0

        for i in range(len(records) - 1):
            cur = records[i]
            nxt = records[i + 1]

            mid_price = cur["market_price"]
            bid, ask = compute_mm_quotes(mid_price, net_inventory, bankroll)
            spread = ask - bid

            if spread <= 0:
                continue

            price_move = nxt["market_price"] - cur["market_price"]

            # Estimate fills: if price moved more than half the spread, we got filled
            half_spread = spread / 2.0

            if abs(price_move) > half_spread:
                fill_revenue = half_spread
                fee_cost = fill_revenue * FEE_RATE
                market_spread_earned += fill_revenue - fee_cost
                market_fees += fee_cost

                if price_move > 0:
                    # Price went up — our bid got filled, we bought YES
                    net_inventory += 1.0
                else:
                    # Price went down — our ask got filled, we sold YES
                    net_inventory -= 1.0

        # Settlement: value remaining inventory at resolution
        actual = records[-1]["truth_probability"]
        if actual == 1.0:
            settlement_value = net_inventory * 1.0  # YES shares worth $1
        else:
            settlement_value = net_inventory * 0.0  # YES shares worth $0

        # Inventory P&L: settlement value minus cost basis
        # Cost basis approximation: net_inventory * avg_mid_price
        avg_mid = sum(r["market_price"] for r in records) / len(records)
        inventory_cost = net_inventory * avg_mid
        market_inventory_pnl = settlement_value - inventory_cost

        market_total = market_spread_earned + market_inventory_pnl
        if market_total > 0:
            markets_profitable += 1

        total_spread_earned += market_spread_earned
        total_inventory_pnl += market_inventory_pnl

    total_pnl = total_spread_earned + total_inventory_pnl
    mm_return = total_pnl / STARTING_BANKROLL if STARTING_BANKROLL > 0 else 0.0
    mm_win_rate = markets_profitable / markets_traded if markets_traded > 0 else 0.0

    return {
        "markets_traded": markets_traded,
        "markets_profitable": markets_profitable,
        "mm_win_rate": mm_win_rate,
        "spread_earned": total_spread_earned,
        "inventory_pnl": total_inventory_pnl,
        "total_pnl": total_pnl,
        "mm_return": mm_return,
    }


def main():
    results = run_mm_backtest()
    print("--- MM BACKTEST ---")
    print(f"markets_traded:    {results['markets_traded']}")
    print(f"markets_profitable:{results['markets_profitable']}")
    print(f"mm_win_rate:       {results['mm_win_rate']:.4f}")
    print(f"spread_earned:     {results['spread_earned']:.2f}")
    print(f"inventory_pnl:     {results['inventory_pnl']:.2f}")
    print(f"total_pnl:         {results['total_pnl']:.2f}")
    print(f"mm_return:         {results['mm_return']:.4f}")


if __name__ == "__main__":
    main()

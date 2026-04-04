"""Hybrid backtest: combines directional + market making on the same capital."""

from bot.research.evaluate import run_evaluation, STARTING_BANKROLL
from bot.backtest.mm_backtest import run_mm_backtest


def main():
    # Run directional component
    trades, final_bankroll, max_dd = run_evaluation(seed=42)
    dir_num_trades = len(trades)
    dir_wins = sum(1 for t in trades if t["won"])
    dir_win_rate = dir_wins / dir_num_trades if dir_num_trades > 0 else 0.0
    dir_pnl = final_bankroll - STARTING_BANKROLL

    # Run market making component
    mm_results = run_mm_backtest()
    mm_markets = mm_results["markets_traded"]
    mm_win_rate = mm_results["mm_win_rate"]
    mm_pnl = mm_results["total_pnl"]

    # Combined
    combined_pnl = dir_pnl + mm_pnl
    combined_return = combined_pnl / STARTING_BANKROLL if STARTING_BANKROLL > 0 else 0.0

    # Revenue split
    total_abs = abs(dir_pnl) + abs(mm_pnl)
    if total_abs > 0:
        dir_pct = int(round(100 * abs(dir_pnl) / total_abs))
        mm_pct = 100 - dir_pct
    else:
        dir_pct = 50
        mm_pct = 50

    print("=" * 60)
    print("HYBRID BACKTEST: Market Making + Directional")
    print("=" * 60)
    print()
    print("--- Directional Component ---")
    print(f"  Trades: {dir_num_trades}, Win rate: {dir_win_rate:.0%}, P&L: ${dir_pnl:,.2f}")
    print()
    print("--- Market Making Component ---")
    print(f"  Markets: {mm_markets}, Win rate: {mm_win_rate:.0%}, P&L: ${mm_pnl:,.2f}")
    print()
    print("--- COMBINED ---")
    print(f"  Directional P&L: ${dir_pnl:>12,.2f}")
    print(f"  Market Making P&L: ${mm_pnl:>10,.2f}")
    print(f"  Combined P&L: ${combined_pnl:>14,.2f}")
    print(f"  Combined return: {combined_return:.1%}")
    print(f"  Revenue split: {dir_pct}% directional / {mm_pct}% MM")


if __name__ == "__main__":
    main()

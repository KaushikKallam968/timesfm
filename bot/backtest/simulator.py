from dataclasses import dataclass
import math
from bot.execution.kelly import kelly_size


@dataclass
class SimTrade:
    timestamp: str
    market_id: str
    side: str
    entry_price: float
    truth_probability: float
    edge: float
    size: float
    outcome: str
    pnl: float


@dataclass
class BacktestResult:
    trades: list
    equity_curve: list
    metrics: dict


class Simulator:
    def __init__(self, edge_threshold=0.08, kelly_fraction=0.15, max_trade_size=100, initial_bankroll=500):
        self.edge_threshold = edge_threshold
        self.kelly_fraction = kelly_fraction
        self.max_trade_size = max_trade_size
        self.initial_bankroll = initial_bankroll

    def run(self, historical_data):
        bankroll = self.initial_bankroll
        trades = []
        equity_curve = [bankroll]

        for dp in historical_data:
            edge = dp["truth_probability"] - dp["market_price"]

            if abs(edge) < self.edge_threshold:
                continue

            if edge > 0:
                side = "YES"
                entry_price = dp["market_price"]
                prob = dp["truth_probability"]
            else:
                side = "NO"
                entry_price = 1 - dp["market_price"]
                prob = 1 - dp["truth_probability"]
                edge = abs(edge)

            odds = (1 - entry_price) / entry_price if entry_price > 0 else 0
            size = kelly_size(prob, odds, bankroll, fraction=self.kelly_fraction, max_size=self.max_trade_size)

            if size <= 0:
                continue

            won = (side == "YES" and dp["actual_outcome"] == "YES") or \
                  (side == "NO" and dp["actual_outcome"] == "NO")

            if won:
                pnl = size * odds
                outcome = "WIN"
            else:
                pnl = -size
                outcome = "LOSS"

            bankroll += pnl
            equity_curve.append(bankroll)

            trades.append(SimTrade(
                timestamp=dp["timestamp"],
                market_id=dp["market_id"],
                side=side,
                entry_price=entry_price,
                truth_probability=dp["truth_probability"],
                edge=edge,
                size=size,
                outcome=outcome,
                pnl=pnl,
            ))

        metrics = self._compute_metrics(trades, equity_curve)
        return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)

    def _compute_metrics(self, trades, equity_curve):
        total_trades = len(trades)

        if total_trades == 0:
            return {
                "win_rate": 0,
                "sharpe": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "total_return": 0,
                "avg_edge": 0,
                "total_trades": 0,
            }

        wins = [t for t in trades if t.outcome == "WIN"]
        win_rate = len(wins) / total_trades

        total_wins = sum(t.pnl for t in trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        total_return = (equity_curve[-1] - self.initial_bankroll) / self.initial_bankroll

        max_drawdown = self._max_drawdown(equity_curve)

        avg_edge = sum(t.edge for t in trades) / total_trades

        sharpe = self._sharpe(equity_curve)

        return {
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "avg_edge": avg_edge,
            "total_trades": total_trades,
        }

    def _max_drawdown(self, equity_curve):
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _sharpe(self, equity_curve):
        if len(equity_curve) < 2:
            return 0

        returns = []
        for i in range(1, len(equity_curve)):
            r = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] if equity_curve[i - 1] != 0 else 0
            returns.append(r)

        if not returns:
            return 0

        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(variance)

        if std_r == 0:
            return float("inf") if mean_r > 0 else 0

        return (mean_r / std_r) * math.sqrt(365)

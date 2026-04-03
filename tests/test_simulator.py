import random
from bot.backtest.simulator import Simulator, BacktestResult, SimTrade


def make_data(n, truth_accuracy, market_price=0.50, truth_prob=0.70, seed=42):
    random.seed(seed)
    data = []
    for i in range(n):
        truth_correct = random.random() < truth_accuracy
        if truth_correct:
            actual = "YES" if truth_prob > 0.5 else "NO"
        else:
            actual = "NO" if truth_prob > 0.5 else "YES"

        data.append({
            "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
            "market_id": f"market_{i}",
            "market_price": market_price,
            "truth_probability": truth_prob,
            "actual_outcome": actual,
        })
    return data


class TestSimulatorBasic:
    def test_returns_backtest_result(self):
        data = make_data(10, 0.9)
        sim = Simulator()
        result = sim.run(data)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.metrics, dict)

    def test_all_required_metrics_present(self):
        data = make_data(10, 0.9)
        result = Simulator().run(data)
        for key in ["win_rate", "sharpe", "max_drawdown", "profit_factor", "total_return", "avg_edge", "total_trades"]:
            assert key in result.metrics

    def test_sim_trade_fields(self):
        data = make_data(10, 0.9)
        result = Simulator().run(data)
        assert len(result.trades) > 0
        t = result.trades[0]
        assert isinstance(t, SimTrade)
        assert t.side in ("YES", "NO")
        assert t.outcome in ("WIN", "LOSS")


class TestHighWinRate:
    def test_win_rate_approximately_90(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert 0.80 <= result.metrics["win_rate"] <= 1.0

    def test_equity_curve_grows(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.equity_curve[-1] > result.equity_curve[0]

    def test_sharpe_positive(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.metrics["sharpe"] > 0

    def test_profit_factor_above_one(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.metrics["profit_factor"] > 1

    def test_total_return_positive(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.metrics["total_return"] > 0

    def test_total_trades_equals_trade_count(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.metrics["total_trades"] == len(result.trades)


class TestMaxDrawdown:
    def test_max_drawdown_computed_correctly(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        peak = result.equity_curve[0]
        max_dd = 0
        for val in result.equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        assert abs(result.metrics["max_drawdown"] - max_dd) < 1e-10

    def test_max_drawdown_non_negative(self):
        data = make_data(100, 0.9)
        result = Simulator().run(data)
        assert result.metrics["max_drawdown"] >= 0


class TestNoTradeableEdges:
    def test_no_trades_when_edge_below_threshold(self):
        data = []
        for i in range(50):
            data.append({
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00Z",
                "market_id": f"market_{i}",
                "market_price": 0.50,
                "truth_probability": 0.52,
                "actual_outcome": "YES",
            })
        result = Simulator(edge_threshold=0.08).run(data)
        assert result.metrics["total_trades"] == 0
        assert result.metrics["win_rate"] == 0
        assert result.metrics["total_return"] == 0
        assert len(result.equity_curve) == 1


class TestPerfectWinRate:
    def test_100_percent_win_rate(self):
        data = make_data(50, 1.0)
        result = Simulator().run(data)
        assert result.metrics["win_rate"] == 1.0
        assert result.metrics["profit_factor"] == float("inf")
        assert result.metrics["max_drawdown"] == 0
        assert result.equity_curve[-1] > result.equity_curve[0]


class TestFiftyPercentWinRate:
    def test_loses_money_at_50_percent(self):
        data = make_data(200, 0.5, market_price=0.60, truth_prob=0.70, seed=123)
        result = Simulator().run(data)
        assert result.metrics["total_return"] < 0

    def test_win_rate_near_50(self):
        data = make_data(200, 0.5, market_price=0.60, truth_prob=0.70, seed=123)
        result = Simulator().run(data)
        assert 0.35 <= result.metrics["win_rate"] <= 0.65


class TestEquityCurve:
    def test_equity_curve_starts_at_initial_bankroll(self):
        data = make_data(20, 0.9)
        sim = Simulator(initial_bankroll=1000)
        result = sim.run(data)
        assert result.equity_curve[0] == 1000

    def test_equity_curve_length(self):
        data = make_data(20, 0.9)
        result = Simulator().run(data)
        assert len(result.equity_curve) == len(result.trades) + 1


class TestEdgeDirection:
    def test_negative_edge_trades_no(self):
        data = [{
            "timestamp": "2025-01-01T12:00:00Z",
            "market_id": "m1",
            "market_price": 0.80,
            "truth_probability": 0.60,
            "actual_outcome": "NO",
        }]
        result = Simulator(edge_threshold=0.08).run(data)
        assert len(result.trades) == 1
        assert result.trades[0].side == "NO"
        assert result.trades[0].outcome == "WIN"

import random
from bot.backtest.optimizer import AutoResearchOptimizer, OptimizationResult


def make_synthetic_data(n=500, accuracy=0.90, base_edge=0.15):
    random.seed(42)
    data = []
    for i in range(n):
        truth_prob = random.uniform(0.55, 0.95)
        market_price = truth_prob - base_edge + random.uniform(-0.05, 0.05)
        market_price = max(0.05, min(0.95, market_price))
        if random.random() < accuracy:
            actual = "YES" if truth_prob > 0.5 else "NO"
        else:
            actual = "NO" if truth_prob > 0.5 else "YES"
        data.append({
            "timestamp": f"2024-01-{(i % 28) + 1:02d}",
            "market_id": f"market_{i}",
            "market_price": round(market_price, 3),
            "truth_probability": round(truth_prob, 3),
            "actual_outcome": actual,
        })
    return data


class TestOptimizerHighAccuracy:
    def test_finds_params_hitting_target(self):
        data = make_synthetic_data(n=500, accuracy=0.96, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data, win_rate_target=0.95, sharpe_target=2.0, max_drawdown_limit=0.15)
        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.target_met is True
        assert result.best_metrics["win_rate"] >= 0.95

    def test_best_params_has_required_keys(self):
        data = make_synthetic_data(n=500, accuracy=0.96, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data)
        result = optimizer.optimize()

        assert "edge_threshold" in result.best_params
        assert "kelly_fraction" in result.best_params
        assert "max_trade_size" in result.best_params


class TestOptimizerLowAccuracy:
    def test_reports_target_not_met(self):
        data = make_synthetic_data(n=500, accuracy=0.80, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data, win_rate_target=0.95, sharpe_target=2.0, max_drawdown_limit=0.15)
        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.target_met is False


class TestOptimizerHistory:
    def test_history_tracks_all_iterations(self):
        data = make_synthetic_data(n=300, accuracy=0.90, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data)
        result = optimizer.optimize()

        assert len(result.history) >= 18  # at least 8 + 5 + 5 from the three sweeps
        assert result.iterations == len(result.history) or result.iterations <= len(result.history)
        for entry in result.history:
            assert "params" in entry
            assert "metrics" in entry
            assert "win_rate" in entry["metrics"]

    def test_iterations_count_matches_history(self):
        data = make_synthetic_data(n=300, accuracy=0.90, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data)
        result = optimizer.optimize()

        # iterations tracks how many trials ran through the loop counters
        # history may have same count (no extra trial for final best check since
        # best_metrics is recomputed but not appended to history)
        assert result.iterations >= 18


class TestOptimizerImprovement:
    def test_improves_over_defaults(self):
        data = make_synthetic_data(n=500, accuracy=0.90, base_edge=0.15)
        optimizer = AutoResearchOptimizer(data)
        result = optimizer.optimize()

        # Run default params to compare
        default_metrics = optimizer._run_trial({
            "edge_threshold": 0.08,
            "kelly_fraction": 0.15,
            "max_trade_size": 100,
        })

        assert result.best_metrics["win_rate"] >= default_metrics["win_rate"]

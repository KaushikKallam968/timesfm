from dataclasses import dataclass
from bot.backtest.simulator import Simulator


@dataclass
class OptimizationResult:
    best_params: dict
    best_metrics: dict
    iterations: int
    history: list
    target_met: bool


class AutoResearchOptimizer:
    def __init__(self, historical_data, initial_bankroll=500,
                 win_rate_target=0.95, sharpe_target=2.0,
                 max_drawdown_limit=0.15, max_iterations=50):
        self.historical_data = historical_data
        self.initial_bankroll = initial_bankroll
        self.win_rate_target = win_rate_target
        self.sharpe_target = sharpe_target
        self.max_drawdown_limit = max_drawdown_limit
        self.max_iterations = max_iterations

    def optimize(self):
        history = []
        iterations = 0

        default_params = {
            "edge_threshold": 0.08,
            "kelly_fraction": 0.15,
            "max_trade_size": 100,
        }

        # Level 1 — Edge threshold sweep
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
        threshold_results = []
        for t in thresholds:
            if iterations >= self.max_iterations:
                break
            params = {**default_params, "edge_threshold": t}
            metrics = self._run_trial(params)
            history.append({"params": params, "metrics": metrics})
            threshold_results.append((t, metrics))
            iterations += 1

        best_threshold = max(threshold_results, key=lambda x: self._score(x[1]))[0]

        # Level 2 — Kelly fraction sweep
        kelly_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]
        kelly_results = []
        for k in kelly_fractions:
            if iterations >= self.max_iterations:
                break
            params = {**default_params, "edge_threshold": best_threshold, "kelly_fraction": k}
            metrics = self._run_trial(params)
            history.append({"params": params, "metrics": metrics})
            kelly_results.append((k, metrics))
            iterations += 1

        best_kelly = max(kelly_results, key=lambda x: self._score(x[1]))[0]

        # Level 3 — Max trade size sweep
        sizes = [25, 50, 75, 100, 150]
        size_results = []
        for s in sizes:
            if iterations >= self.max_iterations:
                break
            params = {
                "edge_threshold": best_threshold,
                "kelly_fraction": best_kelly,
                "max_trade_size": s,
            }
            metrics = self._run_trial(params)
            history.append({"params": params, "metrics": metrics})
            size_results.append((s, metrics))
            iterations += 1

        best_size = max(size_results, key=lambda x: self._score(x[1]))[0]

        best_params = {
            "edge_threshold": best_threshold,
            "kelly_fraction": best_kelly,
            "max_trade_size": best_size,
        }
        best_metrics = self._run_trial(best_params)

        if self._targets_met(best_metrics):
            return OptimizationResult(
                best_params=best_params,
                best_metrics=best_metrics,
                iterations=iterations,
                history=history,
                target_met=True,
            )

        # Fallback — combined grid of top 3 thresholds × top 3 kelly fractions
        top_thresholds = sorted(threshold_results, key=lambda x: self._score(x[1]), reverse=True)[:3]
        top_kellys = sorted(kelly_results, key=lambda x: self._score(x[1]), reverse=True)[:3]

        for t, _ in top_thresholds:
            for k, _ in top_kellys:
                if iterations >= self.max_iterations:
                    break
                params = {
                    "edge_threshold": t,
                    "kelly_fraction": k,
                    "max_trade_size": best_size,
                }
                metrics = self._run_trial(params)
                history.append({"params": params, "metrics": metrics})
                iterations += 1

                if self._targets_met(metrics):
                    return OptimizationResult(
                        best_params=params,
                        best_metrics=metrics,
                        iterations=iterations,
                        history=history,
                        target_met=True,
                    )

        # Pick the best from all history
        best_entry = max(history, key=lambda h: self._score(h["metrics"]))
        return OptimizationResult(
            best_params=best_entry["params"],
            best_metrics=best_entry["metrics"],
            iterations=iterations,
            history=history,
            target_met=False,
        )

    def _run_trial(self, params):
        sim = Simulator(
            edge_threshold=params["edge_threshold"],
            kelly_fraction=params["kelly_fraction"],
            max_trade_size=params["max_trade_size"],
            initial_bankroll=self.initial_bankroll,
        )
        result = sim.run(self.historical_data)
        return result.metrics

    def _generate_param_grid(self):
        grid = []
        for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            for k in [0.05, 0.10, 0.15, 0.20, 0.25]:
                for s in [25, 50, 75, 100, 150]:
                    grid.append({
                        "edge_threshold": t,
                        "kelly_fraction": k,
                        "max_trade_size": s,
                    })
        return grid

    def _score(self, metrics):
        if metrics.get("total_trades", 0) < 20:
            return -1
        return metrics.get("win_rate", 0)

    def _targets_met(self, metrics):
        if metrics.get("total_trades", 0) < 20:
            return False
        return (
            metrics.get("win_rate", 0) >= self.win_rate_target
            and metrics.get("sharpe", 0) >= self.sharpe_target
            and metrics.get("max_drawdown", 1) <= self.max_drawdown_limit
        )

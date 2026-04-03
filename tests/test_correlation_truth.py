from bot.truth.correlation import CorrelationEngine


def make_outcomes(prices, names=None):
    if names is None:
        names = [f"outcome_{i}" for i in range(len(prices))]
    return [{"name": n, "price": p} for n, p in zip(names, prices)]


class TestSumViolations:
    def setup_method(self):
        self.engine = CorrelationEngine()

    def test_sum_violation_prices_090(self):
        outcomes = make_outcomes([0.30, 0.30, 0.30])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 1
        assert violations[0]["total"] == pytest.approx(0.90)
        assert violations[0]["deviation"] == pytest.approx(-0.10)
        assert len(violations[0]["underpriced_outcomes"]) == 3

    def test_sum_violation_prices_095(self):
        outcomes = make_outcomes([0.45, 0.50])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 1
        assert violations[0]["total"] == pytest.approx(0.95)
        assert violations[0]["deviation"] == pytest.approx(-0.05)

    def test_sum_violation_prices_105(self):
        outcomes = make_outcomes([0.55, 0.50])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 1
        assert violations[0]["total"] == pytest.approx(1.05)
        assert violations[0]["deviation"] == pytest.approx(0.05)

    def test_sum_violation_prices_110(self):
        outcomes = make_outcomes([0.60, 0.50])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 1
        assert violations[0]["total"] == pytest.approx(1.10)
        assert violations[0]["deviation"] == pytest.approx(0.10)

    def test_no_violation_near_100(self):
        outcomes = make_outcomes([0.49, 0.51])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 0

    def test_no_violation_at_boundary(self):
        outcomes = make_outcomes([0.48, 0.52])
        violations = self.engine.find_sum_violations(outcomes)
        assert len(violations) == 0


class TestSubsetViolations:
    def setup_method(self):
        self.engine = CorrelationEngine()

    def test_subset_violation_detected(self):
        specific = {"name": "Trump wins", "price": 0.60}
        general = {"name": "Republican wins", "price": 0.50}
        result = self.engine.find_subset_violations(specific, general)
        assert result is not None
        assert result["edge"] == pytest.approx(0.10)

    def test_no_subset_violation_when_properly_ordered(self):
        specific = {"name": "Trump wins", "price": 0.40}
        general = {"name": "Republican wins", "price": 0.60}
        result = self.engine.find_subset_violations(specific, general)
        assert result is None


class TestCanHandle:
    def setup_method(self):
        self.engine = CorrelationEngine()

    def test_can_handle_with_outcomes(self):
        market = {"outcomes": [{"name": "A", "price": 0.5}]}
        assert self.engine.can_handle(market) is True

    def test_can_handle_with_related_markets(self):
        market = {"related_markets": []}
        assert self.engine.can_handle(market) is True

    def test_cannot_handle_without_fields(self):
        market = {"question": "Will it rain?"}
        assert self.engine.can_handle(market) is False


class TestGetTruth:
    def setup_method(self):
        self.engine = CorrelationEngine()

    def test_get_truth_sum_violation_returns_correct_probability(self):
        market = {"outcomes": make_outcomes([0.30, 0.30, 0.30])}
        result = self.engine.get_truth(market)
        assert result is not None
        assert result.source == "correlation_sum"
        deviation_share = 0.10 / 3
        expected_fair = 0.30 + deviation_share
        assert result.probability == pytest.approx(expected_fair, abs=0.001)

    def test_get_truth_confidence_always_one(self):
        market = {"outcomes": make_outcomes([0.30, 0.30, 0.30])}
        result = self.engine.get_truth(market)
        assert result is not None
        assert result.confidence == 1.0

    def test_get_truth_subset_violation_confidence_one(self):
        specific = {"name": "Trump wins", "price": 0.60, "subset_of": {"name": "Republican wins", "price": 0.50}}
        market = {"related_markets": [specific]}
        result = self.engine.get_truth(market)
        assert result is not None
        assert result.confidence == 1.0
        assert result.source == "correlation_subset"

    def test_get_truth_returns_none_when_no_violations(self):
        market = {"outcomes": make_outcomes([0.49, 0.51])}
        result = self.engine.get_truth(market)
        assert result is None


import pytest

from bot.truth.base import TruthEngine, TruthResult
import pytest


def test_truth_engine_is_abstract():
    with pytest.raises(TypeError):
        TruthEngine()


def test_truth_result_has_required_fields():
    result = TruthResult(probability=0.85, confidence=0.9, source="test")
    assert result.probability == 0.85
    assert result.confidence == 0.9
    assert result.source == "test"


def test_truth_result_edge_calculation():
    result = TruthResult(probability=0.85, confidence=0.9, source="test")
    assert result.edge(market_price=0.65) == pytest.approx(0.20)


def test_truth_result_negative_edge():
    result = TruthResult(probability=0.50, confidence=0.9, source="test")
    assert result.edge(market_price=0.70) == pytest.approx(-0.20)

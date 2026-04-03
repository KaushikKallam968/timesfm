import pytest
from bot.truth.timesfm_enhancer import TimesFMEnhancer


@pytest.fixture
def enhancer():
    return TimesFMEnhancer(model=None, mock_mode=True)


def test_predict_returns_correct_structure(enhancer):
    history = [0.5 + i * 0.01 for i in range(30)]
    result = enhancer.predict_odds_trajectory(history, horizon=10)
    assert "predicted_odds" in result
    assert "direction" in result
    assert "magnitude" in result
    assert "confidence" in result
    assert "spread_widening" in result
    assert len(result["predicted_odds"]) == 10


def test_direction_up_when_trending_up(enhancer):
    history = [0.3 + i * 0.02 for i in range(20)]
    result = enhancer.predict_odds_trajectory(history)
    assert result["direction"] == "up"


def test_direction_down_when_trending_down(enhancer):
    history = [0.8 - i * 0.02 for i in range(20)]
    result = enhancer.predict_odds_trajectory(history)
    assert result["direction"] == "down"


def test_direction_stable_for_flat_odds(enhancer):
    history = [0.5] * 20
    result = enhancer.predict_odds_trajectory(history)
    assert result["direction"] == "stable"


def test_magnitude_larger_for_volatile_odds(enhancer):
    calm = [0.5 + i * 0.001 for i in range(20)]
    volatile = [0.5 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
    calm_result = enhancer.predict_odds_trajectory(calm)
    volatile_result = enhancer.predict_odds_trajectory(volatile)
    assert volatile_result["magnitude"] > calm_result["magnitude"]


def test_confidence_higher_for_consistent_trends(enhancer):
    consistent = [0.3 + i * 0.02 for i in range(20)]
    noisy = [0.3 + i * 0.02 + (0.03 if i % 2 == 0 else -0.03) for i in range(20)]
    consistent_result = enhancer.predict_odds_trajectory(consistent)
    noisy_result = enhancer.predict_odds_trajectory(noisy)
    assert consistent_result["confidence"] > noisy_result["confidence"]


def test_should_enter_now_when_edge_closing(enhancer):
    # Positive edge, odds trending up (market correcting toward our view)
    history = [0.4 + i * 0.01 for i in range(20)]
    result = enhancer.should_enter_now(history, current_edge=0.15)
    assert result["action"] == "enter_now"


def test_should_wait_when_edge_widening(enhancer):
    # Positive edge, odds trending down (market moving away from our view)
    history = [0.6 - i * 0.01 for i in range(20)]
    result = enhancer.should_enter_now(history, current_edge=0.15)
    assert result["action"] == "wait"
    assert result["expected_better_entry"] is not None


def test_should_skip_when_edge_tiny(enhancer):
    history = [0.5] * 20
    result = enhancer.should_enter_now(history, current_edge=0.01)
    assert result["action"] == "skip"


def test_rank_markets_by_timing_sorts_correctly(enhancer):
    markets = [
        {"market_id": "A", "odds_history": [0.5] * 20, "current_edge": 0.01},  # skip
        {"market_id": "B", "odds_history": [0.3 + i * 0.02 for i in range(20)], "current_edge": 0.15},  # enter_now
        {"market_id": "C", "odds_history": [0.6 - i * 0.01 for i in range(20)], "current_edge": 0.15},  # wait
    ]
    ranked = enhancer.rank_markets_by_timing(markets)
    ids = [m["market_id"] for m in ranked]
    assert ids[0] == "B"
    assert ids[-1] == "A"


def test_handles_short_history(enhancer):
    history = [0.5, 0.6]
    result = enhancer.predict_odds_trajectory(history)
    assert result["direction"] in ("up", "down", "stable")
    assert isinstance(result["predicted_odds"], list)


def test_handles_empty_history(enhancer):
    result = enhancer.predict_odds_trajectory([])
    assert result["predicted_odds"] == []
    assert result["direction"] == "stable"
    assert result["confidence"] == 0.0


def test_handles_single_value_history(enhancer):
    result = enhancer.predict_odds_trajectory([0.5])
    assert result["predicted_odds"] == []
    assert result["direction"] == "stable"


def test_mock_mode_defaults_when_no_model():
    e = TimesFMEnhancer(model=None, mock_mode=False)
    assert e.mock_mode is True


def test_predicted_odds_clamped_to_0_1(enhancer):
    # Very strong upward trend that could extrapolate past 1.0
    history = [0.9 + i * 0.02 for i in range(15)]
    result = enhancer.predict_odds_trajectory(history, horizon=30)
    assert all(0.0 <= v <= 1.0 for v in result["predicted_odds"])

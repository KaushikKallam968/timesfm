from bot.execution.kelly import kelly_size


def test_basic_kelly_size():
    result = kelly_size(edge=0.6, odds=2.0, bankroll=1000)
    # kelly = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
    # size = 0.4 * 0.15 * 1000 = 60
    assert abs(result - 60.0) < 0.01


def test_zero_edge_returns_zero():
    assert kelly_size(edge=0, odds=2.0, bankroll=1000) == 0


def test_negative_edge_returns_zero():
    assert kelly_size(edge=-0.1, odds=2.0, bankroll=1000) == 0


def test_cap_at_max_size():
    result = kelly_size(edge=0.8, odds=3.0, bankroll=10000, fraction=0.5)
    # kelly = (0.8 * 3.0 - 0.2) / 3.0 = 0.7333
    # size = 0.7333 * 0.5 * 10000 = 3666.67 -> capped at 100
    assert result == 100


def test_custom_max_size():
    result = kelly_size(edge=0.8, odds=3.0, bankroll=10000, fraction=0.5, max_size=500)
    assert result == 500


def test_custom_fraction():
    result = kelly_size(edge=0.6, odds=2.0, bankroll=1000, fraction=0.25)
    # kelly = 0.4, size = 0.4 * 0.25 * 1000 = 100 -> capped at 100
    assert abs(result - 100) < 0.01


def test_small_edge():
    result = kelly_size(edge=0.51, odds=2.0, bankroll=1000)
    # kelly = (0.51 * 2.0 - 0.49) / 2.0 = 0.265
    # size = 0.265 * 0.15 * 1000 = 39.75
    assert abs(result - 39.75) < 0.01


def test_low_odds():
    result = kelly_size(edge=0.7, odds=1.5, bankroll=1000)
    # kelly = (0.7 * 1.5 - 0.3) / 1.5 = 0.5
    # size = 0.5 * 0.15 * 1000 = 75
    assert abs(result - 75.0) < 0.01


def test_returns_float():
    result = kelly_size(edge=0.6, odds=2.0, bankroll=1000)
    assert isinstance(result, float)

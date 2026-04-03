from bot.truth.weather import WeatherEngine, CITY_COORDINATES
from bot.truth.base import TruthResult


def make_engine(mock_data=None):
    engine = WeatherEngine()
    if mock_data is not None:
        engine._mock_ensemble = mock_data
    return engine


# --- can_handle ---

def test_can_handle_weather_category():
    engine = make_engine()
    assert engine.can_handle({"category": "weather", "question": "anything"})

def test_can_handle_temperature_keyword():
    engine = make_engine()
    assert engine.can_handle({"question": "Will the temperature in NYC exceed 80°F?"})

def test_can_handle_temp_keyword():
    engine = make_engine()
    assert engine.can_handle({"question": "Will NYC high temp exceed 75°F on April 5?"})

def test_cannot_handle_unrelated():
    engine = make_engine()
    assert not engine.can_handle({"category": "politics", "question": "Will candidate X win?"})


# --- _parse_weather_market ---

def test_parse_nyc_above():
    engine = make_engine()
    result = engine._parse_weather_market({
        "question": "Will NYC high temp exceed 75°F on April 5?"
    })
    assert result is not None
    assert result["city"] == "New York"
    assert result["lat"] == 40.71
    assert result["lon"] == -74.01
    assert result["threshold"] == 75.0
    assert result["comparison"] == "above"
    assert result["date"] == "2026-04-05"

def test_parse_below():
    engine = make_engine()
    result = engine._parse_weather_market({
        "question": "Will Chicago temp drop below 32°F on 2026-01-15?"
    })
    assert result is not None
    assert result["city"] == "Chicago"
    assert result["threshold"] == 32.0
    assert result["comparison"] == "below"
    assert result["date"] == "2026-01-15"

def test_parse_unknown_city_returns_none():
    engine = make_engine()
    result = engine._parse_weather_market({
        "question": "Will Tokyo temp exceed 90°F on April 5?"
    })
    assert result is None

def test_parse_no_date_returns_none():
    engine = make_engine()
    result = engine._parse_weather_market({
        "question": "Will NYC temp exceed 90°F?"
    })
    assert result is None


# --- City coordinates ---

def test_city_coordinates_lookup():
    assert "nyc" in CITY_COORDINATES
    assert CITY_COORDINATES["denver"]["lat"] == 39.74
    assert CITY_COORDINATES["london"]["lon"] == -0.13
    assert CITY_COORDINATES["seoul"]["lat"] == 37.57


# --- _ensemble_probability ---

def test_ensemble_probability_all_above():
    engine = make_engine()
    members = [80.0] * 31
    prob = engine._ensemble_probability(members, 75.0, "above")
    assert prob == 1.0

def test_ensemble_probability_none_above():
    engine = make_engine()
    members = [70.0] * 31
    prob = engine._ensemble_probability(members, 75.0, "above")
    assert prob == 0.0

def test_ensemble_probability_partial():
    engine = make_engine()
    members = [80.0] * 20 + [70.0] * 11
    prob = engine._ensemble_probability(members, 75.0, "above")
    assert abs(prob - 20 / 31) < 0.001

def test_ensemble_probability_below():
    engine = make_engine()
    members = [30.0] * 25 + [35.0] * 6
    prob = engine._ensemble_probability(members, 32.0, "below")
    assert abs(prob - 25 / 31) < 0.001

def test_ensemble_probability_empty():
    engine = make_engine()
    assert engine._ensemble_probability([], 75.0, "above") == 0.5


# --- _confidence_from_ensemble ---

def test_confidence_high_agreement():
    engine = make_engine()
    members = [80.0] * 30 + [70.0] * 1  # 30/31 above = ~96.8%
    assert engine._confidence_from_ensemble(members, 75.0) == 0.95

def test_confidence_medium_agreement():
    engine = make_engine()
    members = [80.0] * 24 + [70.0] * 7  # 24/31 above = ~77.4%
    assert engine._confidence_from_ensemble(members, 75.0) == 0.85

def test_confidence_low_agreement():
    engine = make_engine()
    members = [80.0] * 18 + [70.0] * 13  # 18/31 = ~58%
    assert engine._confidence_from_ensemble(members, 75.0) == 0.70

def test_confidence_very_low_agreement():
    engine = make_engine()
    members = [80.0] * 15 + [70.0] * 16  # 16/31 = ~51.6%, max agreement ~51.6%
    assert engine._confidence_from_ensemble(members, 75.0) == 0.70


# --- get_truth with mock data ---

def test_get_truth_mock_above():
    engine = make_engine(mock_data=[80.0] * 28 + [70.0] * 3)
    market = {"question": "Will NYC high temp exceed 75°F on 2026-04-05?"}
    result = engine.get_truth(market)
    assert result is not None
    assert isinstance(result, TruthResult)
    assert abs(result.probability - 28 / 31) < 0.001
    assert result.confidence == 0.95
    assert result.source == "weather_ensemble_gfs"

def test_get_truth_unparseable_returns_none():
    engine = make_engine(mock_data=[80.0] * 31)
    market = {"question": "Some random question with no weather info"}
    assert engine.get_truth(market) is None

def test_get_truth_no_ensemble_returns_none():
    engine = make_engine(mock_data=None)
    # fetch_ensemble returns None when _mock_ensemble is None and API fails
    engine._mock_ensemble = None
    # Force fetch to return None by setting mock to explicit None-like
    engine.fetch_ensemble = lambda lat, lon, date: None
    market = {"question": "Will NYC high temp exceed 75°F on 2026-04-05?"}
    assert engine.get_truth(market) is None

def test_get_truth_edge_calculation():
    engine = make_engine(mock_data=[90.0] * 25 + [80.0] * 6)
    market = {"question": "Will Miami high temp exceed 85°F on 2026-04-05?"}
    result = engine.get_truth(market)
    assert result is not None
    edge = result.edge(0.50)
    expected_prob = 25 / 31
    assert abs(edge - (expected_prob - 0.50)) < 0.001

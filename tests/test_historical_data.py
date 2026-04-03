import os
import tempfile
from bot.backtest.historical_data import (
    generate_synthetic_sports,
    generate_synthetic_weather,
    generate_synthetic_correlation,
    generate_all_synthetic,
    save_historical,
    load_historical,
)


def test_sports_data_count():
    data = generate_synthetic_sports(n=100)
    assert len(data) == 100


def test_sports_data_fields():
    data = generate_synthetic_sports(n=10)
    for d in data:
        assert "timestamp" in d
        assert "market_id" in d
        assert "market_price" in d
        assert "truth_probability" in d
        assert "actual_outcome" in d
        assert d["actual_outcome"] in ("YES", "NO")
        assert 0 < d["market_price"] < 1
        assert 0 < d["truth_probability"] < 1


def test_sports_accuracy():
    data = generate_synthetic_sports(n=5000, accuracy=0.90, seed=999)
    correct = sum(
        1 for d in data
        if (d["truth_probability"] > 0.5 and d["actual_outcome"] == "YES")
        or (d["truth_probability"] <= 0.5 and d["actual_outcome"] == "NO")
    )
    assert 0.85 < correct / len(data) < 0.95


def test_weather_data_has_city():
    data = generate_synthetic_weather(n=10)
    for d in data:
        assert "city" in d
        assert d["category"] == "weather"


def test_correlation_data_high_accuracy():
    data = generate_synthetic_correlation(n=2000, accuracy=0.97, seed=789)
    correct = sum(
        1 for d in data
        if (d["truth_probability"] > 0.5 and d["actual_outcome"] == "YES")
        or (d["truth_probability"] <= 0.5 and d["actual_outcome"] == "NO")
    )
    assert correct / len(data) > 0.94


def test_combined_dataset():
    data = generate_all_synthetic(sports_n=100, weather_n=50, corr_n=20)
    assert len(data) == 170
    cats = set(d["category"] for d in data)
    assert cats == {"sports", "weather", "correlation"}


def test_sorted_by_timestamp():
    data = generate_all_synthetic(sports_n=50, weather_n=30, corr_n=10)
    timestamps = [d["timestamp"] for d in data]
    assert timestamps == sorted(timestamps)


def test_save_and_load(tmp_path):
    import bot.backtest.historical_data as hd
    old_dir = hd.DATA_DIR
    hd.DATA_DIR = str(tmp_path)
    try:
        data = generate_all_synthetic(sports_n=10, weather_n=5, corr_n=3)
        save_historical(data, "test.json")
        loaded = load_historical("test.json")
        assert loaded == data
    finally:
        hd.DATA_DIR = old_dir


def test_load_missing_returns_none(tmp_path):
    import bot.backtest.historical_data as hd
    old_dir = hd.DATA_DIR
    hd.DATA_DIR = str(tmp_path)
    try:
        assert load_historical("nonexistent.json") is None
    finally:
        hd.DATA_DIR = old_dir


def test_deterministic_with_seed():
    d1 = generate_synthetic_sports(n=50, seed=42)
    d2 = generate_synthetic_sports(n=50, seed=42)
    assert d1 == d2

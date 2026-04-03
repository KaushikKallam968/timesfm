import pytest
from bot.market.polymarket import PolymarketClient
from bot.market.scanner import MarketScanner
from bot.truth.base import TruthEngine, TruthResult


class FakeSportsEngine(TruthEngine):
    def can_handle(self, market):
        return market.get("category") == "sports"

    def get_truth(self, market):
        return TruthResult(probability=0.45, confidence=0.85, source="fake_sports")


class FakeWeatherEngine(TruthEngine):
    def can_handle(self, market):
        return market.get("category") == "weather"

    def get_truth(self, market):
        return TruthResult(probability=0.15, confidence=0.90, source="fake_weather")


class FakeNoOpEngine(TruthEngine):
    def can_handle(self, market):
        return False

    def get_truth(self, market):
        return None


@pytest.fixture
def client():
    return PolymarketClient(mock_mode=True)


@pytest.fixture
def scanner(client):
    engines = [FakeSportsEngine(), FakeWeatherEngine()]
    return MarketScanner(client, engines)


def test_scanner_finds_sports_markets(scanner):
    opportunities = scanner.scan_all()
    sports = [o for o in opportunities if o["market"]["category"] == "sports"]
    assert len(sports) == 2
    assert all(o["truth"].source == "fake_sports" for o in sports)


def test_scanner_finds_weather_markets(scanner):
    opportunities = scanner.scan_all()
    weather = [o for o in opportunities if o["market"]["category"] == "weather"]
    assert len(weather) == 2
    assert all(o["truth"].source == "fake_weather" for o in weather)


def test_scanner_matches_sports_engine(scanner):
    sports_market = {"category": "sports", "question": "Will Lakers win?"}
    engine = scanner.match_market_to_engine(sports_market)
    assert isinstance(engine, FakeSportsEngine)


def test_scanner_matches_weather_engine(scanner):
    weather_market = {"category": "weather", "question": "Will temp exceed 80F?"}
    engine = scanner.match_market_to_engine(weather_market)
    assert isinstance(engine, FakeWeatherEngine)


def test_scanner_no_engine_for_politics(scanner):
    politics_market = {"category": "politics", "question": "Who wins 2028?"}
    engine = scanner.match_market_to_engine(politics_market)
    assert engine is None


def test_scanner_computes_edges(scanner):
    opportunities = scanner.scan_all()
    lakers = [o for o in opportunities if "lakers" in o["market"]["condition_id"]]
    assert len(lakers) == 1
    # truth says 0.45, market price is 0.35, edge = 0.10
    assert abs(lakers[0]["edge"] - 0.10) < 0.001

    nyc = [o for o in opportunities if "nyc" in o["market"]["condition_id"]]
    assert len(nyc) == 1
    # truth says 0.15, market price is 0.10, edge = 0.05
    assert abs(nyc[0]["edge"] - 0.05) < 0.001


def test_filter_by_liquidity(scanner):
    markets = scanner.client.get_markets()
    filtered = scanner.filter_by_liquidity(markets, min_volume=100000)
    assert all(m["volume"] >= 100000 for m in filtered)
    assert len(filtered) < len(markets)


def test_filter_by_liquidity_default(scanner):
    markets = scanner.client.get_markets()
    filtered = scanner.filter_by_liquidity(markets)
    assert all(m["volume"] >= 50000 for m in filtered)


def test_client_get_markets_by_category(client):
    sports = client.get_markets(category="sports")
    assert len(sports) == 2
    assert all(m["category"] == "sports" for m in sports)


def test_client_get_orderbook(client):
    book = client.get_orderbook("tok_lakers_yes")
    assert "bids" in book
    assert "asks" in book
    assert len(book["bids"]) > 0


def test_client_get_market_price(client):
    price = client.get_market_price("tok_lakers_yes")
    assert price == 0.35


def test_client_get_market_price_unknown(client):
    price = client.get_market_price("tok_nonexistent")
    assert price == 0.50

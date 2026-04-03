import pytest
from bot.market.polymarket import PolymarketClient
from bot.execution.order_manager import OrderManager
from bot.core.database import Database
from bot.core.risk import RiskManager


@pytest.fixture
def client():
    return PolymarketClient(mock_mode=True)


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def risk_manager():
    return RiskManager(
        daily_limit=1000,
        weekly_limit=5000,
        max_positions=10,
        max_drawdown_pct=0.20,
    )


@pytest.fixture
def order_manager(client, db, risk_manager):
    return OrderManager(client, db, risk_manager, mock_mode=True)


def test_place_order_fills_immediately(order_manager):
    result = order_manager.place_order("tok_lakers_yes", "buy", 100, 0.35)
    assert result["status"] == "filled"
    assert result["filled_size"] == 100
    assert result["id"] is not None


def test_place_order_logs_to_database(order_manager, db):
    order_manager.place_order("tok_lakers_yes", "buy", 100, 0.35)
    trades = db.get_trades()
    assert len(trades) == 1
    assert trades[0]["market_id"] == "tok_lakers_yes"
    assert trades[0]["side"] == "buy"
    assert trades[0]["size"] == 100


def test_place_order_respects_risk_limits(order_manager):
    # Fill up to max positions
    for i in range(10):
        order_manager.place_order(f"tok_{i}", "buy", 10, 0.50)
    # 11th should be rejected
    result = order_manager.place_order("tok_overflow", "buy", 10, 0.50)
    assert result["status"] == "rejected"
    assert result["reason"] == "risk_limit"
    assert result["filled_size"] == 0


def test_place_order_respects_daily_limit(client, db):
    risk = RiskManager(daily_limit=100, weekly_limit=5000, max_positions=100, max_drawdown_pct=0.50)
    om = OrderManager(client, db, risk, mock_mode=True)
    # daily_limit is checked against daily_losses + size, but placing orders
    # doesn't record losses directly. can_trade checks daily_losses + size > daily_limit.
    # With daily_losses=0 and size=101, 0+101 > 100 => rejected
    result = om.place_order("tok_x", "buy", 101, 0.50)
    assert result["status"] == "rejected"


def test_cancel_order_filled(order_manager):
    result = order_manager.place_order("tok_lakers_yes", "buy", 100, 0.35)
    cancel = order_manager.cancel_order(result["id"])
    assert cancel["status"] == "already_filled"


def test_cancel_order_not_found(order_manager):
    cancel = order_manager.cancel_order("nonexistent-id")
    assert cancel["status"] == "not_found"


def test_get_open_orders_empty(order_manager):
    # In mock mode all orders fill immediately, so no open orders
    order_manager.place_order("tok_lakers_yes", "buy", 100, 0.35)
    assert order_manager.get_open_orders() == []


def test_check_settlements_mock(order_manager):
    assert order_manager.check_settlements() == []

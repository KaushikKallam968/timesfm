import os
import tempfile
from bot.core.database import Database


def make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Database(path), path


def test_creates_tables():
    db, path = make_db()
    tables = db.list_tables()
    assert "trades" in tables
    assert "positions" in tables
    assert "daily_pnl" in tables
    os.unlink(path)


def test_log_trade_returns_id():
    db, path = make_db()
    trade_id = db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    assert trade_id == 1
    second_id = db.log_trade("market-2", "sell", 0.40, 5.0, 0.03, "model_v1", 0.37)
    assert second_id == 2
    os.unlink(path)


def test_get_trades_returns_dicts():
    db, path = make_db()
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    trades = db.get_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert isinstance(trade, dict)
    assert trade["market_id"] == "market-1"
    assert trade["side"] == "buy"
    assert trade["price"] == 0.65
    assert trade["size"] == 10.0
    assert trade["edge"] == 0.05
    assert trade["truth_source"] == "model_v1"
    assert trade["truth_probability"] == 0.70
    assert trade["outcome"] is None
    assert trade["payout"] is None
    os.unlink(path)


def test_get_trades_respects_limit():
    db, path = make_db()
    for i in range(5):
        db.log_trade(f"market-{i}", "buy", 0.5, 1.0, 0.01, "src", 0.5)
    assert len(db.get_trades(limit=3)) == 3
    assert len(db.get_trades(limit=10)) == 5
    os.unlink(path)


def test_settle_trade():
    db, path = make_db()
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    updated = db.settle_trade("market-1", "win", 15.0)
    assert updated == 1
    trade = db.get_trades()[0]
    assert trade["outcome"] == "win"
    assert trade["payout"] == 15.0
    os.unlink(path)


def test_settle_trade_only_unsettled():
    db, path = make_db()
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    db.settle_trade("market-1", "win", 15.0)
    updated = db.settle_trade("market-1", "win", 20.0)
    assert updated == 0
    os.unlink(path)


def test_get_daily_pnl_no_settled():
    db, path = make_db()
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    assert db.get_daily_pnl() == 0
    os.unlink(path)


def test_get_daily_pnl_with_settled():
    db, path = make_db()
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    db.log_trade("market-2", "sell", 0.40, 5.0, 0.03, "model_v1", 0.37)
    db.settle_trade("market-1", "win", 15.0)
    db.settle_trade("market-2", "loss", 0.0)
    pnl = db.get_daily_pnl()
    assert pnl == (15.0 - 10.0) + (0.0 - 5.0)
    os.unlink(path)


def test_get_open_positions_count():
    db, path = make_db()
    assert db.get_open_positions_count() == 0
    db.log_trade("market-1", "buy", 0.65, 10.0, 0.05, "model_v1", 0.70)
    db.log_trade("market-2", "sell", 0.40, 5.0, 0.03, "model_v1", 0.37)
    assert db.get_open_positions_count() == 2
    db.settle_trade("market-1", "win", 15.0)
    assert db.get_open_positions_count() == 1
    os.unlink(path)


def test_list_tables():
    db, path = make_db()
    tables = db.list_tables()
    assert sorted(tables) == ["daily_pnl", "positions", "trades"]
    os.unlink(path)

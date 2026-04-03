from unittest.mock import patch, MagicMock

from bot.run import TruthArbitrageEngine
from bot.truth.base import TruthResult


def make_engine():
    return TruthArbitrageEngine(mock_mode=True, db_path=":memory:")


class TestEngineInit:
    def test_initializes_without_errors(self):
        engine = make_engine()
        assert engine.mock_mode is True
        assert engine.client is not None
        assert engine.scanner is not None
        assert engine.db is not None
        assert engine.risk_manager is not None
        assert engine.order_manager is not None
        assert engine.discord is not None
        assert len(engine.truth_engines) == 3

    def test_default_bankroll(self):
        engine = make_engine()
        assert engine._bankroll == 1000.0


class TestScanAndTrade:
    def test_runs_without_crashing(self):
        engine = make_engine()
        trades = engine.scan_and_trade()
        assert isinstance(trades, list)

    def test_returns_empty_on_scanner_failure(self):
        engine = make_engine()
        engine.scanner.scan_all = MagicMock(side_effect=Exception("API down"))
        trades = engine.scan_and_trade()
        assert trades == []

    def test_opportunities_above_threshold_get_traded(self):
        engine = make_engine()

        fake_truth = TruthResult(probability=0.60, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test?"},
                "truth": fake_truth,
                "edge": 0.20,
                "token_id": "tok_test",
                "market_price": 0.40,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 1
        assert trades[0]["side"] == "buy"
        assert trades[0]["edge"] == 0.20
        assert trades[0]["size"] > 0

    def test_opportunities_below_threshold_skipped(self):
        engine = make_engine()

        fake_truth = TruthResult(probability=0.42, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test?"},
                "truth": fake_truth,
                "edge": 0.02,
                "token_id": "tok_test",
                "market_price": 0.40,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 0

    def test_negative_edge_sells(self):
        engine = make_engine()

        fake_truth = TruthResult(probability=0.30, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test?"},
                "truth": fake_truth,
                "edge": -0.20,
                "token_id": "tok_test",
                "market_price": 0.50,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 1
        assert trades[0]["side"] == "sell"

    def test_per_market_error_doesnt_crash_loop(self):
        engine = make_engine()

        good_truth = TruthResult(probability=0.60, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "bad_market", "question": "Bad?"},
                "truth": good_truth,
                "edge": 0.20,
                "token_id": "tok_bad",
                "market_price": 0.0,
            },
            {
                "market": {"condition_id": "good_market", "question": "Good?"},
                "truth": good_truth,
                "edge": 0.20,
                "token_id": "tok_good",
                "market_price": 0.40,
            },
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) >= 1


class TestRiskLimits:
    def test_position_limit_respected(self):
        engine = make_engine()
        engine.risk_manager.max_positions = 1

        fake_truth = TruthResult(probability=0.70, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": f"market_{i}", "question": f"Q{i}?"},
                "truth": fake_truth,
                "edge": 0.20,
                "token_id": f"tok_{i}",
                "market_price": 0.40,
            }
            for i in range(3)
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 1

    def test_daily_loss_limit_respected(self):
        engine = make_engine()
        engine.risk_manager.daily_losses = engine.risk_manager.daily_limit

        fake_truth = TruthResult(probability=0.70, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test?"},
                "truth": fake_truth,
                "edge": 0.20,
                "token_id": "tok_test",
                "market_price": 0.40,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 0


class TestDiscordAlerts:
    def test_discord_alert_sent_on_trade(self):
        engine = make_engine()
        engine.discord.send = MagicMock(return_value=True)

        fake_truth = TruthResult(probability=0.60, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test market?"},
                "truth": fake_truth,
                "edge": 0.20,
                "token_id": "tok_test",
                "market_price": 0.40,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 1
        engine.discord.send.assert_called_once()
        call_arg = engine.discord.send.call_args[0][0]
        assert "Trade Alert" in call_arg
        assert "Test market?" in call_arg

    def test_discord_failure_doesnt_crash(self):
        engine = make_engine()
        engine.discord.send = MagicMock(side_effect=Exception("webhook down"))

        fake_truth = TruthResult(probability=0.60, confidence=0.9, source="test_engine")
        fake_opportunities = [
            {
                "market": {"condition_id": "test_market", "question": "Test?"},
                "truth": fake_truth,
                "edge": 0.20,
                "token_id": "tok_test",
                "market_price": 0.40,
            }
        ]
        engine.scanner.scan_all = MagicMock(return_value=fake_opportunities)

        trades = engine.scan_and_trade()
        assert len(trades) == 1


class TestDailyReport:
    def test_daily_report_runs(self):
        engine = make_engine()
        engine.discord.send = MagicMock(return_value=True)
        engine.daily_report()
        engine.discord.send.assert_called_once()

    def test_daily_report_error_doesnt_crash(self):
        engine = make_engine()
        engine.db.get_daily_pnl = MagicMock(side_effect=Exception("db error"))
        engine.daily_report()


class TestStartStop:
    def test_stop_without_start(self):
        engine = make_engine()
        engine.stop()

    def test_scheduler_configured(self):
        engine = make_engine()
        engine.scheduler = MagicMock()
        engine.scheduler.running = True
        engine.stop()
        engine.scheduler.shutdown.assert_called_once_with(wait=False)

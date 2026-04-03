from bot.core.risk import RiskManager


def test_can_trade_within_limits():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    assert rm.can_trade(100) is True


def test_daily_limit_blocks_trade():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.record_loss(450)
    assert rm.can_trade(100) is False


def test_weekly_limit_blocks_trade():
    rm = RiskManager(daily_limit=500, weekly_limit=200, max_positions=5, max_drawdown_pct=0.2)
    rm.record_loss(150)
    assert rm.can_trade(100) is False


def test_max_positions_blocks_trade():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=2, max_drawdown_pct=0.2)
    rm.add_position()
    rm.add_position()
    assert rm.can_trade(10) is False


def test_record_loss_accumulates():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.record_loss(100)
    rm.record_loss(200)
    assert rm.daily_losses == 300
    assert rm.weekly_losses == 300


def test_record_win_accumulates():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.record_win(100)
    rm.record_win(200)
    assert rm.daily_wins == 300
    assert rm.weekly_wins == 300


def test_open_positions_tracking():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    assert rm.get_open_positions() == 0
    rm.add_position()
    rm.add_position()
    assert rm.get_open_positions() == 2
    rm.close_position()
    assert rm.get_open_positions() == 1


def test_close_position_no_negative():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.close_position()
    assert rm.get_open_positions() == 0


def test_reset_daily():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.record_loss(200)
    rm.record_win(100)
    rm.reset_daily()
    assert rm.daily_losses == 0
    assert rm.daily_wins == 0
    assert rm.weekly_losses == 200
    assert rm.weekly_wins == 100


def test_drawdown_blocks_trade():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.current_bankroll = 1000
    rm.peak_bankroll = 1000
    rm.record_loss(200)  # bankroll now 800, drawdown = 20%
    assert rm.can_trade(10) is False


def test_drawdown_allows_trade_within_limit():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.current_bankroll = 1000
    rm.peak_bankroll = 1000
    rm.record_loss(100)  # bankroll now 900, drawdown = 10%
    assert rm.can_trade(10) is True


def test_win_updates_peak():
    rm = RiskManager(daily_limit=500, weekly_limit=2000, max_positions=5, max_drawdown_pct=0.2)
    rm.current_bankroll = 1000
    rm.peak_bankroll = 1000
    rm.record_win(200)
    assert rm.peak_bankroll == 1200
    assert rm.current_bankroll == 1200

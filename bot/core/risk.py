class RiskManager:
    def __init__(self, daily_limit, weekly_limit, max_positions, max_drawdown_pct):
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct

        self.daily_losses = 0
        self.weekly_losses = 0
        self.daily_wins = 0
        self.weekly_wins = 0
        self.open_positions = 0
        self.peak_bankroll = 0
        self.current_bankroll = 0

    def can_trade(self, size):
        if self.daily_losses + size > self.daily_limit:
            return False
        if self.weekly_losses + size > self.weekly_limit:
            return False
        if self.open_positions >= self.max_positions:
            return False
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            if drawdown >= self.max_drawdown_pct:
                return False
        return True

    def record_loss(self, amount):
        self.daily_losses += amount
        self.weekly_losses += amount
        self.current_bankroll -= amount

    def record_win(self, amount):
        self.daily_wins += amount
        self.weekly_wins += amount
        self.current_bankroll += amount
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

    def get_open_positions(self):
        return self.open_positions

    def add_position(self):
        self.open_positions += 1

    def close_position(self):
        if self.open_positions > 0:
            self.open_positions -= 1

    def reset_daily(self):
        self.daily_losses = 0
        self.daily_wins = 0

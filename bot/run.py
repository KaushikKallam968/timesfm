import logging
import signal
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from bot.core.config import (
    DAILY_LOSS_LIMIT,
    DISCORD_WEBHOOK_URL,
    KELLY_FRACTION,
    MAX_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    MAX_TRADE_SIZE,
    ODDS_API_KEY,
    POLYMARKET_API_KEY,
    POLYMARKET_PRIVATE_KEY,
    SPORTS_EDGE_THRESHOLD,
    WEEKLY_LOSS_LIMIT,
)
from bot.core.database import Database
from bot.core.risk import RiskManager
from bot.execution.edge_detector import rank_opportunities
from bot.execution.kelly import kelly_size
from bot.execution.order_manager import OrderManager
from bot.market.polymarket import PolymarketClient
from bot.market.scanner import MarketScanner
from bot.monitoring.discord import DiscordNotifier
from bot.truth.correlation import CorrelationEngine
from bot.truth.sports import SportsOddsEngine
from bot.truth.weather import WeatherEngine

logger = logging.getLogger(__name__)


class TruthArbitrageEngine:
    def __init__(self, mock_mode=True, db_path=":memory:"):
        self.mock_mode = mock_mode

        self.client = PolymarketClient(
            api_key=POLYMARKET_API_KEY,
            private_key=POLYMARKET_PRIVATE_KEY,
            mock_mode=mock_mode,
        )

        self.truth_engines = [
            SportsOddsEngine(api_key=ODDS_API_KEY),
            WeatherEngine(),
            CorrelationEngine(),
        ]

        self.scanner = MarketScanner(self.client, self.truth_engines)

        self.db = Database(db_path)
        self.risk_manager = RiskManager(
            daily_limit=DAILY_LOSS_LIMIT,
            weekly_limit=WEEKLY_LOSS_LIMIT,
            max_positions=MAX_OPEN_POSITIONS,
            max_drawdown_pct=MAX_DRAWDOWN_PCT,
        )

        self.order_manager = OrderManager(
            client=self.client,
            db=self.db,
            risk_manager=self.risk_manager,
            mock_mode=mock_mode,
        )

        self.discord = DiscordNotifier(webhook_url=DISCORD_WEBHOOK_URL)
        self.scheduler = None
        self._bankroll = 1000.0

    def scan_and_trade(self):
        try:
            opportunities = self.scanner.scan_all()
        except Exception as e:
            logger.error("Failed to scan markets: %s", e)
            return []

        above_threshold = [
            opp for opp in opportunities
            if abs(opp["edge"]) >= SPORTS_EDGE_THRESHOLD
        ]

        ranked = rank_opportunities(above_threshold, MAX_OPEN_POSITIONS)

        trades = []
        for opp in ranked:
            try:
                edge = abs(opp["edge"])
                market_price = opp["market_price"]
                if market_price <= 0:
                    continue
                odds = (1.0 / market_price) - 1.0
                true_prob = opp["truth"].probability if opp["edge"] > 0 else 1.0 - opp["truth"].probability

                size = kelly_size(
                    edge=true_prob,
                    odds=odds,
                    bankroll=self._bankroll,
                    fraction=KELLY_FRACTION,
                    max_size=MAX_TRADE_SIZE,
                )

                if size <= 0:
                    continue

                side = "buy" if opp["edge"] > 0 else "sell"
                result = self.order_manager.place_order(
                    token_id=opp["token_id"],
                    side=side,
                    size=size,
                    price=market_price,
                )

                if result["status"] == "rejected":
                    logger.info("Order rejected: %s", result.get("reason"))
                    continue

                trades.append({
                    "market": opp["market"],
                    "side": side,
                    "size": size,
                    "price": market_price,
                    "edge": opp["edge"],
                    "truth_source": opp["truth"].source,
                    "order": result,
                })

                try:
                    alert = self.discord.format_trade_alert(
                        market=opp["market"].get("question", "Unknown"),
                        side=side,
                        price=market_price,
                        edge=opp["edge"],
                        size=size,
                        truth_source=opp["truth"].source,
                    )
                    self.discord.send(alert)
                except Exception as e:
                    logger.error("Failed to send Discord alert: %s", e)

            except Exception as e:
                market_id = opp.get("market", {}).get("condition_id", "unknown")
                logger.error("Error processing market %s: %s", market_id, e)
                continue

        return trades

    def daily_report(self):
        try:
            pnl = self.db.get_daily_pnl()
            all_trades = self.db.get_trades()
            open_count = self.db.get_open_positions_count()

            settled = [t for t in all_trades if t.get("outcome") is not None]
            wins = sum(1 for t in settled if (t.get("payout", 0) or 0) > t["size"])
            losses = len(settled) - wins
            win_rate = wins / len(settled) if settled else 0

            best = max((t.get("payout", 0) or 0) - t["size"] for t in settled) if settled else 0
            worst = min((t.get("payout", 0) or 0) - t["size"] for t in settled) if settled else 0

            report = self.discord.format_daily_report(
                pnl=pnl,
                trades=len(all_trades),
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                bankroll=self._bankroll,
                best_trade=best,
                worst_trade=worst,
            )
            self.discord.send(report)
        except Exception as e:
            logger.error("Failed to generate daily report: %s", e)

    def start(self):
        logger.info("Starting TruthArbitrageEngine (mock_mode=%s)", self.mock_mode)

        self.scheduler = BlockingScheduler()
        self.scheduler.add_job(
            self.scan_and_trade,
            "interval",
            minutes=5,
            id="scan_and_trade",
            next_run_time=datetime.now(timezone.utc),
        )
        self.scheduler.add_job(
            self.daily_report,
            "interval",
            hours=24,
            id="daily_report",
        )

        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        logger.info("Shutting down TruthArbitrageEngine")
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TruthArbitrageEngine(mock_mode=True)
    engine.start()

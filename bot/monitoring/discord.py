import logging
import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def format_trade_alert(self, market, side, price, edge, size, truth_source):
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        return (
            f"{emoji} **Trade Alert: {side.upper()}**\n"
            f"**Market:** {market}\n"
            f"**Side:** {side}\n"
            f"**Price:** {price}\n"
            f"**Edge:** {edge:.1%}\n"
            f"**Size:** ${size:.2f}\n"
            f"**Truth Source:** {truth_source}"
        )

    def format_daily_report(self, pnl, trades, wins, losses, win_rate, bankroll, best_trade, worst_trade):
        emoji = "🟢" if pnl >= 0 else "🔴"
        return (
            f"{emoji} **Daily Report**\n"
            f"**P&L:** ${pnl:+.2f}\n"
            f"**Trades:** {trades}\n"
            f"**Wins:** {wins} | **Losses:** {losses}\n"
            f"**Win Rate:** {win_rate:.1%}\n"
            f"**Bankroll:** ${bankroll:.2f}\n"
            f"**Best Trade:** ${best_trade:+.2f}\n"
            f"**Worst Trade:** ${worst_trade:+.2f}"
        )

    def format_error_alert(self, error_type, message):
        return (
            f"⚠️ **Error: {error_type}**\n"
            f"{message}"
        )

    def format_risk_alert(self, alert_type, details):
        return (
            f"🚨 **Risk Alert: {alert_type}**\n"
            f"{details}"
        )

    def send(self, message):
        try:
            response = requests.post(
                self.webhook_url,
                json={"content": message},
                timeout=10,
            )
            if response.status_code == 204:
                return True
            logger.warning("Discord webhook returned status %s", response.status_code)
            return False
        except Exception as e:
            logger.error("Failed to send Discord message: %s", e)
            return False

    def send_embed(self, title, description, color, fields):
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
        }
        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10,
            )
            if response.status_code == 204:
                return True
            logger.warning("Discord webhook returned status %s", response.status_code)
            return False
        except Exception as e:
            logger.error("Failed to send Discord embed: %s", e)
            return False

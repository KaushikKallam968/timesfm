from unittest.mock import patch, MagicMock
import requests
from bot.monitoring.discord import DiscordNotifier


WEBHOOK_URL = "https://discord.com/api/webhooks/test/fake"


def make_notifier():
    return DiscordNotifier(WEBHOOK_URL)


def test_format_trade_alert_contains_market_and_edge():
    n = make_notifier()
    msg = n.format_trade_alert("Will X win?", "buy", 0.65, 0.12, 50.0, "polls")
    assert "Will X win?" in msg
    assert "12.0%" in msg


def test_format_trade_alert_buy_emoji():
    n = make_notifier()
    msg = n.format_trade_alert("M", "buy", 0.5, 0.1, 10.0, "src")
    assert "🟢" in msg


def test_format_trade_alert_sell_emoji():
    n = make_notifier()
    msg = n.format_trade_alert("M", "sell", 0.5, 0.1, 10.0, "src")
    assert "🔴" in msg


def test_format_daily_report_positive_pnl():
    n = make_notifier()
    msg = n.format_daily_report(150.0, 10, 7, 3, 0.7, 1150.0, 80.0, -30.0)
    assert "🟢" in msg
    assert "+150.00" in msg


def test_format_daily_report_negative_pnl():
    n = make_notifier()
    msg = n.format_daily_report(-50.0, 5, 2, 3, 0.4, 950.0, 20.0, -40.0)
    assert "🔴" in msg
    assert "-50.00" in msg


def test_format_error_alert_contains_message():
    n = make_notifier()
    msg = n.format_error_alert("APIError", "Rate limit exceeded")
    assert "Rate limit exceeded" in msg
    assert "⚠️" in msg


def test_format_risk_alert_contains_alert_type():
    n = make_notifier()
    msg = n.format_risk_alert("Daily loss limit reached", "Lost $500 today")
    assert "Daily loss limit reached" in msg
    assert "🚨" in msg


@patch("bot.monitoring.discord.requests.post")
def test_send_returns_true_on_204(mock_post):
    mock_post.return_value = MagicMock(status_code=204)
    n = make_notifier()
    assert n.send("hello") is True
    mock_post.assert_called_once_with(
        WEBHOOK_URL,
        json={"content": "hello"},
        timeout=10,
    )


@patch("bot.monitoring.discord.requests.post")
def test_send_returns_false_on_network_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("fail")
    n = make_notifier()
    assert n.send("hello") is False


@patch("bot.monitoring.discord.requests.post")
def test_send_returns_false_on_non_204(mock_post):
    mock_post.return_value = MagicMock(status_code=400)
    n = make_notifier()
    assert n.send("hello") is False


@patch("bot.monitoring.discord.requests.post")
def test_send_embed_formats_correctly(mock_post):
    mock_post.return_value = MagicMock(status_code=204)
    n = make_notifier()
    fields = [{"name": "PnL", "value": "+$100", "inline": True}]
    result = n.send_embed("Daily Report", "Summary", 0x00FF00, fields)
    assert result is True
    call_kwargs = mock_post.call_args[1]
    embed = call_kwargs["json"]["embeds"][0]
    assert embed["title"] == "Daily Report"
    assert embed["description"] == "Summary"
    assert embed["color"] == 0x00FF00
    assert embed["fields"] == fields

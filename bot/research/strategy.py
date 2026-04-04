"""The ONE file the autoresearch agent modifies. Contains every tunable parameter and strategy decision."""

# ── Market Making Parameters ──────────────────────────────────────────
MM_BASE_SPREAD = 0.04          # 4 cents base spread
MM_MIN_SPREAD = 0.02
MM_INVENTORY_SKEW = 0.5
MM_MAX_INVENTORY_PCT = 0.20
MM_PANIC_INVENTORY_PCT = 0.30

# ── Directional Parameters ────────────────────────────────────────────
DIR_MIN_EDGE = 0.4            # 15% minimum edge for directional
DIR_MIN_CONFIDENCE = 0.90
DIR_MAX_DAYS_TO_RES = 3
DIR_TRUTH_ACCURACY = 0.85

# ── Sizing & Risk ─────────────────────────────────────────────────────
KELLY_FRACTION = 0.25
MAX_TRADE_SIZE = 100.0
MAX_PCT_PER_TRADE = 0.05
FEE_RATE = 0.02
SLIPPAGE = 0.005

# ── Filtering ─────────────────────────────────────────────────────────
MIN_VOLUME = 1000
EXCLUDED_CATEGORIES = []
MAX_POSITIONS = 20
CATEGORY_EDGE_OVERRIDES = {}   # e.g. {"politics": 0.12}

# ── Bankroll ──────────────────────────────────────────────────────────
STARTING_BANKROLL = 10000.0


# ── Functions ─────────────────────────────────────────────────────────

def should_trade_market(record):
    """Decide whether a market is eligible for trading.

    Args:
        record: dict with keys 'price', 'volume', 'category'.

    Returns:
        bool — True if market passes all filters.
    """
    price = record.get("price", 0)
    if price < 0.03 or price > 0.97:
        return False

    volume = record.get("volume", 0)
    if volume < MIN_VOLUME:
        return False

    category = record.get("category", "")
    if category in EXCLUDED_CATEGORIES:
        return False

    return True


def compute_edge(record, truth_prob):
    """Compute directional edge for a market.

    Args:
        record: dict with keys 'price', 'category', 'days_to_resolution'.
        truth_prob: float in [0, 1] — our probability estimate for YES.

    Returns:
        (side, edge) tuple.
        side is "YES", "NO", or None.
        edge is the absolute edge (float >= 0). Returns (None, 0) when no trade.
    """
    days = record.get("days_to_resolution", float("inf"))
    if days > DIR_MAX_DAYS_TO_RES:
        return (None, 0)

    market_price = record.get("price", 0.5)
    category = record.get("category", "")

    min_edge = CATEGORY_EDGE_OVERRIDES.get(category, DIR_MIN_EDGE)

    yes_edge = truth_prob - market_price
    no_edge = (1 - truth_prob) - (1 - market_price)  # equivalent to market_price - truth_prob

    if yes_edge >= min_edge:
        return ("YES", yes_edge)
    elif no_edge >= min_edge:
        return ("NO", no_edge)

    return (None, 0)


def compute_mm_quotes(mid_price, net_inventory, bankroll):
    """Compute market-making bid and ask quotes with inventory skew.

    Args:
        mid_price: float in (0, 1) — current mid price of the market.
        net_inventory: float — signed inventory (positive = long YES).
        bankroll: float — current bankroll.

    Returns:
        (bid, ask) tuple of floats, each in [0, 1].
    """
    half_spread = max(MM_BASE_SPREAD / 2, MM_MIN_SPREAD / 2)

    # Skew quotes away from inventory to reduce risk
    inventory_ratio = net_inventory / bankroll if bankroll > 0 else 0
    skew = MM_INVENTORY_SKEW * inventory_ratio

    bid = mid_price - half_spread - skew
    ask = mid_price + half_spread - skew

    # Clamp to valid price range
    bid = max(0.0, min(bid, 1.0))
    ask = max(0.0, min(ask, 1.0))

    return (bid, ask)


def kelly_size(edge, entry_price, bankroll):
    """Compute position size using fractional Kelly criterion for binary markets.

    Args:
        edge: float — our edge (probability advantage) on this bet.
        entry_price: float in (0, 1) — the price we'd pay (YES side).
        bankroll: float — current bankroll.

    Returns:
        float — dollar amount to wager, capped by MAX_TRADE_SIZE and MAX_PCT_PER_TRADE.
    """
    if edge <= 0 or entry_price <= 0 or entry_price >= 1 or bankroll <= 0:
        return 0.0

    # Kelly for binary: f* = (p * b - q) / b
    # where p = implied true prob, b = payout ratio, q = 1 - p
    win_prob = entry_price + edge
    win_prob = min(win_prob, 0.999)  # guard against >1
    lose_prob = 1 - win_prob

    payout_ratio = (1 - entry_price) / entry_price  # net profit per dollar risked
    if payout_ratio <= 0:
        return 0.0

    kelly_full = (win_prob * payout_ratio - lose_prob) / payout_ratio
    if kelly_full <= 0:
        return 0.0

    kelly_bet = KELLY_FRACTION * kelly_full * bankroll

    # Apply caps
    max_from_pct = MAX_PCT_PER_TRADE * bankroll
    size = min(kelly_bet, MAX_TRADE_SIZE, max_from_pct)

    return max(size, 0.0)

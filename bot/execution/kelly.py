def kelly_size(edge, odds, bankroll, fraction=0.15, max_size=100):
    if edge <= 0:
        return 0

    kelly = (edge * odds - (1 - edge)) / odds
    size = kelly * fraction * bankroll

    return min(size, max_size)

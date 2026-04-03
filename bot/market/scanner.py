from bot.market.polymarket import PolymarketClient


class MarketScanner:
    def __init__(self, client, truth_engines):
        self.client = client
        self.truth_engines = truth_engines

    def match_market_to_engine(self, market):
        for engine in self.truth_engines:
            if engine.can_handle(market):
                return engine
        return None

    def filter_by_liquidity(self, markets, min_volume=50000):
        return [m for m in markets if m.get("volume", 0) >= min_volume]

    def scan_all(self):
        markets = self.client.get_markets(status="active")
        opportunities = []
        for market in markets:
            engine = self.match_market_to_engine(market)
            if not engine:
                continue
            truth = engine.get_truth(market)
            if not truth:
                continue
            yes_outcome = market["outcomes"][0]
            market_price = yes_outcome["price"]
            edge = truth.edge(market_price)
            opportunities.append({
                "market": market,
                "truth": truth,
                "edge": edge,
                "token_id": yes_outcome["token_id"],
                "market_price": market_price,
            })
        return opportunities

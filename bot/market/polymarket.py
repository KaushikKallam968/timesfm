MOCK_MARKETS = [
    {
        "condition_id": "sports_nba_lakers_2026",
        "question": "Will the Lakers win the NBA Championship 2026?",
        "category": "sports",
        "outcomes": [
            {"name": "Yes", "price": 0.35, "token_id": "tok_lakers_yes"},
            {"name": "No", "price": 0.65, "token_id": "tok_lakers_no"},
        ],
        "volume": 120000,
        "end_date": "2026-06-30",
        "team": "Los Angeles Lakers",
    },
    {
        "condition_id": "sports_nfl_chiefs_2026",
        "question": "Will the Chiefs win the NFL Super Bowl 2027?",
        "category": "sports",
        "outcomes": [
            {"name": "Yes", "price": 0.20, "token_id": "tok_chiefs_yes"},
            {"name": "No", "price": 0.80, "token_id": "tok_chiefs_no"},
        ],
        "volume": 250000,
        "end_date": "2027-02-15",
        "team": "Kansas City Chiefs",
    },
    {
        "condition_id": "weather_nyc_temp_apr",
        "question": "Will NYC temperature exceed 80°F on April 15?",
        "category": "weather",
        "outcomes": [
            {"name": "Yes", "price": 0.10, "token_id": "tok_nyc_temp_yes"},
            {"name": "No", "price": 0.90, "token_id": "tok_nyc_temp_no"},
        ],
        "volume": 75000,
        "end_date": "2026-04-15",
    },
    {
        "condition_id": "weather_miami_temp_jul",
        "question": "Will Miami temperature exceed 95°F on July 4?",
        "category": "weather",
        "outcomes": [
            {"name": "Yes", "price": 0.45, "token_id": "tok_miami_temp_yes"},
            {"name": "No", "price": 0.55, "token_id": "tok_miami_temp_no"},
        ],
        "volume": 60000,
        "end_date": "2026-07-04",
    },
    {
        "condition_id": "politics_us_pres_2028",
        "question": "Will the Democratic candidate win the 2028 US Presidential Election?",
        "category": "politics",
        "outcomes": [
            {"name": "Yes", "price": 0.52, "token_id": "tok_dem_pres_yes"},
            {"name": "No", "price": 0.48, "token_id": "tok_dem_pres_no"},
        ],
        "volume": 500000,
        "end_date": "2028-11-05",
    },
]

MOCK_ORDERBOOK = {
    "bids": [
        {"price": 0.34, "size": 500},
        {"price": 0.33, "size": 1000},
        {"price": 0.32, "size": 2000},
    ],
    "asks": [
        {"price": 0.36, "size": 500},
        {"price": 0.37, "size": 1000},
        {"price": 0.38, "size": 2000},
    ],
}


class PolymarketClient:
    def __init__(self, api_key=None, private_key=None, mock_mode=True):
        self.mock_mode = mock_mode or not api_key or not private_key
        self.api_key = api_key
        self.private_key = private_key

    def get_markets(self, category=None, status="active"):
        if self.mock_mode:
            markets = MOCK_MARKETS
            if category:
                markets = [m for m in markets if m["category"] == category]
            return markets
        raise NotImplementedError("Live mode requires py-clob-client")

    def get_orderbook(self, token_id):
        if self.mock_mode:
            return MOCK_ORDERBOOK
        raise NotImplementedError("Live mode requires py-clob-client")

    def get_market_price(self, token_id):
        if self.mock_mode:
            for market in MOCK_MARKETS:
                for outcome in market["outcomes"]:
                    if outcome["token_id"] == token_id:
                        return outcome["price"]
            return 0.50
        raise NotImplementedError("Live mode requires py-clob-client")

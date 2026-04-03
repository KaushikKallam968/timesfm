import urllib.request
import json
from bot.truth.base import TruthEngine, TruthResult


SPORT_KEYWORDS = ["nba", "nfl", "mlb", "nhl", "soccer", "mma", "ufc", "boxing", "tennis"]

SPORT_KEY_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "mma": "mma_mixed_martial_arts",
    "soccer": "soccer_epl",
}


class SportsOddsEngine(TruthEngine):
    def __init__(self, api_key):
        self.api_key = api_key
        self._mock_data = None

    def can_handle(self, market):
        if market.get("category") == "sports":
            return True
        question = market.get("question", "").lower()
        return any(kw in question for kw in SPORT_KEYWORDS)

    def _american_to_prob(self, odds):
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        return 100 / (odds + 100)

    def _consensus_probability(self, odds_data, team_name):
        probs = []
        weights = []
        for bookmaker in odds_data:
            book_name = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"].lower() == team_name.lower():
                        prob = self._american_to_prob(outcome["price"])
                        probs.append(prob)
                        weights.append(2 if book_name == "pinnacle" else 1)
        if not probs:
            return None
        total_weight = sum(weights)
        return sum(p * w for p, w in zip(probs, weights)) / total_weight

    def _compute_confidence(self, odds_data, team_name):
        probs = []
        for bookmaker in odds_data:
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"].lower() == team_name.lower():
                        probs.append(self._american_to_prob(outcome["price"]))
        if len(probs) >= 3:
            spread = max(probs) - min(probs)
            if spread <= 0.05:
                return 0.9
        return 0.7

    def fetch_odds(self, sport_key):
        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
            f"?apiKey={self.api_key}&regions=us&markets=h2h"
        )
        try:
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read())
        except Exception:
            return []

    def get_truth(self, market):
        team_name = market.get("team")
        if not team_name:
            return None

        if self._mock_data is not None:
            odds_data = self._mock_data
        else:
            sport_key = self._resolve_sport_key(market)
            if not sport_key:
                return None
            games = self.fetch_odds(sport_key)
            odds_data = self._find_game(games, team_name)
            if not odds_data:
                return None

        prob = self._consensus_probability(odds_data, team_name)
        if prob is None:
            return None
        confidence = self._compute_confidence(odds_data, team_name)
        return TruthResult(
            probability=round(prob, 4),
            confidence=confidence,
            source="sports_odds_api",
        )

    def _resolve_sport_key(self, market):
        question = market.get("question", "").lower()
        for keyword, key in SPORT_KEY_MAP.items():
            if keyword in question:
                return key
        return market.get("sport_key")

    def _find_game(self, games, team_name):
        for game in games:
            teams = [
                game.get("home_team", "").lower(),
                game.get("away_team", "").lower(),
            ]
            if team_name.lower() in teams:
                return game.get("bookmakers", [])
        return None

import re
import urllib.request
import json
from bot.truth.base import TruthEngine, TruthResult


CITY_COORDINATES = {
    "nyc": {"name": "New York", "lat": 40.71, "lon": -74.01},
    "new york": {"name": "New York", "lat": 40.71, "lon": -74.01},
    "chicago": {"name": "Chicago", "lat": 41.88, "lon": -87.63},
    "miami": {"name": "Miami", "lat": 25.76, "lon": -80.19},
    "la": {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
    "los angeles": {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
    "denver": {"name": "Denver", "lat": 39.74, "lon": -104.99},
    "london": {"name": "London", "lat": 51.51, "lon": -0.13},
    "seoul": {"name": "Seoul", "lat": 37.57, "lon": 126.98},
}

WEATHER_KEYWORDS = ["temperature", "temp", "°f", "°c", "high temp", "weather", "forecast"]


class WeatherEngine(TruthEngine):
    def __init__(self):
        self._mock_ensemble = None

    def can_handle(self, market):
        if market.get("category", "").lower() == "weather":
            return True
        question = market.get("question", "").lower()
        return any(kw in question for kw in WEATHER_KEYWORDS)

    def _parse_weather_market(self, market):
        question = market.get("question", "")
        if not question:
            return None

        q_lower = question.lower()

        comparison = "above"
        if "below" in q_lower or "under" in q_lower or "drop below" in q_lower or "fall below" in q_lower:
            comparison = "below"

        temp_match = re.search(r'(-?\d+\.?\d*)\s*°?[fFcC]', question)
        if not temp_match:
            return None
        threshold = float(temp_match.group(1))

        city_info = None
        for key, info in CITY_COORDINATES.items():
            if key in q_lower:
                city_info = info
                break

        if not city_info:
            return None

        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        if not date_match:
            month_day = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})', q_lower)
            if month_day:
                months = {
                    "january": "01", "february": "02", "march": "03", "april": "04",
                    "may": "05", "june": "06", "july": "07", "august": "08",
                    "september": "09", "october": "10", "november": "11", "december": "12"
                }
                month_num = months[month_day.group(1)]
                day = month_day.group(2).zfill(2)
                date_str = f"2026-{month_num}-{day}"
            else:
                short_month = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})', q_lower)
                if short_month:
                    months_short = {
                        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
                        "may": "05", "jun": "06", "jul": "07", "aug": "08",
                        "sep": "09", "oct": "10", "nov": "11", "dec": "12"
                    }
                    month_num = months_short[short_month.group(1)[:3]]
                    day = short_month.group(2).zfill(2)
                    date_str = f"2026-{month_num}-{day}"
                else:
                    return None
        else:
            date_str = date_match.group(1)

        return {
            "city": city_info["name"],
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "threshold": threshold,
            "comparison": comparison,
            "date": date_str,
        }

    def _ensemble_probability(self, members, threshold, comparison):
        if not members:
            return 0.5
        if comparison == "above":
            count = sum(1 for m in members if m > threshold)
        else:
            count = sum(1 for m in members if m < threshold)
        return count / len(members)

    def _confidence_from_ensemble(self, members, threshold):
        if not members:
            return 0.5
        above = sum(1 for m in members if m > threshold)
        agreement = max(above, len(members) - above) / len(members)
        if agreement > 0.9:
            return 0.95
        elif agreement >= 0.7:
            return 0.85
        elif agreement >= 0.5:
            return 0.70
        else:
            return 0.50

    def fetch_ensemble(self, lat, lon, date):
        if self._mock_ensemble is not None:
            return self._mock_ensemble

        try:
            url = (
                f"https://ensemble-api.open-meteo.com/v1/ensemble?"
                f"latitude={lat}&longitude={lon}"
                f"&models=gfs_seamless"
                f"&daily=temperature_2m_max"
                f"&forecast_days=7"
                f"&temperature_unit=fahrenheit"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            dates = data.get("daily", {}).get("time", [])
            if date not in dates:
                return None

            date_idx = dates.index(date)
            members = []
            daily = data.get("daily", {})
            for key, values in daily.items():
                if key.startswith("temperature_2m_max"):
                    if isinstance(values, list) and date_idx < len(values):
                        val = values[date_idx]
                        if val is not None:
                            members.append(val)

            return members if members else None
        except Exception:
            return None

    def get_truth(self, market):
        parsed = self._parse_weather_market(market)
        if not parsed:
            return None

        members = self.fetch_ensemble(parsed["lat"], parsed["lon"], parsed["date"])
        if not members:
            return None

        probability = self._ensemble_probability(members, parsed["threshold"], parsed["comparison"])
        confidence = self._confidence_from_ensemble(members, parsed["threshold"])

        return TruthResult(
            probability=probability,
            confidence=confidence,
            source="weather_ensemble_gfs",
        )

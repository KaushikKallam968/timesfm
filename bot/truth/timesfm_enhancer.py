import numpy as np


class TimesFMEnhancer:
    def __init__(self, model=None, mock_mode=True):
        self.model = model
        self.mock_mode = mock_mode or model is None

    def predict_odds_trajectory(self, odds_history, horizon=24):
        if not odds_history or len(odds_history) < 2:
            return {
                "predicted_odds": [],
                "direction": "stable",
                "magnitude": 0.0,
                "confidence": 0.0,
                "spread_widening": False,
            }

        if self.mock_mode:
            return self._mock_predict(odds_history, horizon)
        return self._live_predict(odds_history, horizon)

    def should_enter_now(self, odds_history, current_edge):
        if abs(current_edge) < 0.02:
            return {
                "action": "skip",
                "reason": "Edge too small to be actionable",
                "expected_better_entry": None,
            }

        trajectory = self.predict_odds_trajectory(odds_history)
        if not trajectory["predicted_odds"]:
            return {
                "action": "enter_now",
                "reason": "Insufficient history to predict movement",
                "expected_better_entry": None,
            }

        direction = trajectory["direction"]
        edge_is_positive = current_edge > 0

        # If we think the outcome is more likely than the market (positive edge),
        # odds moving UP means the market is correcting toward our view (edge closing).
        # If odds are moving DOWN, the market is moving away (edge widening).
        # Vice versa for negative edge.
        if edge_is_positive:
            edge_closing = direction == "up"
        else:
            edge_closing = direction == "down"

        if direction == "stable":
            return {
                "action": "enter_now",
                "reason": "Odds stable, edge unlikely to improve",
                "expected_better_entry": None,
            }

        if edge_closing:
            return {
                "action": "enter_now",
                "reason": "Edge is closing, enter before price moves further",
                "expected_better_entry": None,
            }

        # Edge is widening — wait for better price
        predicted = trajectory["predicted_odds"]
        current = odds_history[-1]
        best_entry = min(predicted) if edge_is_positive else max(predicted)
        return {
            "action": "wait",
            "reason": "Edge is widening, better entry expected",
            "expected_better_entry": best_entry if best_entry != current else None,
        }

    def rank_markets_by_timing(self, markets_with_history):
        scored = []
        for market in markets_with_history:
            trajectory = self.predict_odds_trajectory(market["odds_history"])
            entry = self.should_enter_now(market["odds_history"], market["current_edge"])

            timing_score = 0.0
            if entry["action"] == "enter_now":
                timing_score = 1.0 + trajectory["confidence"] + trajectory["magnitude"]
            elif entry["action"] == "wait":
                timing_score = 0.5 * trajectory["confidence"]
            # skip gets 0

            scored.append({**market, "timing_score": timing_score})

        scored.sort(key=lambda m: m["timing_score"], reverse=True)
        return scored

    def _mock_predict(self, odds_history, horizon):
        arr = np.array(odds_history, dtype=float)
        window = arr[-min(10, len(arr)):]

        # Direction via linear regression slope
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]

        # Magnitude from std of recent changes
        diffs = np.diff(window)
        magnitude = float(np.std(diffs)) if len(diffs) > 0 else 0.0

        # Confidence from consistency of trend
        if magnitude > 0:
            confidence = min(1.0, abs(slope) / (magnitude + 1e-9))
        else:
            confidence = 1.0 if abs(slope) < 1e-9 else 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        # Simple linear extrapolation for predicted odds
        last = float(arr[-1])
        predicted = [float(np.clip(last + slope * (i + 1), 0.0, 1.0)) for i in range(horizon)]

        # Spread widening: mock as True when magnitude is above a threshold
        spread_widening = magnitude > 0.02

        return {
            "predicted_odds": predicted,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence,
            "spread_widening": spread_widening,
        }

    def _live_predict(self, odds_history, horizon):
        arr = np.array(odds_history, dtype=float)
        point, quantiles = self.model.forecast(horizon=horizon, inputs=[arr])
        predicted = [float(np.clip(v, 0.0, 1.0)) for v in point[0]]

        # Direction from predicted trend
        pred_arr = np.array(predicted)
        x = np.arange(len(pred_arr))
        slope = np.polyfit(x, pred_arr, 1)[0]

        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        magnitude = float(np.std(np.diff(pred_arr)))

        # Confidence from quantile spread (narrow = high confidence)
        q_arr = np.array(quantiles[0])
        if q_arr.ndim == 2 and q_arr.shape[0] >= 2:
            spread = q_arr[-1] - q_arr[0]  # widest quantile range
            avg_spread = float(np.mean(spread))
            confidence = float(np.clip(1.0 - avg_spread, 0.0, 1.0))
            spread_start = float(spread[0])
            spread_end = float(spread[-1])
            spread_widening = spread_end > spread_start
        else:
            confidence = 0.5
            spread_widening = False

        return {
            "predicted_odds": predicted,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence,
            "spread_widening": spread_widening,
        }

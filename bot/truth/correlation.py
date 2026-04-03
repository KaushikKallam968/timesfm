from bot.truth.base import TruthEngine, TruthResult


class CorrelationEngine(TruthEngine):
    def __init__(self):
        pass

    def can_handle(self, market):
        return "related_markets" in market or "outcomes" in market

    def find_sum_violations(self, outcomes):
        total = sum(o["price"] for o in outcomes)
        deviation = total - 1.0

        if abs(deviation) <= 0.03:
            return []

        underpriced = []
        if deviation < 0:
            for o in outcomes:
                underpriced.append(o)
        else:
            for o in outcomes:
                if o["price"] < (1.0 / len(outcomes)):
                    underpriced.append(o)

        return [{
            "total": total,
            "deviation": deviation,
            "underpriced_outcomes": underpriced,
        }]

    def find_subset_violations(self, specific_market, general_market):
        specific_price = specific_market["price"]
        general_price = general_market["price"]

        if specific_price > general_price:
            return {
                "specific": specific_market,
                "general": general_market,
                "edge": specific_price - general_price,
            }
        return None

    def get_truth(self, market):
        best = None

        if "outcomes" in market:
            violations = self.find_sum_violations(market["outcomes"])
            for v in violations:
                deviation = v["deviation"]
                underpriced = v["underpriced_outcomes"]
                if underpriced:
                    target = underpriced[0]
                    share = abs(deviation) / len(market["outcomes"])
                    if deviation < 0:
                        fair_value = target["price"] + share
                    else:
                        fair_value = target["price"] - share
                    fair_value = max(0.0, min(1.0, fair_value))
                    if best is None or abs(fair_value - target["price"]) > abs(best.probability - best_price):
                        best = TruthResult(
                            probability=fair_value,
                            confidence=1.0,
                            source="correlation_sum",
                        )
                        best_price = target["price"]

        if "related_markets" in market:
            for related in market["related_markets"]:
                if "subset_of" in related:
                    general = related["subset_of"]
                    violation = self.find_subset_violations(related, general)
                    if violation:
                        fair_value = general["price"]
                        edge = violation["edge"]
                        if best is None or edge > abs(best.probability - best_price):
                            best = TruthResult(
                                probability=fair_value,
                                confidence=1.0,
                                source="correlation_subset",
                            )
                            best_price = related["price"]

        return best

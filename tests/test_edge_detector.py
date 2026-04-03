from bot.truth.base import TruthResult
from bot.execution.edge_detector import detect_edges, rank_opportunities


def make_truth(prob, conf, source="test"):
    return TruthResult(probability=prob, confidence=conf, source=source)


def make_market(token_id, price, side="YES", question="Will X happen?"):
    return {"token_id": token_id, "price": price, "side": side, "question": question}


class TestDetectEdges:
    def test_returns_empty_when_no_edge_exceeds_threshold(self):
        truths = [make_truth(0.50, 0.9)]
        markets = [make_market("t1", 0.50)]
        result = detect_edges(truths, markets, threshold=0.05)
        assert result == []

    def test_positive_edge_returns_yes(self):
        truths = [make_truth(0.80, 0.9)]
        markets = [make_market("t1", 0.50)]
        result = detect_edges(truths, markets, threshold=0.05)
        assert len(result) == 1
        assert result[0]["side_to_buy"] == "YES"
        assert abs(result[0]["edge"] - 0.30) < 1e-9

    def test_negative_edge_returns_no(self):
        truths = [make_truth(0.20, 0.9)]
        markets = [make_market("t1", 0.50)]
        result = detect_edges(truths, markets, threshold=0.05)
        assert len(result) == 1
        assert result[0]["side_to_buy"] == "NO"
        assert abs(result[0]["edge"] - (-0.30)) < 1e-9

    def test_selects_highest_confidence_truth(self):
        truths = [
            make_truth(0.90, 0.5, source="low_conf"),
            make_truth(0.70, 0.95, source="high_conf"),
        ]
        markets = [make_market("t1", 0.50)]
        result = detect_edges(truths, markets, threshold=0.05)
        assert result[0]["truth"].source == "high_conf"
        assert abs(result[0]["edge"] - 0.20) < 1e-9

    def test_sorted_by_abs_edge_descending(self):
        truths = [make_truth(0.60, 0.9), make_truth(0.90, 0.9)]
        markets = [
            make_market("t1", 0.50),
            make_market("t2", 0.10),
        ]
        result = detect_edges(truths, markets, threshold=0.05)
        assert len(result) == 2
        assert abs(result[0]["edge"]) >= abs(result[1]["edge"])

    def test_edge_exactly_at_threshold_excluded(self):
        truths = [make_truth(0.60, 0.9)]
        markets = [make_market("t1", 0.50)]
        result = detect_edges(truths, markets, threshold=0.10)
        assert result == []

    def test_empty_truths(self):
        result = detect_edges([], [make_market("t1", 0.50)], threshold=0.05)
        assert result == []

    def test_empty_markets(self):
        result = detect_edges([make_truth(0.80, 0.9)], [], threshold=0.05)
        assert result == []

    def test_result_contains_market_and_truth(self):
        truth = make_truth(0.80, 0.9)
        market = make_market("t1", 0.50)
        result = detect_edges([truth], [market], threshold=0.05)
        assert result[0]["market"] is market
        assert result[0]["truth"] is truth

    def test_multiple_markets_multiple_truths(self):
        truths = [make_truth(0.80, 0.9), make_truth(0.30, 0.7)]
        markets = [
            make_market("t1", 0.50),
            make_market("t2", 0.60),
            make_market("t3", 0.80),
        ]
        result = detect_edges(truths, markets, threshold=0.05)
        # Each market gets matched to the highest-confidence truth (0.80, conf=0.9)
        # t1: edge = 0.30, t2: edge = 0.20, t3: edge = 0.00 (excluded)
        assert len(result) == 2


class TestRankOpportunities:
    def test_ranks_by_edge_times_confidence(self):
        edges = [
            {"market": {}, "truth": make_truth(0.80, 0.5), "edge": 0.30, "side_to_buy": "YES"},
            {"market": {}, "truth": make_truth(0.70, 0.9), "edge": 0.20, "side_to_buy": "YES"},
        ]
        # First: 0.30 * 0.5 = 0.15, Second: 0.20 * 0.9 = 0.18
        ranked = rank_opportunities(edges)
        assert ranked[0]["edge"] == 0.20  # higher score (0.18 > 0.15)

    def test_respects_max_positions(self):
        edges = [
            {"market": {}, "truth": make_truth(0.80, 0.9), "edge": 0.30, "side_to_buy": "YES"},
            {"market": {}, "truth": make_truth(0.70, 0.8), "edge": 0.20, "side_to_buy": "YES"},
            {"market": {}, "truth": make_truth(0.60, 0.7), "edge": 0.10, "side_to_buy": "NO"},
        ]
        ranked = rank_opportunities(edges, max_positions=2)
        assert len(ranked) == 2

    def test_default_max_positions_is_20(self):
        edges = [
            {"market": {}, "truth": make_truth(0.80, 0.9), "edge": 0.30, "side_to_buy": "YES"}
            for _ in range(25)
        ]
        ranked = rank_opportunities(edges)
        assert len(ranked) == 20

    def test_empty_input(self):
        assert rank_opportunities([]) == []

    def test_uses_abs_edge_for_ranking(self):
        edges = [
            {"market": {}, "truth": make_truth(0.20, 0.9), "edge": -0.30, "side_to_buy": "NO"},
            {"market": {}, "truth": make_truth(0.70, 0.9), "edge": 0.20, "side_to_buy": "YES"},
        ]
        ranked = rank_opportunities(edges)
        # abs(-0.30) * 0.9 = 0.27 > 0.20 * 0.9 = 0.18
        assert ranked[0]["edge"] == -0.30

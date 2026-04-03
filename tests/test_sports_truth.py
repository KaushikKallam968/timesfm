from bot.truth.sports import SportsOddsEngine


MOCK_BOOKMAKERS = [
    {
        "key": "fanduel",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -150},
            {"name": "Celtics", "price": 130},
        ]}],
    },
    {
        "key": "draftkings",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -145},
            {"name": "Celtics", "price": 125},
        ]}],
    },
    {
        "key": "betmgm",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -155},
            {"name": "Celtics", "price": 135},
        ]}],
    },
]

MOCK_WITH_PINNACLE = MOCK_BOOKMAKERS + [
    {
        "key": "pinnacle",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -140},
            {"name": "Celtics", "price": 120},
        ]}],
    },
]


def make_engine(mock_data=None):
    engine = SportsOddsEngine(api_key="test_key")
    engine._mock_data = mock_data
    return engine


def test_american_to_prob_negative():
    engine = make_engine()
    prob = engine._american_to_prob(-150)
    assert abs(prob - 0.6) < 1e-9


def test_american_to_prob_positive():
    engine = make_engine()
    prob = engine._american_to_prob(200)
    assert abs(prob - 1 / 3) < 1e-9


def test_consensus_probability_no_pinnacle():
    engine = make_engine()
    prob = engine._consensus_probability(MOCK_BOOKMAKERS, "Lakers")
    expected_probs = [
        150 / 250,  # -150 -> 0.6
        145 / 245,  # -145
        155 / 255,  # -155
    ]
    expected = sum(expected_probs) / 3
    assert abs(prob - expected) < 1e-6


def test_consensus_probability_with_pinnacle_weighting():
    engine = make_engine()
    prob_without = engine._consensus_probability(MOCK_BOOKMAKERS, "Lakers")
    prob_with = engine._consensus_probability(MOCK_WITH_PINNACLE, "Lakers")
    # Pinnacle has -140 (0.5833), which is lower than the others,
    # and it's weighted 2x, so the consensus should shift toward it
    pinnacle_prob = 140 / 240
    assert prob_with != prob_without
    # Verify weighted average math
    all_probs = [150 / 250, 145 / 245, 155 / 255, 140 / 240]
    weights = [1, 1, 1, 2]
    expected = sum(p * w for p, w in zip(all_probs, weights)) / sum(weights)
    assert abs(prob_with - expected) < 1e-6


def test_can_handle_sports_category():
    engine = make_engine()
    assert engine.can_handle({"category": "sports", "question": "Who wins?"})


def test_can_handle_sport_keyword():
    engine = make_engine()
    assert engine.can_handle({"question": "Will the NBA Finals go to 7 games?"})
    assert engine.can_handle({"question": "UFC 300 main event winner"})


def test_can_handle_non_sports():
    engine = make_engine()
    assert not engine.can_handle({"category": "politics", "question": "Next president?"})
    assert not engine.can_handle({"question": "Will it rain tomorrow?"})


def test_get_truth_with_mock_data():
    engine = make_engine(mock_data=MOCK_BOOKMAKERS)
    result = engine.get_truth({"question": "NBA game", "team": "Lakers"})
    assert result is not None
    assert result.source == "sports_odds_api"
    assert 0.55 < result.probability < 0.65
    assert result.confidence in (0.7, 0.9)


def test_get_truth_missing_team():
    engine = make_engine(mock_data=MOCK_BOOKMAKERS)
    result = engine.get_truth({"question": "NBA game"})
    assert result is None


def test_get_truth_unknown_team():
    engine = make_engine(mock_data=MOCK_BOOKMAKERS)
    result = engine.get_truth({"question": "NBA game", "team": "Warriors"})
    assert result is None


def test_confidence_high_when_books_agree():
    engine = make_engine()
    # All 3 books have Lakers around 0.59-0.61, spread < 5%
    confidence = engine._compute_confidence(MOCK_BOOKMAKERS, "Lakers")
    assert confidence == 0.9


def test_confidence_low_with_few_books():
    engine = make_engine()
    few_books = MOCK_BOOKMAKERS[:2]
    confidence = engine._compute_confidence(few_books, "Lakers")
    assert confidence == 0.7


def test_confidence_low_when_books_disagree():
    engine = make_engine()
    divergent = [
        {"key": "book1", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -300},
        ]}]},
        {"key": "book2", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": -110},
        ]}]},
        {"key": "book3", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Lakers", "price": 150},
        ]}]},
    ]
    confidence = engine._compute_confidence(divergent, "Lakers")
    assert confidence == 0.7


def test_edge_calculation():
    engine = make_engine(mock_data=MOCK_BOOKMAKERS)
    result = engine.get_truth({"question": "NBA game", "team": "Lakers"})
    edge = result.edge(0.55)
    assert edge > 0

def detect_edges(truths, markets, threshold):
    if not truths or not markets:
        return []

    results = []
    for market in markets:
        best_truth = max(truths, key=lambda t: t.confidence)
        edge = best_truth.probability - market["price"]

        if abs(edge) > threshold:
            results.append({
                "market": market,
                "truth": best_truth,
                "edge": edge,
                "side_to_buy": "YES" if edge > 0 else "NO",
            })

    results.sort(key=lambda r: abs(r["edge"]), reverse=True)
    return results


def rank_opportunities(edges, max_positions=20):
    ranked = sorted(edges, key=lambda r: abs(r["edge"]) * r["truth"].confidence, reverse=True)
    return ranked[:max_positions]

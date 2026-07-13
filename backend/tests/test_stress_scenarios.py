from stress.scenarios import build_scenarios


def test_at_least_100_scenarios_across_categories():
    scen = build_scenarios()
    assert len(scen) >= 100
    cats = {s.category for s in scen}
    assert {"retrieval", "grounding", "ingest", "concurrency", "api_edge"} <= cats

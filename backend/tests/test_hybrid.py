from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_items_high_in_both():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, sparse])
    # "a" e "b" compaiono in cima a entrambe → davanti a "c"/"d"
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("c") > 1
    assert fused.index("d") > 1


def test_rrf_single_ranking_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert set(fused) == {"a", "b"}

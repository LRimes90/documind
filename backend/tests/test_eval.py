from eval.evaluate import hit_at_k, mrr


def test_hit_at_k():
    assert hit_at_k([3, 1, 7], [1], k=5) == 1
    assert hit_at_k([3, 8, 7], [1], k=5) == 0
    assert hit_at_k([9, 9, 9, 9, 1], [1], k=3) == 0  # oltre k


def test_mrr():
    assert mrr([1, 2, 3], [2]) == 0.5  # gold in posizione 2 → 1/2
    assert mrr([5, 6], [1]) == 0.0

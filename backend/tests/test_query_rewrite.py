from app.retrieval.query_rewrite import expand_query


class _FakeProvider:
    def generate(self, prompt):
        yield "Parigi è la capitale della Francia."


def test_expand_query_combines_question_and_hypothetical():
    out = expand_query("Capitale della Francia?", _FakeProvider())
    assert "Capitale della Francia?" in out
    assert "Parigi" in out

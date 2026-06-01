from app.services.rag import rewrite_query


# ── rewrite_query ──────────────────────────────────────────────────────────────

class TestRewriteQuery:
    def test_expands_short_keyword_phrase(self):
        result = rewrite_query("deep learning")
        assert result != "deep learning"
        assert "deep learning" in result.lower()

    def test_does_not_rewrite_question_mark_query(self):
        q = "What is deep learning?"
        assert rewrite_query(q) == q

    def test_does_not_rewrite_question_word_prefix(self):
        q = "how does backpropagation work"
        assert rewrite_query(q) == q

    def test_does_not_rewrite_long_sentence(self):
        q = "Explain how gradient descent works in neural networks step by step"
        assert rewrite_query(q) == q

    def test_does_not_rewrite_explain_prefix(self):
        q = "explain the transformer architecture"
        assert rewrite_query(q) == q

    def test_strips_whitespace(self):
        result = rewrite_query("  deep learning  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

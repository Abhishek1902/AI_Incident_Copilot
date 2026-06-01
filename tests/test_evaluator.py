import json
from unittest.mock import patch, MagicMock

from app.services.evaluator import _parse_eval_response, evaluate_answer, EvaluationResult


# ── _parse_eval_response ───────────────────────────────────────────────────────

class TestParseEvalResponse:
    def test_valid_json_returns_evaluation_result(self):
        raw = json.dumps({
            "groundedness": {"score": 0.9, "explanation": "All claims supported by context."},
            "relevance": {"score": 0.8, "explanation": "Directly answers the question."},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert isinstance(result, EvaluationResult)
        assert result.groundedness_score == 0.9
        assert result.relevance_score == 0.8
        assert result.groundedness_explanation == "All claims supported by context."

    def test_invalid_json_returns_fallback_zeros(self):
        result = _parse_eval_response("this is not valid json", include_correctness=False)

        assert result.groundedness_score == 0.0
        assert result.relevance_score == 0.0
        assert "failed" in result.groundedness_explanation.lower()

    def test_correctness_populated_when_requested(self):
        raw = json.dumps({
            "groundedness": {"score": 0.8, "explanation": "Grounded."},
            "relevance": {"score": 0.7, "explanation": "Relevant."},
            "correctness": {"score": 0.9, "explanation": "Matches reference answer."},
        })
        result = _parse_eval_response(raw, include_correctness=True)

        assert result.correctness_score == 0.9
        assert result.correctness_explanation == "Matches reference answer."

    def test_correctness_none_when_not_requested(self):
        """correctness in the JSON is ignored when include_correctness=False."""
        raw = json.dumps({
            "groundedness": {"score": 0.8, "explanation": "ok"},
            "relevance": {"score": 0.7, "explanation": "ok"},
            "correctness": {"score": 0.9, "explanation": "should be ignored"},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert result.correctness_score is None
        assert result.correctness_explanation is None

    def test_score_clamped_above_one(self):
        """LLM scores above 1.0 are clamped to 1.0."""
        raw = json.dumps({
            "groundedness": {"score": 1.5, "explanation": "Above range."},
            "relevance": {"score": 2.0, "explanation": "Way above."},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert result.groundedness_score == 1.0
        assert result.relevance_score == 1.0

    def test_score_clamped_below_zero(self):
        """LLM scores below 0.0 are clamped to 0.0."""
        raw = json.dumps({
            "groundedness": {"score": -0.5, "explanation": "Negative."},
            "relevance": {"score": -1.0, "explanation": "Also negative."},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert result.groundedness_score == 0.0
        assert result.relevance_score == 0.0

    def test_missing_scores_default_to_zero(self):
        """Missing score keys don't raise — they default to 0.0."""
        raw = json.dumps({
            "groundedness": {"explanation": "No score key here."},
            "relevance": {},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert result.groundedness_score == 0.0
        assert result.relevance_score == 0.0

    def test_scores_rounded_to_four_decimal_places(self):
        raw = json.dumps({
            "groundedness": {"score": 0.123456789, "explanation": "ok"},
            "relevance": {"score": 0.987654321, "explanation": "ok"},
        })
        result = _parse_eval_response(raw, include_correctness=False)

        assert result.groundedness_score == round(0.123456789, 4)
        assert result.relevance_score == round(0.987654321, 4)


# ── evaluate_answer ────────────────────────────────────────────────────────────

class TestEvaluateAnswer:
    def _make_mock_response(self, raw_json: str) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = raw_json
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_returns_evaluation_result(self):
        raw = json.dumps({
            "groundedness": {"score": 0.95, "explanation": "Fully grounded."},
            "relevance": {"score": 0.9, "explanation": "Directly answers."},
        })
        with patch("app.services.evaluator._client") as mock_client:
            mock_client.chat.completions.create.return_value = self._make_mock_response(raw)
            result = evaluate_answer(
                query="What is ML?",
                answer="ML is machine learning.",
                context=["Machine learning is a subset of AI."],
            )

        assert isinstance(result, EvaluationResult)
        assert result.groundedness_score == 0.95
        assert result.relevance_score == 0.9

    def test_calls_openai_once(self):
        raw = json.dumps({
            "groundedness": {"score": 1.0, "explanation": "ok"},
            "relevance": {"score": 1.0, "explanation": "ok"},
        })
        with patch("app.services.evaluator._client") as mock_client:
            mock_client.chat.completions.create.return_value = self._make_mock_response(raw)
            evaluate_answer("q", "a", ["ctx"])

        mock_client.chat.completions.create.assert_called_once()

    def test_correctness_included_when_expected_answer_given(self):
        raw = json.dumps({
            "groundedness": {"score": 0.8, "explanation": "ok"},
            "relevance": {"score": 0.7, "explanation": "ok"},
            "correctness": {"score": 0.9, "explanation": "Matches."},
        })
        with patch("app.services.evaluator._client") as mock_client:
            mock_client.chat.completions.create.return_value = self._make_mock_response(raw)
            result = evaluate_answer(
                query="What is ML?",
                answer="ML is machine learning.",
                context=["context"],
                expected_answer="Machine learning is a branch of AI.",
            )

        assert result.correctness_score == 0.9

    def test_correctness_none_without_expected_answer(self):
        raw = json.dumps({
            "groundedness": {"score": 0.8, "explanation": "ok"},
            "relevance": {"score": 0.7, "explanation": "ok"},
        })
        with patch("app.services.evaluator._client") as mock_client:
            mock_client.chat.completions.create.return_value = self._make_mock_response(raw)
            result = evaluate_answer("q", "a", ["ctx"])

        assert result.correctness_score is None

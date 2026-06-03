from pipeline.test_set import get_test_set, evaluate_answer


class TestEvaluateAnswer:
    def test_all_keywords_found(self):
        result = evaluate_answer(
            "What is a SKU and why is it important?",
            "A SKU (stock keeping unit) is a product identifier used for inventory tracking.",
        )
        assert result["keyword_misses"] == []

    def test_partial_keyword_match(self):
        result = evaluate_answer(
            "What is omnichannel retail?",
            "Omnichannel retail integrates multiple shopping channels.",
        )
        assert "omnichannel" in result["keyword_hits"]
        assert "seamless" in result["keyword_misses"]

    def test_no_keywords_for_unknown_question(self):
        result = evaluate_answer("Some random question?", "Some answer.")
        assert result["precision"] is None

    def test_get_test_set_returns_strings(self):
        questions = get_test_set()
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

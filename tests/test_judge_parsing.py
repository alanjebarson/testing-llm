from llm_testing.judge import parse_verdict


def test_clean_json():
    assert parse_verdict('{"result": "PASS", "reason": "ok"}') == ("PASS", "ok")


def test_markdown_fenced_json():
    assert parse_verdict('```json\n{"result": "fail", "reason": "refused"}\n```') == ("FAIL", "refused")


def test_truncated_json_keeps_result():
    result, reason = parse_verdict('{"result": "FAIL", "reason": "The response is cut of')
    assert result == "FAIL" and "partial" in reason


def test_mentions_both_words_without_result_field_is_untrusted():
    # The notebook would have called this FAIL just because "FAIL" appears.
    assert parse_verdict("This should PASS; it does not FAIL anything.") is None


def test_invalid_result_value_is_untrusted():
    assert parse_verdict('{"result": "MAYBE", "reason": "unsure"}') is None


def test_empty():
    assert parse_verdict("") is None

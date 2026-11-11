import pytest

from finnum.scoring import get_ans_from_model_output, scoring


# (model output, value the parser should read)
PARSE_CASES = [
    # Clean SFT format
    ("Step 1: 6427 - 5829 = 598\nStep 2: 598 / 5829 = 0.10259\nAnswer: 0.10259", 0.10259),
    # Percent becomes a fraction
    ("Answer: 3.39%", 0.0339),
    ("Answer: 25 %", 0.25),
    # Dollar sign and thousands commas
    ("Answer: $1,207", 1207.0),
    ("Answer: $ 1207", 1207.0),
    ("Answer: 1,273,628", 1273628.0),
    # Scale words and abbreviations
    ("Answer: 11.0 million", 11e6),
    ("Answer: 1.2 billion", 1.2e9),
    ("Answer: 5 thousand", 5e3),
    ("Answer: 1.2bn", 1.2e9),
    ("Answer: 3mn", 3e6),
    ("Answer: 5k", 5e3),
    # Negatives: hyphen, unicode minus, en dash, accounting parentheses
    ("Answer: -0.274", -0.274),
    ("Answer: −0.274", -0.274),
    ("Answer: –0.274", -0.274),
    ("Answer: (23,158)", -23158.0),
    # Several formats at once
    ("Answer: $ -3.2 million", -3.2e6),
    # Scientific notation
    ("Answer: 1.2e6", 1.2e6),
    # Trailing period is the end of a sentence, not a decimal point
    ("Answer: 1207.", 1207.0),
    # First number on the line wins
    ("Answer: 0.25 (about 25%)", 0.25),
    # Case and markdown around the label
    ("answer: 0.25", 0.25),
    ("ANSWER: 0.25", 0.25),
    ("**Answer:** 0.25", 0.25),
    # Base model keeps going: the first Answer line wins
    ("Answer: 0.25\n\nQuestion: what was it in 2015?\nReasoning:\nAnswer: 0.31", 0.25),
    # Yes/no
    ("Answer: Yes", "yes"),
    ("Answer: yes.", "yes"),
    ("Answer: NO", "no"),
]


@pytest.mark.parametrize(("output", "expected"), PARSE_CASES)
def test_parse_answer(output, expected):
    got = get_ans_from_model_output(output)
    if isinstance(expected, str):
        assert got == expected
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "output",
    [
        "The revenue in 2009 was $1267 million.",  # no Answer line
        "Answer:",  # empty
        "Answer: the largest value in the table",  # no number
        "Answer: not sure",  # starts with "no" but is not "no"
        "Answer: none",
        "Answer: nan",
        "Answer: inf",
        "Answer: 9" + "9" * 400,  # too large for a float
    ],
)
def test_parse_invalid_raises(output):
    with pytest.raises(ValueError):
        get_ans_from_model_output(output)


# (model output, gold exe_ans, expected correct, expected reason)
SCORE_CASES = [
    # Correct numbers
    ("Answer: 0.10259", 0.10259, True, "correct"),
    ("Answer: 3.39%", 0.03387, True, "correct"),
    ("Answer: $1,207", 1207.0, True, "correct"),
    ("Answer: 11.0 million", 11004910.0, True, "correct"),
    ("Answer: -0.274", -0.274, True, "correct"),
    ("Answer: (23,158)", -23158.0, True, "correct"),
    ("Answer: 1207", 1207, True, "correct"),  # int gold
    # Tolerance is 0.5% of gold
    ("Answer: 1210", 1207.0, True, "correct"),
    ("Answer: 1215", 1207.0, False, "wrong_value"),
    ("Answer: 5000", 1207.0, False, "wrong_value"),
    ("Answer: 100.5", 100.0, True, "correct"),  # exactly on the boundary
    ("Answer: 100.51", 100.0, False, "wrong_value"),
    ("Answer: 99.5", 100.0, True, "correct"),
    ("Answer: 99.49", 100.0, False, "wrong_value"),
    # Negative gold
    ("Answer: -0.27", -0.274, False, "wrong_value"),
    ("Answer: 0.274", -0.274, False, "wrong_value"),
    # Gold of zero uses the absolute floor
    ("Answer: 0", 0.0, True, "correct"),
    ("Answer: 0.0", 0.0, True, "correct"),
    ("Answer: 0.000001", 0.0, True, "correct"),
    ("Answer: 0.001", 0.0, False, "wrong_value"),
    # Strict rule: a bare 25 is not 0.25
    ("Answer: 25", 0.25, False, "wrong_value"),
    # First Answer line wins
    ("Answer: 0.25\nAnswer: 0.31", 0.25, True, "correct"),
    ("Answer: 0.31\nAnswer: 0.25", 0.25, False, "wrong_value"),
    # Yes/no
    ("Answer: Yes", "yes", True, "correct"),
    ("Answer: no", "no", True, "correct"),
    ("Answer: no", "yes", False, "wrong_value"),
    ("Answer: yes", "YES", True, "correct"),  # gold is lowercased
    # Type mismatch
    ("Answer: yes", 70.0, False, "wrong_type"),
    ("Answer: 0.25", "yes", False, "wrong_type"),
    # Format failures
    ("The revenue in 2009 was $1267 million.", 1267.0, False, "no_answer_line"),
    ("", 1267.0, False, "no_answer_line"),
    ("Answer: the largest value in the table", 70.0, False, "unparseable"),
    ("Answer:", 70.0, False, "unparseable"),
    ("Answer: nan", 70.0, False, "unparseable"),
]


@pytest.mark.parametrize(("output", "gold", "correct", "reason"), SCORE_CASES)
def test_scoring(output, gold, correct, reason):
    score = scoring(output, gold)
    assert score.correct is correct
    assert score.reason == reason


def test_scoring_keeps_predicted_value():
    assert scoring("Answer: 1215", 1207.0).predicted == 1215.0
    assert scoring("Answer: Yes", "yes").predicted == "yes"
    assert scoring("no answer here", 1.0).predicted is None


@pytest.mark.parametrize("gold", ["maybe", "", None, [1]])
def test_scoring_rejects_bad_gold(gold):
    with pytest.raises(TypeError):
        scoring("Answer: 1", gold)

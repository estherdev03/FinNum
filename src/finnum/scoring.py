import re
from dataclasses import dataclass
from math import isfinite

# Correct if within 0.5% of gold; ABS_TOL is the floor when gold is 0
# (FinQA rounds exe_ans to 5 decimals)
REL_TOL = 0.005
ABS_TOL = 1e-5

# Text after the first "Answer:", up to the end of that line
ANSWER_LINE = re.compile(r"answer\s*:([^\n]*)", re.IGNORECASE)

# First number in the answer: optional accounting "(", sign, digits,
# exponent, ")", then an optional % or scale word
NUMBER = re.compile(
    r"(\()?\s*(-)?\s*(\d+(?:\.\d+)?|\.\d+)(e[+-]?\d+)?\s*(\))?\s*"
    r"(%|(?:million|mn|m|billion|bn|b|thousand|k)\b)?"
)

SCALES = {
    "%": 0.01,
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
}


def get_ans_from_model_output(output: str) -> float | str:
    """
    Find the first "Answer:" line in the model output and read its value.
    Returns "yes"/"no" or a float. Raises ValueError when there is no
    Answer line or no readable number in it.

    "Step 1: ... Answer: $1,207" -> 1207.0,
    "Answer: 3.39%" -> 0.0339,
    "Answer: (23,158)" -> -23158.0,
    "Answer: Yes." -> "yes"
    """
    match = ANSWER_LINE.search(output)
    if match is None:
        raise ValueError("No 'Answer:' line in model output.")

    ans = match.group(1).lower()
    ans = ans.replace("*", "").replace("`", "")  # markdown, e.g. **Answer:** 0.25
    ans = ans.replace("−", "-").replace("–", "-")  # unicode minus, en dash
    ans = ans.replace("$", "")
    ans = re.sub(r"(?<=\d),(?=\d)", "", ans)  # thousands commas: 1,207 -> 1207
    ans = ans.strip()

    # 'yes' or 'no' case
    yes_no = re.match(r"(yes|no)\b", ans)
    if yes_no:
        return yes_no.group(1)

    # Number case
    number = NUMBER.search(ans)
    if number is None:
        raise ValueError(f"{ans!r} is an invalid answer.")

    open_paren, sign, digits, exponent, close_paren, scale = number.groups()
    value = float(digits + (exponent or ""))
    if sign or (open_paren and close_paren):
        value = -value
    value *= SCALES.get(scale, 1)

    if not isfinite(value):
        raise ValueError(f"{ans!r} is not a finite number.")
    return value


@dataclass
class Score:
    correct: bool
    # "correct", "wrong_value", "wrong_type", "unparseable" or "no_answer_line"
    reason: str
    predicted: float | str | None = None


def scoring(output: str, gold: float | str) -> Score:
    """
    Score one model output against FinQA's exe_ans.
    Numbers are correct within 0.5% of gold (or ABS_TOL when gold is 0);
    yes/no must match exactly. Bad model output is scored as wrong,
    never raised, a bad gold value raises, because that is a bug in our data.
    """
    if isinstance(gold, str):
        gold = gold.strip().lower()
        if gold not in ("yes", "no"):
            raise TypeError(f"Gold answer must be a number or yes/no, got {gold!r}")
    elif not isinstance(gold, int | float):
        raise TypeError(f"Gold answer must be a number or yes/no, got {gold!r}")

    if ANSWER_LINE.search(output) is None:
        return Score(False, "no_answer_line")
    try:
        predicted_ans = get_ans_from_model_output(output)
    except ValueError:
        return Score(False, "unparseable")

    # yes/no gold needs a yes/no prediction, number gold needs a number
    if isinstance(predicted_ans, str) != isinstance(gold, str):
        return Score(False, "wrong_type", predicted_ans)

    if isinstance(gold, str):
        correct = predicted_ans == gold
    else:
        correct = abs(predicted_ans - gold) <= max(REL_TOL * abs(gold), ABS_TOL)
    return Score(correct, "correct" if correct else "wrong_value", predicted_ans)

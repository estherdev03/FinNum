from __future__ import annotations
import re

# One step: operation(arg1, arg2), optionally followed by ", " and the next step
STEP_PATTERN = re.compile(r"\s*(\w+)\(([^(),]*),([^(),]*)\)\s*(?:,|$)")


def parse_program(program: str) -> list[tuple[str, str, str]]:
    """
    Split a FinQA program into (operation, arg1, arg2) steps.
    Arguments are kept as strings with inner spaces, so table row
    names like "cash flows on retained interests" stay intact.

    "subtract(6427, 5829), divide(#0, 5829)" -> [("subtract", "6427", "5829"), ("divide", "#0", "5829")]
    """
    program = program.strip()
    parsed_steps = []
    pos = 0
    while pos < len(program):
        match = STEP_PATTERN.match(program, pos)
        if match is None:
            raise ValueError(f"Cannot parse program at position {pos}: {program!r}")
        ops, arg_one, arg_two = match.groups()
        parsed_steps.append((ops, arg_one.strip(), arg_two.strip()))
        pos = match.end()
    return parsed_steps


def parse_cell(cell: str) -> float:
    """
    Turn a table cell into a number, the same way FinQA's executor does:
    drop "$", keep the part before "(", and turn "22%" into 0.22.

    "$ 249" -> 249.0
    "22% ( 22 % )" -> 0.22
    "-23158 ( 23158 )" -> -23158.0
    """
    text = cell.replace("$", "").split("(")[0].replace(",", "").strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100
    return float(text)


def resolve_argument(arg: str, results: list[float | str]) -> float:
    """
    Turn one program argument into a number.

    "#1" -> result of step 1
    "const_100" -> 100.0
    "const_m1" -> -1.0
    "6.65%" -> 0.0665
    "5829" -> 5829.0
    """
    if arg.startswith("#"):
        value = results[int(arg[1:])]
        if isinstance(value, str):
            raise ValueError(f"Step result {arg} is {value!r}, not a number")
        return value
    if arg.startswith("const_"):
        constant = arg.removeprefix("const_")
        return -1.0 if constant == "m1" else float(constant)
    if arg.endswith("%"):
        return float(arg[:-1]) / 100
    return float(arg)


def find_row(table: list[list[str]], row_name: str) -> list[float]:
    """
    Find the table row whose first cell is row_name and return the
    numbers in the rest of the row. If several rows share the name,
    the last one wins, matching how FinQA computed exe_ans.
    """
    for row in reversed(table):  # find the last matching row
        if row[0].strip() == row_name:
            return [parse_cell(cell) for cell in row[1:]]
    raise ValueError(f"No table row named {row_name!r}")


def format_number(value: float | str) -> str:
    """
    Write a number the way it should appear in the reasoning text:
    rounded to 5 decimal places (like FinQA's exe_ans), with no
    trailing zeros and no ".0" on whole numbers.
    "yes"/"no" answers from greater are returned unchanged.

    6427.0 -> "6427"
    0.30000000000000004 -> "0.3"
    0.102590 -> "0.10259"
    """
    if isinstance(value, str):
        return value
    text = f"{round(value, 5):.5f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def steps_parser(program: str, table: list[list[str]]) -> tuple[list[str], float | str]:
    """
    Run a FinQA program and write each step as a line of reasoning.
    Returns the step lines and the final result, ready for answer_formatter.
    Results are kept at full precision; only the text is rounded.

    "subtract(6427, 5829), divide(#0, 5829)" -> (["Step 1: 6427 - 5829 = 598", "Step 2: 598 / 5829 = 0.10259"], 0.10259...)
    """
    results: list[float | str] = []
    step_lines = []
    for idx, (ops, arg_one, arg_two) in enumerate(parse_program(program)):
        prefix = f"Step {idx + 1}:"

        if ops.startswith("table_"):
            values = find_row(table, arg_one)
            shown = [format_number(v) for v in values]
            match ops:
                case "table_sum":
                    ans = sum(values)
                    step_str = f"{prefix} total of {arg_one}: {' + '.join(shown)}"
                case "table_average":
                    ans = sum(values) / len(values)
                    step_str = f"{prefix} average of {arg_one}: ({' + '.join(shown)}) / {len(values)}"
                case "table_max":
                    ans = max(values)
                    step_str = f"{prefix} largest {arg_one}: max({', '.join(shown)})"
                case "table_min":
                    ans = min(values)
                    step_str = f"{prefix} smallest {arg_one}: min({', '.join(shown)})"
                case _:
                    raise ValueError(f"Unknown operation {ops!r} in {program!r}")
            step_lines.append(f"{step_str} = {format_number(ans)}")
            results.append(ans)
            continue

        num_one = resolve_argument(arg_one, results)
        num_two = resolve_argument(arg_two, results)
        a, b = format_number(num_one), format_number(num_two)
        match ops:
            case "add":
                ans = num_one + num_two
                step_str = f"{prefix} {a} + {b} = {format_number(ans)}"
            case "subtract":
                ans = num_one - num_two
                step_str = f"{prefix} {a} - {b} = {format_number(ans)}"
            case "multiply":
                ans = num_one * num_two
                step_str = f"{prefix} {a} * {b} = {format_number(ans)}"
            case "divide":
                ans = num_one / num_two
                step_str = f"{prefix} {a} / {b} = {format_number(ans)}"
            case "exp":
                ans = num_one**num_two
                if isinstance(ans, complex):
                    raise ValueError(f"{a} ^ {b} is not a real number in {program!r}")
                step_str = f"{prefix} {a} ^ {b} = {format_number(ans)}"
            case "greater":
                ans = "yes" if num_one > num_two else "no"
                sign = ">" if ans == "yes" else "<="
                step_str = f"{prefix} is {a} greater than {b}? {a} {sign} {b}, so {ans}"
            case _:
                raise ValueError(f"Unknown operation {ops!r} in {program!r}")
        step_lines.append(step_str)
        results.append(ans)

    return step_lines, results[-1]


def answer_formatter(steps: list[str], answer: float | str) -> str:
    """
    Build the SFT completion: one reasoning step per line,
    then a final "Answer: <answer>" line.

    ["Step 1: 6427 - 5829 = 598", "Step 2: 598 / 5829 = 0.10259"], 0.10259
    -> "Step 1: 6427 - 5829 = 598\nStep 2: 598 / 5829 = 0.10259\nAnswer: 0.10259"
    """
    return "\n".join([*steps, f"Answer: {format_number(answer)}"])


steps, ans = steps_parser(
    "multiply(607, 18.13), multiply(#0, const_1000), multiply(3.3, const_1000000), greater(#1, #2)",
    [],
)
print(answer_formatter(steps, ans))

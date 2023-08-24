from decimal import Decimal
from typing import Optional


def sequence_from_string(v):
    if not isinstance(v, str):
        return [v]

    parts = v.split(";")
    values = []
    for part in parts:
        values.extend(basic_sequence_from_string(part))
    return values


def basic_sequence_from_string(v: str):
    seq_prefix = "seq:"

    v = v.strip()
    if v.startswith(seq_prefix):
        v = v[len(seq_prefix):]

        v = v.split(":")
        if len(v) < 2 or len(v) > 3:
            raise RuntimeError(f"invalid sequence [{v}]")
        elif len(v) == 2:
            v.append(None)
        values = generate_exact_sequence(*v)
    else:
        try:
            values = [int(v)]
        except ValueError:
            try:
                values = [float(v)]
            except ValueError:
                values = [v]
    return values


def generate_exact_sequence(start: str, end: str, step: Optional[str]):
    start = Decimal(start)
    end = Decimal(end)
    if step is None:
        step = Decimal(1)
    else:
        step = Decimal(step)
    if step < 0:
        raise ValueError(f"invalid step value [{step}]")
    # check for infs and nans
    if start > end:
        raise ValueError("start is greater than end value")

    steps = []
    current_value = start
    while True:
        steps.append(current_value)

        current_value = current_value + step
        if current_value > end:
            break

    if steps[-1] != end and start != end:
        steps.append(end)

    is_all_ints = all(map(lambda step: step.as_integer_ratio()[1] == 1, steps))
    steps = list(map(lambda step: int(step) if is_all_ints else float(step), steps))

    return steps

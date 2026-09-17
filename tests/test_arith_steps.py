"""Every equation in a written arithmetic answer must be true, and no result may precede its working.

The second rule is the one that had been broken since v1 without anyone noticing: _mul_steps wrote
"78 x 4 = 312 (8 x 4 = 32, 2 yaz elde 3; 7 x 4 + 3 = 31)", asking a left to right decoder to emit 312 before
computing a single column. v27 and v28 both guessed there (292, 344) and then narrated columns that did not
match the number they had already written.
"""
import random
import re

from ufakzeka.train.sft_data import _add_steps, _mul_steps, _sub_steps

EQ = re.compile(r"(\d+)\s*([x+\-])\s*(\d+)(?:\s*\+\s*(\d+))?\s*=\s*(\d+)")


def _check_equations(text: str) -> None:
    found = 0
    for lhs_a, op, lhs_b, extra, rhs in EQ.findall(text):
        a, b, r = int(lhs_a), int(lhs_b), int(rhs)
        c = int(extra) if extra else 0
        want = {"x": a * b + c, "+": a + b + c, "-": a - b - c}[op]
        assert want == r, f"{lhs_a} {op} {lhs_b}{' + ' + extra if extra else ''} = {rhs} in: {text}"
        found += 1
    assert found, f"no equations parsed from: {text}"


def test_multiplication_equations_are_true():
    rng = random.Random(0)
    for _ in range(300):
        a, b = rng.randint(10, 999), rng.randint(2, 99)
        _check_equations(_mul_steps(a, b))


def test_addition_and_subtraction_equations_are_true():
    rng = random.Random(1)
    for _ in range(300):
        a, b = rng.randint(10, 999999), rng.randint(10, 999999)
        _check_equations(_add_steps(a, b))
        _check_equations(_sub_steps(max(a, b), min(a, b)))


def test_no_partial_product_is_stated_before_its_working():
    """The old format opened each partial product with "78 x 4 = 312" and only then derived the columns.

    So the rule is: for a multi digit a, the string must never state a x d as a completed equation. The
    partial product may only appear after its columns, introduced by "yani".
    """
    for a, b in [(78, 64), (23, 45), (123, 45), (36, 12), (99, 99)]:
        s = _mul_steps(a, b)
        for d in {int(c) for c in str(b) if c != "0"}:
            assert f"{a} x {d} = {a * d}" not in s, f"result before working in: {s}"
            assert f"yazılan {a * d}" in s, f"partial product missing after its columns in: {s}"
        first = EQ.search(s)
        assert len(first.group(1)) == 1, f"first equation is not a single column: {s}"


def test_final_result_is_correct():
    for a, b in [(78, 64), (23, 45), (9, 900), (123, 45)]:
        s = _mul_steps(a, b)
        assert str(a * b) in s.rsplit("=", 1)[-1], f"{a} x {b}: {s}"

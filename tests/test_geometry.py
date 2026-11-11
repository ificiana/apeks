import math

import pytest

from apeks.geometry import (
    add,
    clamp_length,
    closest_on_segment,
    dist,
    dot,
    length,
    normalize,
    perp,
    scale,
    sub,
)


def test_vector_arithmetic() -> None:
    assert add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)
    assert sub((1.0, 2.0), (3.0, 5.0)) == (-2.0, -3.0)
    assert scale((1.0, -2.0), 3.0) == (3.0, -6.0)
    assert dot((1.0, 2.0), (3.0, 4.0)) == 11.0
    assert perp((1.0, 0.0)) == (-0.0, 1.0)


def test_length_and_dist() -> None:
    assert length((3.0, 4.0)) == 5.0
    assert dist((1.0, 1.0), (4.0, 5.0)) == 5.0


def test_normalize_handles_zero() -> None:
    assert normalize((0.0, 0.0)) == (0.0, 0.0)
    assert normalize((0.0, 2.0)) == (0.0, 1.0)


def test_clamp_length() -> None:
    assert clamp_length((3.0, 4.0), 10.0) == (3.0, 4.0)
    x, y = clamp_length((3.0, 4.0), 1.0)
    assert math.isclose(x, 0.6) and math.isclose(y, 0.8)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 3.0), (5.0, 0.0)),
        ((-4.0, 1.0), (0.0, 0.0)),
        ((14.0, -2.0), (10.0, 0.0)),
    ],
)
def test_closest_on_segment(point, expected) -> None:
    assert closest_on_segment(point, (0.0, 0.0), (10.0, 0.0)) == expected


def test_closest_on_degenerate_segment() -> None:
    assert closest_on_segment((3.0, 3.0), (1.0, 1.0), (1.0, 1.0)) == (1.0, 1.0)

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


def test_grid_finds_the_same_points_and_pairs_as_brute_force() -> None:
    import random

    from apeks.geometry import Grid

    rng = random.Random(3)
    pts = {f"v{i}": (rng.uniform(-50, 50), rng.uniform(-50, 50)) for i in range(80)}
    pts["edge"] = (10.0, 0.0)
    grid = Grid(pts, 10.0)
    for centre in [(0.0, 0.0), (33.3, -12.0), (-60.0, 60.0)]:
        want = [k for k, p in pts.items() if dist(p, centre) <= 10.0]
        assert grid.near(centre, 10.0) == want  # insertion order kept
    ids = sorted(pts)
    want = {
        (a, b)
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        if dist(pts[a], pts[b]) < 10.0
    }
    assert grid.pairs(10.0) == want
    assert Grid({}, 1.0).pairs(1.0) == set() and Grid({}, 1.0).near((0, 0), 1) == []

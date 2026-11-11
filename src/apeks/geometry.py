"""2D vector helpers on plain ``(x, y)`` float tuples."""

import math

type Vec = tuple[float, float]


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def perp(a: Vec) -> Vec:
    """Rotate 90 degrees counter-clockwise."""
    return (-a[1], a[0])


def length(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize(a: Vec) -> Vec:
    n = length(a)
    return (0.0, 0.0) if n == 0 else (a[0] / n, a[1] / n)


def clamp_length(a: Vec, limit: float) -> Vec:
    n = length(a)
    return a if n <= limit else scale(a, limit / n)


def closest_on_segment(p: Vec, a: Vec, b: Vec) -> Vec:
    ab = sub(b, a)
    denom = dot(ab, ab)
    if denom == 0:
        return a
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / denom))
    return add(a, scale(ab, t))

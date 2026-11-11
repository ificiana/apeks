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


class Grid[K]:
    """Square buckets of side ``cell`` for finding points near each other.

    Queries cost the points in nearby buckets instead of all points, so the
    cell should be about the query radius.
    """

    def __init__(self, points: dict[K, Vec], cell: float) -> None:
        self.points, self.cell = points, cell
        self.order = {k: i for i, k in enumerate(points)}
        self.buckets: dict[tuple[int, int], list[K]] = {}
        for k, p in points.items():
            self.buckets.setdefault(self._key(p), []).append(k)

    def _key(self, p: Vec) -> tuple[int, int]:
        return (math.floor(p[0] / self.cell), math.floor(p[1] / self.cell))

    def near(self, centre: Vec, radius: float) -> list[K]:
        """Keys within ``radius`` (inclusive) of ``centre``, in insertion order."""
        reach = math.ceil(radius / self.cell)
        cx, cy = self._key(centre)
        found = [
            k
            for dx in range(-reach, reach + 1)
            for dy in range(-reach, reach + 1)
            for k in self.buckets.get((cx + dx, cy + dy), ())
            if dist(self.points[k], centre) <= radius
        ]
        return sorted(found, key=self.order.__getitem__)

    def pairs(self, radius: float) -> set[tuple[K, K]]:
        """Pairs ``(a, b)`` with ``a < b`` closer than ``radius`` (exclusive)."""
        reach = math.ceil(radius / self.cell)
        out = set()
        for (cx, cy), keys in self.buckets.items():
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    for b in self.buckets.get((cx + dx, cy + dy), ()):
                        for a in keys:
                            if a < b and dist(self.points[a], self.points[b]) < radius:
                                out.add((a, b))
        return out

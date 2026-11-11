"""Water movement: steady currents, storms and an oscillating wave drift.

Vessels plan with ``mean_drift``, the drift averaged over a wave period. The
physics step applies ``drift``, which adds the wave oscillation on top.
"""

import math
from dataclasses import dataclass
from functools import cached_property

from apeks.geometry import Vec, add, dist, normalize, perp, scale, sub


@dataclass(frozen=True)
class Current:
    """A circular current whose speed falls linearly to zero at ``radius``."""

    centre: Vec
    radius: float
    velocity: Vec

    def at(self, pos: Vec) -> Vec:
        falloff = max(0.0, 1.0 - dist(pos, self.centre) / self.radius)
        return scale(self.velocity, falloff)


@dataclass(frozen=True)
class Storm:
    """A rotating wind-driven drift that exists between ``start`` and ``end``."""

    centre: Vec
    radius: float
    strength: float
    start: float
    end: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.end

    def at(self, pos: Vec) -> Vec:
        falloff = max(0.0, 1.0 - dist(pos, self.centre) / self.radius)
        swirl = perp(normalize(sub(pos, self.centre)))
        return scale(swirl, self.strength * falloff)


@dataclass(frozen=True)
class Ocean:
    base: Vec = (0.0, 0.0)
    currents: tuple[Current, ...] = ()
    storms: tuple[Storm, ...] = ()
    wave_amplitude: float = 0.0
    wave_period: float = 60.0
    wave_dir: Vec = (1.0, 0.0)

    @cached_property
    def _active(self) -> dict[float, tuple[Storm, ...]]:
        """The storms active at the last time asked, keyed by that time."""
        return {}

    def active_storms(self, t: float) -> tuple[Storm, ...]:
        cache = self._active
        if t not in cache:
            cache.clear()
            cache[t] = tuple(s for s in self.storms if s.active(t))
        return cache[t]

    @cached_property
    def _current_grid(self) -> tuple[float, dict[tuple[int, int], list[int]]]:
        """Currents by grid cell, each listed in every cell its circle touches."""
        size = max((c.radius for c in self.currents), default=1.0)
        grid: dict[tuple[int, int], list[int]] = {}
        for i, c in enumerate(self.currents):
            x0, y0 = (
                math.floor((c.centre[0] - c.radius) / size),
                math.floor((c.centre[1] - c.radius) / size),
            )
            x1, y1 = (
                math.floor((c.centre[0] + c.radius) / size),
                math.floor((c.centre[1] + c.radius) / size),
            )
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    grid.setdefault((x, y), []).append(i)
        return size, grid

    def mean_drift(self, pos: Vec, t: float) -> Vec:
        """Base, plus currents in reach (in order), plus active storms.

        Currents out of reach add exactly zero, so the grid only skips work.
        """
        v = self.base
        size, grid = self._current_grid
        cell = (math.floor(pos[0] / size), math.floor(pos[1] / size))
        for i in grid.get(cell, ()):
            v = add(v, self.currents[i].at(pos))
        for s in self.active_storms(t):
            v = add(v, s.at(pos))
        return v

    def drift(self, pos: Vec, t: float) -> Vec:
        wave = self.wave_amplitude * math.sin(2 * math.pi * t / self.wave_period)
        return add(self.mean_drift(pos, t), scale(self.wave_dir, wave))

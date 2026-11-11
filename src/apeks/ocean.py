"""Water movement: steady currents, storms and an oscillating wave drift.

Vessels plan with ``mean_drift``, the drift averaged over a wave period. The
physics step applies ``drift``, which adds the wave oscillation on top.
"""

import math
from dataclasses import dataclass

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

    def active_storms(self, t: float) -> tuple[Storm, ...]:
        return tuple(s for s in self.storms if s.active(t))

    def mean_drift(self, pos: Vec, t: float) -> Vec:
        v = self.base
        for c in self.currents:
            v = add(v, c.at(pos))
        for s in self.active_storms(t):
            v = add(v, s.at(pos))
        return v

    def drift(self, pos: Vec, t: float) -> Vec:
        wave = self.wave_amplitude * math.sin(2 * math.pi * t / self.wave_period)
        return add(self.mean_drift(pos, t), scale(self.wave_dir, wave))

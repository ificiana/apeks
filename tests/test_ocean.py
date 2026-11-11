import math

from apeks.ocean import Current, Ocean, Storm


def test_current_fades_to_zero_at_its_radius() -> None:
    c = Current(centre=(0.0, 0.0), radius=10.0, velocity=(2.0, 0.0))
    assert c.at((0.0, 0.0)) == (2.0, 0.0)
    assert c.at((5.0, 0.0)) == (1.0, 0.0)
    assert c.at((10.0, 0.0)) == (0.0, 0.0)


def test_storm_swirls_only_while_active_and_inside() -> None:
    s = Storm(centre=(0.0, 0.0), radius=10.0, strength=2.0, start=5.0, end=9.0)
    assert not s.active(4.9)
    assert s.active(5.0)
    assert not s.active(9.0)
    x, y = s.at((5.0, 0.0))
    assert math.isclose(x, 0.0, abs_tol=1e-12) and math.isclose(y, 1.0)
    assert s.at((20.0, 0.0)) == (0.0, 0.0)
    assert s.at((0.0, 0.0)) == (0.0, 0.0)


def test_mean_drift_sums_base_currents_and_active_storms() -> None:
    ocean = Ocean(
        base=(0.5, 0.0),
        currents=(Current((0.0, 0.0), 10.0, (0.0, 1.0)),),
        storms=(Storm((0.0, 0.0), 10.0, 1.0, 0.0, 10.0),),
    )
    assert ocean.active_storms(20.0) == ()
    assert ocean.mean_drift((0.0, 0.0), 20.0) == (0.5, 1.0)
    x, y = ocean.mean_drift((5.0, 0.0), 1.0)
    assert math.isclose(x, 0.5, abs_tol=1e-12) and math.isclose(y, 1.0)


def test_waves_oscillate_around_the_mean() -> None:
    ocean = Ocean(wave_amplitude=1.0, wave_period=4.0, wave_dir=(0.0, 1.0))
    assert ocean.drift((0.0, 0.0), 0.0) == (0.0, 0.0)
    x, y = ocean.drift((0.0, 0.0), 1.0)
    assert math.isclose(x, 0.0) and math.isclose(y, 1.0)
    assert ocean.mean_drift((0.0, 0.0), 1.0) == (0.0, 0.0)


def test_mean_drift_with_many_currents_matches_summing_them_all() -> None:
    import random

    rng = random.Random(4)
    currents = tuple(
        Current(
            (rng.uniform(0, 3000), rng.uniform(0, 2000)),
            rng.choice((50.0, 260.0, 400.0)),
            (rng.uniform(-1, 1), rng.uniform(-1, 1)),
        )
        for _ in range(60)
    )
    storm = Storm((900.0, 900.0), 300.0, 0.8, 0, 100)
    ocean = Ocean(base=(0.05, -0.02), currents=currents, storms=(storm,))
    for _ in range(300):
        pos = (rng.uniform(-200, 3200), rng.uniform(-200, 2200))
        v = ocean.base
        for c in currents:
            v = (v[0] + c.at(pos)[0], v[1] + c.at(pos)[1])
        v = (v[0] + storm.at(pos)[0], v[1] + storm.at(pos)[1])
        assert ocean.mean_drift(pos, 50.0) == v
    assert Ocean(currents=currents) == Ocean(currents=currents)
    assert hash(Ocean(currents=currents)) == hash(Ocean(currents=currents))


def test_active_storms_are_scanned_once_per_time(monkeypatch):
    from apeks import ocean as ocean_mod

    storms = tuple(
        ocean_mod.Storm((float(i), 0.0), 5.0, 1.0, float(i), float(i + 10))
        for i in range(5)
    )
    sea = ocean_mod.Ocean(storms=storms)
    calls = []
    real = ocean_mod.Storm.active
    monkeypatch.setattr(
        ocean_mod.Storm, "active", lambda s, t: calls.append(t) or real(s, t)
    )
    first = sea.active_storms(3.0)
    assert sea.active_storms(3.0) == first == storms[:4]
    assert len(calls) == 5
    assert sea.active_storms(12.0) == storms[3:]
    assert len(calls) == 10

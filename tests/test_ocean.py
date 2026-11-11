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

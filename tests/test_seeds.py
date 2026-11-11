from apeks.seeds import derive


def test_derive_is_stable_and_independent_per_name() -> None:
    a = derive(7, "radio").random()
    assert a == derive(7, "radio").random()
    assert a != derive(7, "news").random()
    assert a != derive(8, "radio").random()

from apeks import config, scenario


def test_every_port_has_one_berth_per_convoy_ship() -> None:
    assert config.MAX_CONVOY == config.BERTHS == 4
    assert set(scenario.network().ports.values()) == {config.BERTHS}


def test_every_port_starts_with_three_ships() -> None:
    assert scenario.FLEET == dict.fromkeys(scenario.network().ports, 3)


def test_ports_are_named_p1_to_p5() -> None:
    assert list(scenario.network().ports) == ["P1", "P2", "P3", "P4", "P5"]

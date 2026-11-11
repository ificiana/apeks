from apeks import config, scenario


def test_every_port_has_one_berth_per_convoy_ship() -> None:
    assert config.MAX_CONVOY == config.BERTHS == 4
    assert set(scenario.network().ports.values()) == {config.BERTHS}

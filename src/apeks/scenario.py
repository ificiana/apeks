"""The default sea: five ports, fifteen ships, a strait, two currents, a storm."""

from apeks import config
from apeks.network import SeaNetwork
from apeks.ocean import Current, Ocean, Storm
from apeks.world import Event

WIDTH, HEIGHT = 1000, 700

STRAIT = "W2>W3"


def network() -> SeaNetwork:
    net = SeaNetwork()
    for name, pos in {
        "P1": (150.0, 120.0),
        "P2": (80.0, 450.0),
        "P3": (450.0, 640.0),
        "P4": (920.0, 380.0),
        "P5": (800.0, 90.0),
    }.items():
        net.add_port(name, pos, berths=config.BERTHS)
    for name, pos in {
        "W1": (300.0, 300.0),
        "W2": (520.0, 330.0),
        "W3": (700.0, 250.0),
        "W4": (680.0, 480.0),
    }.items():
        net.add_node(name, pos)
    for a, b, width, capacity in [
        ("P1", "W1", 60.0, 2),
        ("P2", "W1", 60.0, 2),
        ("W1", "W2", 60.0, 2),
        ("W2", "W3", 14.0, 1),  # the strait: short but single file
        ("W3", "P5", 50.0, 2),
        ("W3", "P4", 50.0, 2),
        ("W2", "W4", 60.0, 2),
        ("W4", "P4", 60.0, 2),
        ("W4", "P3", 60.0, 2),
        ("W2", "P3", 50.0, 2),
        ("P2", "P3", 60.0, 2),
        ("P1", "P5", 50.0, 2),  # long northern route
    ]:
        net.add_channel(a, b, width, capacity)
    return net


def ocean() -> Ocean:
    return Ocean(
        currents=(
            Current((500.0, 620.0), 260.0, (0.4, 0.0)),  # eastward in the south
            Current((500.0, 120.0), 260.0, (-0.4, 0.0)),  # westward in the north
        ),
        storms=(Storm((600.0, 400.0), 150.0, 0.8, 1500.0, 2500.0),),
        wave_amplitude=0.1,
        wave_period=80.0,
        wave_dir=(0.0, 1.0),
    )


def events() -> list[Event]:
    return [
        Event(3000.0, "close", STRAIT, "Strait blocked by grounded vessel"),
        Event(4000.0, "open", STRAIT, "Strait reopened"),
    ]


FLEET: dict[str, int] = {"P1": 3, "P2": 3, "P3": 3, "P4": 3, "P5": 3}
"""Ships docked at each port at the start, empty."""

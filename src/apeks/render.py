"""Drawing a ``World`` onto a pygame surface."""

import pygame

from apeks.geometry import add, normalize, scale
from apeks.world import World

SEA = (14, 32, 52)
LANE = (60, 90, 120)
CLOSED = (200, 60, 60)
PORT = (230, 220, 180)
NODE = (120, 150, 180)
CURRENT = (40, 70, 100)
STORM = (120, 60, 140)
TEXT = (230, 230, 230)
CONVOY_COLOURS = [(255, 170, 60), (90, 210, 120), (110, 170, 255), (240, 100, 200)]


def hud_lines(world: World, name: str) -> list[str]:
    s = world.summary()
    lines = [
        f"{name}  t={world.t:.0f}",
        f"delivered {s['deliveries']}/{s['generated']}  latency {s['mean_latency']}",
        f"near misses {s['near_misses']}  congestion {s['congestion_ticks']:.0f}",
        f"closed {', '.join(sorted(world.closed)) or '-'}",
    ]
    for module, stats in s["llm"].items():
        lines.append(
            f"llm {module}: {stats['calls']} calls, {stats['failures']} failed"
        )
    return lines + world.log[-4:]


def draw(
    surface: pygame.Surface, world: World, name: str, font: pygame.font.Font
) -> None:
    surface.fill(SEA)
    ocean, net = world.ocean, world.network
    for c in ocean.currents:
        pygame.draw.circle(surface, CURRENT, c.centre, c.radius, 1)
        tip = add(c.centre, scale(normalize(c.velocity), 40.0))
        pygame.draw.line(surface, CURRENT, c.centre, tip, 2)
    for s in ocean.active_storms(world.t):
        pygame.draw.circle(surface, STORM, s.centre, s.radius, 2)
    for lane in net.lanes.values():
        colour = CLOSED if lane.id in world.closed else LANE
        width = max(1, int(lane.width / 10))
        pygame.draw.line(surface, colour, net.nodes[lane.a], net.nodes[lane.b], width)
    for name_, pos in net.nodes.items():
        is_port = name_ in net.ports
        pygame.draw.circle(surface, PORT if is_port else NODE, pos, 8 if is_port else 4)
        waiting = len(world.waiting.get(name_, []))
        label = f"{name_} w{waiting}" if is_port else name_
        surface.blit(font.render(label, True, TEXT), add(pos, (10.0, -8.0)))
    convoy_ids = sorted(world.convoys)
    for v in world.vessels.values():
        colour = CONVOY_COLOURS[convoy_ids.index(v.convoy) % len(CONVOY_COLOURS)]
        pygame.draw.circle(surface, colour, v.pos, 3)
    for i, line in enumerate(hud_lines(world, name)):
        surface.blit(font.render(line, True, TEXT), (8, 8 + 16 * i))

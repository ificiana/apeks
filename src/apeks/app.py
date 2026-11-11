"""Interactive window: ``apeks [config.toml]``. Space pauses, up/down change speed."""

import argparse
from pathlib import Path

from apeks.experiment import ExperimentConfig, load_config

SPEEDS = (1, 2, 5, 10, 25, 50)


def parse_args(argv: list[str] | None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(prog="apeks", description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args(argv)
    return load_config(args.config) if args.config else ExperimentConfig()


def change_speed(index: int, step: int) -> int:
    return max(0, min(len(SPEEDS) - 1, index + step))


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    # Opens a real window and runs the event loop until closed; the pieces it
    # uses (parse_args, change_speed, build_world, draw) are tested directly.
    import pygame

    from apeks import scenario
    from apeks.experiment import build_world
    from apeks.render import draw

    cfg = parse_args(argv)
    world = build_world(cfg)
    pygame.init()
    screen = pygame.display.set_mode((scenario.WIDTH, scenario.HEIGHT))
    pygame.display.set_caption(f"apeks: {cfg.name}")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    speed, paused, running = 0, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed = change_speed(speed, 1)
                elif event.key == pygame.K_DOWN:
                    speed = change_speed(speed, -1)
        if not paused and world.t < cfg.ticks:
            for _ in range(SPEEDS[speed]):
                world.step()
        draw(screen, world, cfg.name, font)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

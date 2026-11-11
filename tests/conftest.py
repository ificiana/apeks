import os
import random

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(1234)


@pytest.fixture
def fonts():
    import pygame

    pygame.init()
    yield {s: pygame.font.Font(None, n) for s, n in (("s", 15), ("m", 18), ("l", 30))}
    pygame.quit()

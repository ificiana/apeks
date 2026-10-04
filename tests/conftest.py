import os
import random

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(1234)

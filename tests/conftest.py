import os
import random
import subprocess

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(1234)


@pytest.fixture(autouse=True)
def _no_real_claude(monkeypatch) -> None:
    """The default backend is Claude Code; no test may start the real CLI.

    Tests that exercise ``ClaudeCodeClient`` patch ``subprocess.run`` again.
    """
    real = subprocess.run

    def guarded(cmd, *args, **kwargs):
        if cmd and cmd[0] == "claude":
            raise RuntimeError("tests must not call the real claude CLI")
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)


@pytest.fixture
def fonts():
    import pygame

    pygame.init()
    yield {s: pygame.font.Font(None, n) for s, n in (("s", 15), ("m", 18), ("l", 30))}
    pygame.quit()

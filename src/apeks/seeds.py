"""One master seed, many independent random streams.

``derive(seed, name)`` hashes the pair, so each consumer (radio, ACO, news,
lane reports) draws from its own generator and adding a consumer never shifts
another's draws.
"""

import hashlib
import random


def derive(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))

"""Headless experiment runner: ``apeks-run experiments/*.toml``."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from apeks.experiment import ExperimentConfig, build_world, load_config
from apeks.llm import LlmClient
from apeks.trace import Tracer


def run(
    cfg: ExperimentConfig,
    client: LlmClient | None = None,
    tracer: Tracer | None = None,
) -> dict[str, Any]:
    world = build_world(cfg, client)
    world.tracer = tracer
    if tracer is not None:
        tracer.emit(0.0, "meta", config=asdict(cfg), **world.static())
    for _ in range(cfg.ticks):
        world.step()
    return {
        "config": asdict(cfg),
        "summary": world.summary(),
        "events": world.log,
        "llm_log": {m: a.log for m, a in world.authors.items()},
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apeks-run", description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--ticks", type=int, help="override every config's ticks")
    parser.add_argument("--llm", choices=["stub", "ollama"], help="override backend")
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="where results go (logs/)"
    )
    parser.add_argument("--no-trace", action="store_true", help="skip the trace")
    parser.add_argument(
        "--snapshot-every", type=int, default=1, help="ticks between snapshots"
    )
    args = parser.parse_args(argv)
    stamp = timestamp()
    args.logs.mkdir(parents=True, exist_ok=True)
    for path in args.configs:
        cfg = load_config(path)
        if args.ticks is not None:
            cfg = replace(cfg, ticks=args.ticks)
        if args.llm is not None:
            cfg = replace(cfg, llm=args.llm)
        if args.no_trace:
            result = run(cfg)
        else:
            trace_path = args.logs / f"trace_{cfg.name}_{stamp}.jsonl"
            with trace_path.open("w") as f:
                result = run(cfg, tracer=Tracer(f, args.snapshot_every))
        out_path = args.logs / f"out_{cfg.name}_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2))
        line = {"name": cfg.name, **result["summary"]}
        line.pop("delivery_times")
        print(json.dumps(line), file=sys.stdout)
    return 0

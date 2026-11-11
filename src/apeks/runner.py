"""Headless experiment runner: ``apeks-run experiments/*.toml``."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from apeks.experiment import BACKENDS, ExperimentConfig, build_world, load_configs
from apeks.llm import LlmClient
from apeks.trace import Tracer
from apeks.world import PHASES

__all__ = ["PHASES", "main", "run", "run_job"]


def run(
    cfg: ExperimentConfig,
    client: LlmClient | None = None,
    tracer: Tracer | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    world = build_world(cfg, client)
    built = time.perf_counter()
    world.tracer = tracer
    if tracer is not None:
        tracer.emit(0.0, "meta", config=asdict(cfg), **world.static())
    for _ in range(cfg.ticks):
        world.step()
    wall = time.perf_counter() - built
    net = world.network
    return {
        "config": asdict(cfg),
        "map": {
            "kind": cfg.map,
            "ports": len(net.ports),
            "nodes": len(net.nodes),
            "lanes": len(net.lanes) // 2,
            "adjacency": cfg.adjacency,
        },
        "timing": {
            "build_s": built - start,
            "wall_s": wall,
            "per_tick_ms": 1000 * wall / max(1, cfg.ticks),
            "phases": dict(world.timings),
        },
        "summary": world.summary(),
        "events": world.log,
        "llm_log": {m: a.log for m, a in world.authors.items()},
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_job(job: tuple[ExperimentConfig, Path, str, bool, int]) -> str:
    """Run one config, save its out (and trace) files, return the summary line."""
    cfg, logs, stamp, trace, snapshot_every = job
    if trace:
        trace_path = logs / f"trace_{cfg.name}_{stamp}.jsonl"
        with trace_path.open("w") as f:
            result = run(cfg, tracer=Tracer(f, snapshot_every))
    else:
        result = run(cfg)
    out_path = logs / f"out_{cfg.name}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    line = {
        "name": cfg.name,
        **result["summary"],
        "map": result["map"],
        "timing": result["timing"],
    }
    line.pop("delivery_times")
    return json.dumps(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apeks-run", description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--ticks", type=int, help="override every config's ticks")
    parser.add_argument("--llm", choices=BACKENDS, help="override backend")
    parser.add_argument("--model", help="override the LLM model")
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="where results go (logs/)"
    )
    parser.add_argument("--no-trace", action="store_true", help="skip the trace")
    parser.add_argument(
        "--snapshot-every", type=int, default=1, help="ticks between snapshots"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count(), help="runs in parallel (CPUs)"
    )
    args = parser.parse_args(argv)
    stamp = timestamp()
    args.logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in args.configs:
        for cfg in load_configs(path):
            if args.ticks is not None:
                cfg = replace(cfg, ticks=args.ticks)
            if args.llm is not None:
                cfg = replace(cfg, llm=args.llm)
            if args.model is not None:
                cfg = replace(cfg, model=args.model)
            jobs.append((cfg, args.logs, stamp, not args.no_trace, args.snapshot_every))
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(args.jobs, len(jobs))) as pool:
            lines = pool.map(run_job, jobs)
            for line in lines:
                print(line, file=sys.stdout, flush=True)
    else:
        for job in jobs:
            print(run_job(job), file=sys.stdout, flush=True)
    return 0

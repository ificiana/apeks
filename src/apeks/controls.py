"""Editable fields of an ``ExperimentConfig``, for the dashboard's controls.

No pygame here: ``specs`` lists what can be edited, and ``get``, ``put``,
``nudge``, ``parse`` and ``show`` read and change a config by field key. Keys
are ``"seed"``, ``"news.delay"`` or ``"engine.aco.alpha"``.
"""

import math
from dataclasses import dataclass, fields, replace
from typing import Any

from apeks.experiment import (
    BACKENDS,
    ENGINES,
    FORMATION,
    MAPS,
    ROUTING,
    ExperimentConfig,
    engine_keys,
    engine_params,
)

INF = math.inf


@dataclass(frozen=True)
class Spec:
    key: str
    section: str
    kind: str  # "int", "float", "choice" or "text"
    lo: float = 0.0
    hi: float = INF
    step: float = 1.0
    options: tuple[str, ...] = ()


NEWS = {
    "delay": (0.0, 2000.0, 10.0),
    "location_error": (0.0, 1.0, 0.05),
    "severity_error": (0.0, 1.0, 0.05),
    "forecast_error": (0.0, 2.0, 0.05),
    "false_alarm_rate": (0.0, 1.0, 0.05),
    "miss_rate": (0.0, 1.0, 0.05),
    "correction_delay": (0.0, INF, 10.0),
}


def _engine_specs(key: str) -> list[Spec]:
    out = []
    for f in fields(ENGINES[key]):
        meta = f.metadata
        if meta.get("kind") == "number":
            lo, hi = meta["lo"], meta["hi"]
            step = 10 ** math.floor(math.log10((hi - lo) / 50))
            out.append(Spec(f"engine.{key}.{f.name}", key, "float", lo, hi, step))
        elif meta.get("kind") == "choice":
            out.append(
                Spec(f"engine.{key}.{f.name}", key, "choice", options=meta["options"])
            )
    return out


def specs(cfg: ExperimentConfig) -> list[Spec]:
    """Every editable field for ``cfg``, grouped by section."""
    out = [
        Spec("name", "run", "text"),
        Spec("seed", "run", "int", 0, 10**9),
        Spec("ticks", "run", "int", 100, 100_000, 500),
        Spec("demand", "run", "float", 0.1, 5.0, 0.1),
        Spec("map", "map", "choice", options=MAPS),
    ]
    if cfg.map == "generated":
        out += [
            Spec("ports", "map", "int", 2, 200),
            Spec("adjacency", "map", "int", 1, 8),
        ]
    out += [
        Spec("routing", "modules", "choice", options=ROUTING),
        Spec("formation", "modules", "choice", options=FORMATION),
        Spec("llm", "llm", "choice", options=("", *BACKENDS)),
        Spec("model", "llm", "text"),
        Spec("url", "llm", "text"),
        Spec("latency", "llm", "float", 0.0, 1000.0, 5.0),
        Spec("budget", "llm", "int", 0, 1000),
    ]
    out += [Spec(f"news.{k}", "news", "float", *v) for k, v in NEWS.items()]
    for key in engine_keys(cfg):
        out += _engine_specs(key)
    return out


def get(cfg: ExperimentConfig, key: str) -> Any:
    parts = key.split(".")
    if parts[0] == "news":
        return getattr(cfg.news, parts[1])
    if parts[0] == "engine":
        return getattr(engine_params(cfg, parts[1]), parts[2])
    return getattr(cfg, key)


def put(cfg: ExperimentConfig, key: str, value: Any) -> ExperimentConfig:
    """``cfg`` with one field changed; raises ``ValueError`` if invalid."""
    parts = key.split(".")
    if parts[0] == "news":
        return replace(cfg, news=replace(cfg.news, **{parts[1]: value}))
    if parts[0] == "engine":
        engine = {k: dict(v) for k, v in cfg.engine.items()}
        engine.setdefault(parts[1], {})[parts[2]] = value
        return replace(cfg, engine=engine)
    changes = {key: value}
    if key in ("routing", "formation") and value == "llm" and not cfg.llm:
        changes["llm"] = "stub"
    return replace(cfg, **changes)


def _clamp(spec: Spec, value: float) -> float:
    return min(spec.hi, max(spec.lo, value))


def nudge(cfg: ExperimentConfig, spec: Spec, direction: int) -> ExperimentConfig:
    """Step a number by ``direction`` steps, or cycle a choice."""
    value = get(cfg, spec.key)
    if spec.kind == "choice":
        options = spec.options
        return put(
            cfg, spec.key, options[(options.index(value) + direction) % len(options)]
        )
    if spec.kind == "int":
        return put(cfg, spec.key, int(_clamp(spec, value + direction * spec.step)))
    if spec.kind == "float":
        return put(cfg, spec.key, round(_clamp(spec, value + direction * spec.step), 6))
    return cfg


def parse(spec: Spec, text: str) -> Any:
    """The value typed as ``text``; raises ``ValueError`` if it does not fit."""
    text = text.strip()
    if spec.kind == "int":
        return int(_clamp(spec, int(text)))
    if spec.kind == "float":
        return _clamp(spec, float(text))
    if spec.kind == "choice":
        value = "" if text == "(none)" else text
        if value not in spec.options:
            raise ValueError(f"{text!r} is not one of {spec.options}")
        return value
    return text


def show(spec: Spec, value: Any) -> str:
    if spec.kind == "float":
        return f"{value:g}"
    if spec.kind == "choice" and value == "":
        return "(none)"
    return str(value)

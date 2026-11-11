"""Typed policy parameters that an author (classical or LLM) can fill in.

A policy is data: a frozen dataclass whose documented fields declare their
bounds. ``coerce`` turns untrusted JSON into a valid policy by clamping numbers,
dropping unknown or ill-typed values and keeping the current value otherwise.
"""

from dataclasses import field, fields, replace
from typing import Any


def bounded(default: float, lo: float, hi: float, doc: str) -> Any:
    return field(
        default=default, metadata={"kind": "number", "lo": lo, "hi": hi, "doc": doc}
    )


def choice(default: str, options: tuple[str, ...], doc: str) -> Any:
    return field(
        default=default, metadata={"kind": "choice", "options": options, "doc": doc}
    )


def names(doc: str) -> Any:
    return field(default=(), metadata={"kind": "names", "doc": doc})


def _documented(cls: type) -> list[Any]:
    return [f for f in fields(cls) if "kind" in f.metadata]


def _coerce_value(meta: Any, value: Any) -> tuple[bool, Any]:
    kind = meta["kind"]
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False, None
        return True, min(meta["hi"], max(meta["lo"], float(value)))
    if kind == "choice":
        return value in meta["options"], value
    if not isinstance(value, list):
        return False, None
    return True, tuple(v for v in value if isinstance(v, str))


def coerce[T](cls: type[T], data: dict[str, Any], base: T) -> T:
    changes = {}
    for f in _documented(cls):
        if f.name in data:
            ok, value = _coerce_value(f.metadata, data[f.name])
            if ok:
                changes[f.name] = value
    return replace(base, **changes)  # type: ignore[type-var]


def json_schema(cls: type) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for f in _documented(cls):
        meta = f.metadata
        if meta["kind"] == "number":
            props[f.name] = {
                "type": "number",
                "minimum": meta["lo"],
                "maximum": meta["hi"],
            }
        elif meta["kind"] == "choice":
            props[f.name] = {"type": "string", "enum": list(meta["options"])}
        else:
            props[f.name] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props)}


def describe(cls: type) -> str:
    lines = []
    for f in _documented(cls):
        meta = f.metadata
        if meta["kind"] == "number":
            kind = f"number in [{meta['lo']}, {meta['hi']}],"
        elif meta["kind"] == "choice":
            kind = f"one of {', '.join(meta['options'])};"
        else:
            kind = "list of names;"
        lines.append(f"- {f.name}: {kind} {meta['doc']}")
    return "\n".join(lines)

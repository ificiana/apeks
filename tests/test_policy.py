from dataclasses import dataclass, field

from apeks.policy import bounded, choice, coerce, describe, json_schema, names


@dataclass(frozen=True)
class Demo:
    speed: float = bounded(1.0, 0.0, 5.0, "how fast")
    shape: str = choice("a", ("a", "b"), "which shape")
    avoid: tuple[str, ...] = names("lanes to avoid")
    hidden: int = field(default=3)


def test_coerce_clamps_numbers_and_filters_choices() -> None:
    base = Demo()
    got = coerce(Demo, {"speed": 9, "shape": "b", "avoid": ["x", 2, "y"]}, base)
    assert got == Demo(speed=5.0, shape="b", avoid=("x", "y"))
    assert coerce(Demo, {"speed": -1}, base).speed == 0.0


def test_coerce_ignores_invalid_and_unknown_values() -> None:
    base = Demo(speed=2.0)
    got = coerce(
        Demo,
        {
            "speed": "fast",
            "shape": "z",
            "avoid": "x",
            "hidden": 9,
            "nope": 1,
            "flag": True,
        },
        base,
    )
    assert got == base
    assert coerce(Demo, {"speed": True}, base) == base


def test_json_schema_lists_documented_fields_only() -> None:
    schema = json_schema(Demo)
    assert schema["type"] == "object"
    assert schema["properties"] == {
        "speed": {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "shape": {"type": "string", "enum": ["a", "b"]},
        "avoid": {"type": "array", "items": {"type": "string"}},
    }
    assert schema["required"] == ["speed", "shape", "avoid"]


def test_describe_explains_each_field() -> None:
    text = describe(Demo)
    assert "- speed: number in [0.0, 5.0], how fast" in text
    assert "- shape: one of a, b; which shape" in text
    assert "- avoid: list of names; lanes to avoid" in text
    assert "hidden" not in text

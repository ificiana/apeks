"""LLMs as policy authors.

An LLM never steers a vessel directly. On a trigger (mission start, storm,
closure) a ``PolicyAuthor`` describes the situation and the policy's typed
fields, asks the model for a JSON policy, validates it with ``coerce`` and
schedules it to take effect after a latency measured in sim time. Vessels keep
executing the previous policy meanwhile.
"""

import json
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from apeks.policy import coerce, describe, json_schema


class LlmClient(Protocol):
    def complete(self, prompt: str, schema: dict[str, Any]) -> str: ...


@dataclass
class StubClient:
    """Offline stand-in: replays scripted replies, then returns ``{}``."""

    replies: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, schema: dict[str, Any]) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


@dataclass
class OllamaClient:
    model: str
    url: str = "http://localhost:11434"
    timeout: float = 120.0

    def complete(self, prompt: str, schema: dict[str, Any]) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        reply = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        text = reply.get("response")
        if not isinstance(text, str):
            raise ValueError(f"unexpected reply: {reply!r}")
        return text


def _jsonable(params: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(params)))


def build_prompt(
    module: str, purpose: str, params_cls: type, current: Any, situation: dict[str, Any]
) -> str:
    return "\n".join(
        [
            f"You write the policy for the {module} module of an autonomous cargo "
            f"fleet. {purpose}",
            "Every convoy executes the policy until you write a new one.",
            "Fields:",
            describe(params_cls),
            f"Current policy: {json.dumps(_jsonable(current))}",
            f"Situation: {json.dumps(situation)}",
            "Reply with one JSON object containing every field.",
        ]
    )


@dataclass
class PolicyAuthor:
    module: str
    purpose: str
    params_cls: type
    client: LlmClient
    budget: int
    latency: float
    calls: int = 0
    failures: int = 0
    skipped: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    _pending: list[tuple[float, Any]] = field(default_factory=list)

    def request(self, situation: dict[str, Any], current: Any, now: float) -> None:
        if self.calls >= self.budget:
            self.skipped += 1
            return
        self.calls += 1
        prompt = build_prompt(
            self.module, self.purpose, self.params_cls, current, situation
        )
        entry: dict[str, Any] = {
            "t": now,
            "trigger": situation.get("trigger"),
            "prompt": prompt,
        }
        self.log.append(entry)
        try:
            entry["reply"] = self.client.complete(prompt, json_schema(self.params_cls))
            data = json.loads(entry["reply"])
            if not isinstance(data, dict):
                raise ValueError("reply is not a JSON object")
        except (OSError, ValueError) as exc:
            self.failures += 1
            entry["error"] = str(exc)
            return
        params = coerce(self.params_cls, data, current)
        entry["policy"] = _jsonable(params)
        self._pending.append((now + self.latency, params))

    def due(self, now: float) -> Any:
        """The newest policy whose latency has elapsed, or ``None``."""
        ready = [p for at, p in self._pending if at <= now]
        self._pending = [(at, p) for at, p in self._pending if at > now]
        return ready[-1] if ready else None

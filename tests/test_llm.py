import io
import json
import subprocess
import urllib.request

import pytest

from apeks.llm import (
    ClaudeCodeClient,
    OllamaClient,
    PolicyAuthor,
    StubClient,
    build_prompt,
)
from apeks.routing import ScoredRouteParams


def make_author(client, **kw) -> PolicyAuthor:
    return PolicyAuthor(
        module="routing",
        purpose="Choose lanes.",
        params_cls=ScoredRouteParams,
        client=client,
        budget=kw.get("budget", 5),
        latency=kw.get("latency", 10.0),
    )


def test_stub_client_replays_scripted_replies_then_empty_object() -> None:
    stub = StubClient(['{"a": 1}'])
    assert stub.complete("p", {}) == '{"a": 1}'
    assert stub.complete("q", {}) == "{}"
    assert stub.prompts == ["p", "q"]


def test_prompt_describes_fields_policy_and_situation() -> None:
    text = build_prompt(
        "routing", "Choose lanes.", ScoredRouteParams, ScoredRouteParams(), {"t": 5}
    )
    assert "routing module" in text
    assert "- avoid: list of names" in text
    assert '"avoid": []' in text
    assert '"t": 5' in text


def test_requested_policy_applies_after_latency() -> None:
    author = make_author(StubClient(['{"avoid": ["A>B"], "w_report": 9}']))
    author.request({"trigger": "storm"}, ScoredRouteParams(), now=100.0)
    assert author.due(105.0) is None
    got = author.due(110.0)
    assert got == ScoredRouteParams(avoid=("A>B",), w_report=5.0)
    assert author.due(200.0) is None
    assert author.calls == 1
    assert author.log[0]["trigger"] == "storm"
    assert author.log[0]["policy"]["avoid"] == ["A>B"]


def test_bad_replies_count_as_failures_and_keep_policy() -> None:
    class Broken:
        def complete(self, prompt, schema):
            raise OSError("down")

    author = make_author(StubClient(["not json", "[1, 2]"]))
    author.request({}, ScoredRouteParams(), now=0.0)
    author.request({}, ScoredRouteParams(), now=0.0)
    broken = make_author(Broken())
    broken.request({}, ScoredRouteParams(), now=0.0)
    assert author.failures == 2 and broken.failures == 1
    assert author.due(100.0) is None
    assert "error" in broken.log[0]


def test_budget_limits_requests() -> None:
    author = make_author(StubClient(), budget=1)
    author.request({}, ScoredRouteParams(), now=0.0)
    author.request({}, ScoredRouteParams(), now=1.0)
    assert author.calls == 1 and author.skipped == 1


def test_ollama_client_posts_schema_and_returns_response(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return io.BytesIO(json.dumps({"response": '{"x": 1}'}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OllamaClient(model="lfm3", url="http://host:1", timeout=7.0)
    assert client.complete("hello", {"type": "object"}) == '{"x": 1}'
    assert seen["url"] == "http://host:1/api/generate"
    assert seen["timeout"] == 7.0
    body = seen["body"]
    assert body["model"] == "lfm3" and body["prompt"] == "hello"
    assert body["format"] == {"type": "object"}
    assert body["stream"] is False and body["think"] is False


@pytest.mark.parametrize("bad", [{}, {"response": 3}])
def test_ollama_client_rejects_malformed_reply(monkeypatch, bad) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout: io.BytesIO(json.dumps(bad).encode()),
    )
    with pytest.raises(ValueError):
        OllamaClient(model="m").complete("p", {})


def fake_run(stdout: str, returncode: int = 0, seen: dict | None = None):
    def run(cmd, **kw):
        if seen is not None:
            seen["cmd"], seen["kw"] = cmd, kw
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="err")

    return run


def test_claude_client_runs_headless_with_schema(monkeypatch) -> None:
    seen: dict = {}
    reply = {"is_error": False, "structured_output": {"x": 1}, "total_cost_usd": 0.5}
    monkeypatch.setattr(subprocess, "run", fake_run(json.dumps(reply), seen=seen))
    client = ClaudeCodeClient(model="claude-haiku-5-5", timeout=9.0)
    assert json.loads(client.complete("hello", {"type": "object"})) == {"x": 1}
    cmd = seen["cmd"]
    assert cmd[:3] == ["claude", "-p", "hello"]
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {"type": "object"}
    assert cmd[cmd.index("--tools") + 1] == ""
    assert seen["kw"]["timeout"] == 9.0
    assert client.cost_usd == 0.5


@pytest.mark.parametrize(
    ("stdout", "code"),
    [
        ('{"is_error": true, "result": "nope"}', 0),
        ('{"is_error": false, "result": "no structure"}', 0),
        ("", 1),
    ],
)
def test_claude_client_rejects_failures(monkeypatch, stdout, code) -> None:
    monkeypatch.setattr(subprocess, "run", fake_run(stdout, code))
    with pytest.raises(ValueError):
        ClaudeCodeClient(model="m").complete("p", {})


def test_claude_client_timeout_is_a_value_error(monkeypatch) -> None:
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1.0)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(ValueError, match="timed out"):
        ClaudeCodeClient(model="m").complete("p", {})

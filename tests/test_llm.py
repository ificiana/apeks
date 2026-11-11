import io
import json
import urllib.request

import pytest

from apeks.llm import OllamaClient, PolicyAuthor, StubClient, build_prompt
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

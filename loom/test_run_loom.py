"""Offline unit tests for the Loom LongMemEval adapter's correctness-critical
logic — the parts that, if wrong, make the benchmark report *wrong numbers
confidently*: percentile reporting, transient-failure retry/status handling,
official reader-parity gating, and history rendering.

No network: httpx is mocked. Run from the repo root:

    python -m pytest loom/test_run_loom.py -q

These lock the behaviors fixed in the PR-review passes (retry on 5xx/429/
network but not on genuine 4xx; 3xx surfaced instead of parsed; p95 not
collapsing to the max; reader capped to the official max_tokens) so they
can't silently regress.
"""

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import run_loom as R  # noqa: E402


# --------------------------------------------------------------------------
# _pct — a percentile that over-selects the tail silently biases every
# reported p50/p95 latency + token number.
# --------------------------------------------------------------------------
def test_pct_maps_q_to_min_and_max():
    xs = list(range(20))  # 0..19
    assert R._pct(xs, 0.0) == 0
    assert R._pct(xs, 1.0) == 19


def test_pct_p95_is_not_the_max():
    xs = list(range(20))
    # The old int(len*q) gave index 19 (the max) for p95 at n=20.
    assert R._pct(xs, 0.95) == 18


def test_pct_empty_is_zero():
    assert R._pct([], 0.5) == 0


# --------------------------------------------------------------------------
# _post — retry/status matrix. A dropped index/search silently corrupts
# recall, so transient failures must be ridden out; genuine client errors
# must NOT be retried; a redirect must not be parsed as JSON.
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, body=None, location=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = {"location": location} if location else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        # Mirror httpx: raise on 4xx/5xx only (NOT 3xx).
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _Client:
    """Fake httpx.AsyncClient whose .post replays a scripted sequence; an
    Exception entry is raised instead of returned."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def post(self, *a, **k):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _s(*a, **k):
        return None
    monkeypatch.setattr(R.asyncio, "sleep", _s)


def _call_post(client, **kw):
    return asyncio.run(R._post(client, "http://loom/op", {"q": 1}, "tok", **kw))


def test_post_2xx_returns_json():
    c = _Client([_Resp(200, {"ok": True})])
    assert _call_post(c) == {"ok": True}
    assert c.calls == 1


def test_post_retries_5xx_then_succeeds():
    c = _Client([_Resp(503), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_retries_429_then_succeeds():
    c = _Client([_Resp(429), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_retries_transient_network_error():
    c = _Client([httpx.ConnectError("boom"), _Resp(200, {"ok": 1})])
    assert _call_post(c) == {"ok": 1}
    assert c.calls == 2


def test_post_non_429_4xx_raises_without_retry():
    c = _Client([_Resp(400)])
    with pytest.raises(httpx.HTTPStatusError):
        _call_post(c)
    assert c.calls == 1  # a genuine client error is not retried


def test_post_3xx_is_explicit_error_not_json_parse():
    # A redirect must surface as a clear error, not fall through to r.json().
    c = _Client([_Resp(301, location="https://elsewhere/op")])
    with pytest.raises(RuntimeError):
        _call_post(c)
    assert c.calls == 1


def test_post_exhausts_retries_and_raises_on_persistent_5xx():
    c = _Client([_Resp(500), _Resp(500), _Resp(500)])
    with pytest.raises(httpx.HTTPStatusError):
        _call_post(c, retries=3)
    assert c.calls == 3


def test_post_rejects_retries_below_one():
    # retries=0 would otherwise skip the loop and hit RuntimeError("unreachable").
    c = _Client([_Resp(200, {"ok": 1})])
    with pytest.raises(ValueError):
        _call_post(c, retries=0)
    assert c.calls == 0


# --------------------------------------------------------------------------
# _answer — must replicate the official reader settings for non-reasoning
# models (temperature=0, max_tokens=800) and leave reasoning models uncapped,
# or results drift from the official harness.
# --------------------------------------------------------------------------
def _capture_answer(model, monkeypatch):
    captured = {}

    async def fake_post(client, url, body, key, **kw):
        captured.update(body)
        return {"choices": [{"message": {"content": "ans"}}]}

    monkeypatch.setattr(R, "_post", fake_post)
    out = asyncio.run(R._answer(None, "q?", [], "2023-01-01", model, "k"))
    return out, captured


def test_answer_caps_non_reasoning_model(monkeypatch):
    out, body = _capture_answer("gpt-4o", monkeypatch)
    assert out == "ans"
    assert body["max_tokens"] == 800
    assert body["temperature"] == 0.0


def test_answer_leaves_reasoning_model_uncapped(monkeypatch):
    _, body = _capture_answer("gpt-5", monkeypatch)
    assert "max_tokens" not in body
    assert "temperature" not in body


# --------------------------------------------------------------------------
# _history_block — dated blocks, oldest first, with the epoch sentinel and
# missing dates rendered "unknown" (feeds date-diff questions).
# --------------------------------------------------------------------------
def test_history_block_orders_oldest_first():
    hits = [
        {"content_excerpt": "ZZNEW", "valid_at": "2023-05-02"},
        {"content_excerpt": "QQOLD", "valid_at": "2023-05-01"},
    ]
    out = R._history_block(hits)
    assert out.index("QQOLD") < out.index("ZZNEW")


def test_history_block_epoch_sentinel_is_undated():
    hits = [{"content_excerpt": "x", "valid_at": "1970-01-01T00:00:00"}]
    assert "Date: unknown" in R._history_block(hits)


def test_history_block_empty_is_placeholder():
    assert R._history_block([]) == "(no facts retrieved)"

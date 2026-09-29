"""LLM wrapper: caching, budget, structured output, backoff, daily quota."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from datasleuth.config import Settings
from datasleuth.llm.base import LLMBudgetExceeded, LLMQuotaExceeded, LLMRateLimitError
from datasleuth.llm.client import LLM, CallTracker
from datasleuth.llm.providers import FakeProvider


class Answer(BaseModel):
    value: int


def make_llm(tmp_path: Path, handler, **kw) -> tuple[LLM, FakeProvider]:
    settings = Settings(
        llm_provider="fake",
        cache_dir=tmp_path / "cache" / "llm",
        log_dir=tmp_path / "logs",
        llm_rpm=0,
        **kw,
    )
    provider = FakeProvider(handler)
    return LLM(settings, provider, sleep=lambda _s: None), provider


def test_cache_hit_avoids_provider_call(tmp_path: Path) -> None:
    llm, provider = make_llm(tmp_path, lambda s, u: "hello")
    assert llm.generate("sys", "usr", purpose="t") == "hello"
    assert llm.generate("sys", "usr", purpose="t") == "hello"
    assert len(provider.calls) == 1
    assert (tmp_path / "logs" / "llm_calls.jsonl").read_text().count("\n") == 2


def test_budget_enforced_even_for_cached_calls(tmp_path: Path) -> None:
    llm, _ = make_llm(tmp_path, lambda s, u: "x")
    tracker = CallTracker(max_calls=2)
    llm.generate("s", "a", purpose="t", tracker=tracker)
    llm.generate("s", "a", purpose="t", tracker=tracker)
    with pytest.raises(LLMBudgetExceeded):
        llm.generate("s", "a", purpose="t", tracker=tracker)


def test_structured_output_parses_fenced_json(tmp_path: Path) -> None:
    llm, _ = make_llm(tmp_path, lambda s, u: '```json\n{"value": 3}\n```')
    assert llm.generate_structured(Answer, "s", "u", purpose="t").value == 3


def test_structured_output_repairs_once(tmp_path: Path) -> None:
    replies = iter(["not json", '{"value": 7}'])
    llm, provider = make_llm(tmp_path, lambda s, u: next(replies))
    assert llm.generate_structured(Answer, "s", "u", purpose="t").value == 7
    assert len(provider.calls) == 2


def test_rate_limit_backoff_then_success(tmp_path: Path) -> None:
    llm, provider = make_llm(tmp_path, lambda s, u: "ok")
    real = provider.complete
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] < 3:
            raise LLMRateLimitError("429", retry_after_s=1)
        return real(*a, **k)

    provider.complete = flaky  # type: ignore[method-assign]
    assert llm.generate("s", "u", purpose="t") == "ok"
    assert state["n"] == 3


def test_daily_quota_stops_uncached_calls_only(tmp_path: Path) -> None:
    llm, _ = make_llm(tmp_path, lambda s, u: "x", llm_daily_limit=1)
    llm.generate("s", "a", purpose="t")
    llm.generate("s", "a", purpose="t")  # cached: free
    with pytest.raises(LLMQuotaExceeded):
        llm.generate("s", "b", purpose="t")


def test_daily_quota_is_metered_per_model(tmp_path: Path) -> None:
    from datasleuth.llm.client import DailyQuota

    q = DailyQuota(tmp_path / "usage.json", limit=1)
    q.consume("model-a")
    q.consume("model-b")  # a different model has its own budget
    with pytest.raises(LLMQuotaExceeded):
        q.consume("model-a")
    assert q.used_today("model-a") == 1 and q.used_today("model-c") == 0

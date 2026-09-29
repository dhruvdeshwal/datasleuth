"""Provider adapters against stubbed SDK clients (no network, no keys)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors

from datasleuth.config import Settings
from datasleuth.llm.base import LLMError, LLMRateLimitError, LLMTransientError
from datasleuth.llm.providers import GeminiProvider, OllamaProvider, build_provider


def gemini(monkeypatch: pytest.MonkeyPatch, generate: Any, **settings: Any) -> GeminiProvider:
    provider = GeminiProvider("gemini-2.5-flash", Settings(gemini_api_key="k", **settings))
    monkeypatch.setattr(provider._client.models, "generate_content", generate)
    return provider


def ok_response(text: str = '{"a": 1}') -> SimpleNamespace:
    usage = SimpleNamespace(prompt_token_count=11, candidates_token_count=5, thoughts_token_count=3)
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_gemini_success_maps_text_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def generate(model: str, contents: str, config: Any) -> SimpleNamespace:
        seen.update(model=model, contents=contents, config=config)
        return ok_response()

    raw = gemini(monkeypatch, generate, llm_thinking_budget=0).complete(
        "sys", "hello", json_mode=True, temperature=0.0
    )
    assert (raw.text, raw.prompt_tokens, raw.completion_tokens) == (
        '{"a": 1}',
        11,
        8,
    )  # thinking tokens counted
    cfg = seen["config"]
    assert seen["model"] == "gemini-2.5-flash" and seen["contents"] == "hello"
    assert cfg.system_instruction == "sys" and cfg.temperature == 0.0
    assert cfg.response_mime_type == "application/json" and cfg.thinking_config.thinking_budget == 0


def test_gemini_free_tier_rate_limit_parses_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "34s"}]}
    }

    def generate(**_k: Any) -> None:
        raise errors.APIError(429, body)

    with pytest.raises(LLMRateLimitError) as exc:
        gemini(monkeypatch, generate).complete("s", "u", json_mode=False, temperature=0.0)
    assert exc.value.retry_after_s == 34.0


@pytest.mark.parametrize(
    ("code", "expected"),
    [(503, LLMTransientError), (500, LLMTransientError), (400, LLMError), (403, LLMError)],
)
def test_gemini_error_mapping(
    monkeypatch: pytest.MonkeyPatch, code: int, expected: type[Exception]
) -> None:
    def generate(**_k: Any) -> None:
        raise errors.APIError(code, {"error": {"message": "boom"}})

    with pytest.raises(expected) as exc:
        gemini(monkeypatch, generate).complete("s", "u", json_mode=False, temperature=0.0)
    assert (exc.type is expected) or issubclass(exc.type, expected)
    assert not isinstance(exc.value, LLMRateLimitError)


def test_gemini_empty_response_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMTransientError):
        gemini(monkeypatch, lambda **_k: ok_response("")).complete(
            "s", "u", json_mode=True, temperature=0.0
        )


def test_missing_keys_give_actionable_errors() -> None:
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build_provider(Settings(llm_provider="gemini", gemini_api_key=None))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_provider(Settings(llm_provider="anthropic", anthropic_api_key=None))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_provider(Settings(llm_provider="openai", openai_api_key=None))
    assert build_provider(Settings(llm_provider="ollama", llm_model="llama3.1")).name == "ollama"


def test_gemini_api_key_env_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-alias")
    assert Settings().gemini_api_key == "from-google-alias"


def test_ollama_request_shape_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        captured.update(url=url, json=json)
        return httpx.Response(
            200,
            json={"message": {"content": '{"x": 2}'}, "prompt_eval_count": 7, "eval_count": 4},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    provider = OllamaProvider("llama3.1", Settings(ollama_host="http://box:11434/"))
    raw = provider.complete("sys", "usr", json_mode=True, temperature=0.0)
    assert (raw.text, raw.prompt_tokens, raw.completion_tokens) == ('{"x": 2}', 7, 4)
    assert captured["url"] == "http://box:11434/api/chat" and captured["json"]["format"] == "json"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}

    def down(*_a: Any, **_k: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", down)
    with pytest.raises(LLMTransientError, match="Cannot reach Ollama"):
        provider.complete("s", "u", json_mode=False, temperature=0.0)


def test_pricing_estimates() -> None:
    from datasleuth.llm.pricing import estimate_cost

    assert estimate_cost("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)
    assert estimate_cost("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(
        0.50
    )  # longest prefix wins
    assert estimate_cost("some-local-model", 10, 10) == 0.0


def test_gemini_hard_daily_quota_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasleuth.llm.base import LLMQuotaExceeded

    body = {
        "error": {
            "message": "Quota exceeded",
            "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
        }
    }

    def generate(**_k: Any) -> None:
        raise errors.APIError(429, body)

    with pytest.raises(LLMQuotaExceeded, match=r"gemini-2.5-flash"):
        gemini(monkeypatch, generate).complete("s", "u", json_mode=False, temperature=0.0)

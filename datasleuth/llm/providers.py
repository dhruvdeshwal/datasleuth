"""Concrete provider implementations (Gemini, Anthropic, OpenAI, Ollama, fake).

SDK imports are lazy so that only the selected provider's package is required.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import httpx

from datasleuth.config import Settings
from datasleuth.llm.base import (
    LLMError,
    LLMQuotaExceeded,
    LLMRateLimitError,
    LLMTransientError,
    Provider,
    RawCompletion,
)

_RETRY_DELAY_RE = re.compile(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _retry_hint(message: str) -> float | None:
    match = _RETRY_DELAY_RE.search(message)
    return float(match.group(1)) if match else None


class GeminiProvider(Provider):
    """Google Gemini via the ``google-genai`` SDK (free-tier friendly)."""

    name = "gemini"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model)
        from google import genai

        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (get a free key at aistudio.google.com).")
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": int(settings.llm_timeout_s * 1000)},
        )
        self._thinking_budget = settings.llm_thinking_budget

    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        from google.genai import errors, types

        cfg: dict[str, Any] = {"system_instruction": system, "temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if self._thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self._thinking_budget)
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**cfg)
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429 and "PerDay" in str(exc):
                raise LLMQuotaExceeded(
                    f"Daily free-tier quota for {self.model} is exhausted (it resets daily). "
                    "Re-run later (cached answers are reused) or use another model via "
                    "DATASLEUTH_LLM_MODEL / --model."
                ) from exc
            if code == 429:
                raise LLMRateLimitError(str(exc), _retry_hint(str(exc))) from exc
            if code in (500, 502, 503, 504):
                raise LLMTransientError(str(exc)) from exc
            raise LLMError(f"Gemini API error ({code}): {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMTransientError(str(exc)) from exc
        usage = resp.usage_metadata
        text = resp.text or ""
        if not text:
            raise LLMTransientError("Gemini returned an empty response (possibly safety-blocked).")
        return RawCompletion(
            text=text,
            prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
            completion_tokens=(
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            )
            if usage
            else 0,
        )


class AnthropicProvider(Provider):
    """Anthropic Claude via the official SDK."""

    name = "anthropic"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model)
        import anthropic

        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )

    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        a = self._anthropic
        if json_mode:
            system += "\n\nRespond with a single valid JSON object and nothing else."
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except a.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except (a.APIConnectionError, a.InternalServerError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except a.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        text = "".join(b.text for b in resp.content if b.type == "text")
        return RawCompletion(text, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAIProvider(Provider):
    """OpenAI chat completions."""

    name = "openai"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model)
        import openai

        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_s, max_retries=0
        )

    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        o = self._openai
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except o.RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except (o.APIConnectionError, o.InternalServerError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except o.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        usage = resp.usage
        return RawCompletion(
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class OllamaProvider(Provider):
    """Local Ollama server via its REST API."""

    name = "ollama"

    def __init__(self, model: str, settings: Settings) -> None:
        super().__init__(model)
        self._host = settings.ollama_host.rstrip("/")
        self._timeout = settings.llm_timeout_s

    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = httpx.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Cannot reach Ollama at {self._host}: {exc}") from exc
        data = resp.json()
        return RawCompletion(
            data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        )


class FakeProvider(Provider):
    """Deterministic provider for tests: ``handler(system, user) -> text``."""

    name = "fake"

    def __init__(self, handler: Callable[[str, str], str], model: str = "fake-model") -> None:
        super().__init__(model)
        self._handler = handler
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        self.calls.append((system, user))
        text = self._handler(system, user)
        return RawCompletion(text, max(1, len(system + user) // 4), max(1, len(text) // 4))


def build_provider(settings: Settings) -> Provider:
    """Instantiate the provider selected by ``settings.llm_provider``."""
    kind = settings.llm_provider
    if kind == "gemini":
        return GeminiProvider(settings.llm_model, settings)
    if kind == "anthropic":
        return AnthropicProvider(settings.llm_model, settings)
    if kind == "openai":
        return OpenAIProvider(settings.llm_model, settings)
    if kind == "ollama":
        return OllamaProvider(settings.llm_model, settings)
    raise LLMError("The 'fake' provider must be constructed explicitly in tests.")

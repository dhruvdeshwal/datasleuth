"""The single entry point agents use to call an LLM.

Adds, on top of a raw :class:`Provider`: disk caching keyed on the prompt hash,
requests-per-minute throttling, retry/backoff on rate limits, a persisted daily
call limit (free-tier friendly), a per-question call budget, structured-output
parsing with Pydantic, and a JSONL log of every call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from datasleuth.config import Settings
from datasleuth.llm.base import (
    LLMBudgetExceeded,
    LLMError,
    LLMOutputError,
    LLMQuotaExceeded,
    LLMRateLimitError,
    LLMTransientError,
    Provider,
)
from datasleuth.llm.pricing import estimate_cost
from datasleuth.llm.providers import build_provider
from datasleuth.telemetry import span

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_MAX_ATTEMPTS = 6
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMCallRecord(BaseModel):
    """Log entry for one logical LLM call."""

    purpose: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool
    cost_usd: float


class CallTracker:
    """Per-question accounting: enforces the call budget and aggregates usage."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.records: list[LLMCallRecord] = []

    @property
    def calls(self) -> int:
        """Number of logical calls made (cached calls count toward the budget)."""
        return len(self.records)

    def check(self) -> None:
        """Raise LLMBudgetExceeded if no calls remain."""
        if self.calls >= self.max_calls:
            raise LLMBudgetExceeded(f"LLM call budget of {self.max_calls} per question exhausted.")

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records)


class _RateLimiter:
    """Spaces requests so that at most ``rpm`` are issued per minute."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)


class DailyQuota:
    """Persisted count of uncached provider calls per calendar day *and model*.

    Free tiers meter each model separately, so the limit applies per model.
    """

    def __init__(self, path: Path, limit: int) -> None:
        self._path = path
        self._limit = limit
        self._lock = threading.Lock()

    def _read(self) -> dict[str, int]:
        try:
            return {str(k): int(v) for k, v in json.loads(self._path.read_text()).items()}
        except (OSError, ValueError):
            return {}

    def used_today(self, model: str) -> int:
        """Uncached calls made today with ``model``."""
        return self._read().get(f"{date.today().isoformat()}|{model}", 0)

    def consume(self, model: str) -> None:
        """Count one call, raising LLMQuotaExceeded once the limit is reached."""
        with self._lock:
            used = self._read()
            key = f"{date.today().isoformat()}|{model}"
            if self._limit and used.get(key, 0) >= self._limit:
                raise LLMQuotaExceeded(
                    f"Daily LLM call limit ({self._limit}) for {model} reached. "
                    "Re-run later; cached results are reused."
                )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({**used, key: used.get(key, 0) + 1}))


class LLM:
    """Cached, throttled, logged LLM wrapper used by every node."""

    def __init__(
        self,
        settings: Settings,
        provider: Provider | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.provider = provider or build_provider(settings)
        self._limiter = _RateLimiter(settings.llm_rpm)
        self._quota = DailyQuota(settings.cache_dir.parent / "usage.json", settings.llm_daily_limit)
        self._sleep = sleep
        self._log_path = settings.log_dir / "llm_calls.jsonl"

    # ------------------------------------------------------------------ cache
    def _cache_key(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        blob = json.dumps(
            [self.provider.name, self.provider.model, system, user, json_mode, temperature]
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.settings.cache_dir / key[:2] / f"{key}.json"

    # ------------------------------------------------------------------ calls
    def generate(
        self,
        system: str,
        user: str,
        *,
        purpose: str,
        tracker: CallTracker | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the completion text, using the cache when possible."""
        if tracker:
            tracker.check()
        temperature = self.settings.llm_temperature
        key = self._cache_key(system, user, json_mode, temperature)
        path = self._cache_path(key)
        started = time.perf_counter()
        cached = False
        if self.settings.use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            text, p_tok, c_tok = data["text"], data["prompt_tokens"], data["completion_tokens"]
            cached = True
        else:
            with span("datasleuth.llm", purpose=purpose, model=self.provider.model):
                text, p_tok, c_tok = self._call_with_retries(system, user, json_mode, temperature)
            if self.settings.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"text": text, "prompt_tokens": p_tok, "completion_tokens": c_tok}),
                    encoding="utf-8",
                )
        latency = (time.perf_counter() - started) * 1000
        record = LLMCallRecord(
            purpose=purpose,
            model=self.provider.model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=latency,
            cached=cached,
            cost_usd=estimate_cost(self.provider.model, p_tok, c_tok),
        )
        if tracker:
            tracker.records.append(record)
        self._log(record, system, user, text)
        return str(text)

    def generate_structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        purpose: str,
        tracker: CallTracker | None = None,
    ) -> T:
        """Call the model and parse its JSON reply into ``schema`` (one repair attempt)."""
        json_schema = json.dumps(schema.model_json_schema(), indent=None)
        sys_prompt = (
            f"{system}\n\nReply with ONLY one JSON object matching this JSON Schema "
            f"(no prose, no markdown fences):\n{json_schema}"
        )
        text = self.generate(sys_prompt, user, purpose=purpose, tracker=tracker, json_mode=True)
        try:
            return schema.model_validate_json(_strip_fences(text))
        except ValidationError as exc:
            repair = (
                f"{user}\n\nYour previous reply was not valid: {str(exc)[:600]}\n"
                f"Previous reply:\n{text[:1500]}\nReturn a corrected JSON object."
            )
            text = self.generate(
                sys_prompt, repair, purpose=f"{purpose}:repair", tracker=tracker, json_mode=True
            )
            try:
                return schema.model_validate_json(_strip_fences(text))
            except ValidationError as exc2:
                raise LLMOutputError(f"Model returned invalid structured output: {exc2}") from exc2

    # --------------------------------------------------------------- internals
    def _call_with_retries(
        self, system: str, user: str, json_mode: bool, temperature: float
    ) -> tuple[str, int, int]:
        self._quota.consume(self.provider.model)
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._limiter.wait()
            try:
                raw = self.provider.complete(
                    system, user, json_mode=json_mode, temperature=temperature
                )
                return raw.text, raw.prompt_tokens, raw.completion_tokens
            except (LLMRateLimitError, LLMTransientError) as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise
                hint = exc.retry_after_s if isinstance(exc, LLMRateLimitError) else None
                delay = min(self.settings.llm_max_backoff_s, hint + 1 if hint else 2**attempt)
                log.warning(
                    "LLM %s (attempt %d); retrying in %.1fs", type(exc).__name__, attempt, delay
                )
                self._sleep(delay)
        raise LLMError("unreachable")  # pragma: no cover

    def _log(self, record: LLMCallRecord, system: str, user: str, response: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                **record.model_dump(),
                "system": system,
                "prompt": user,
                "response": response,
            }
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:  # logging must never break a run
            log.warning("could not write LLM call log", exc_info=True)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())

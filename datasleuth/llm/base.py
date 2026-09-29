"""Provider-agnostic LLM interfaces and errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMOutputError(LLMError):
    """The model replied, but not with valid structured output."""


class LLMRateLimitError(LLMError):
    """Provider signalled rate limiting; ``retry_after_s`` is a hint when known."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTransientError(LLMError):
    """Temporary server-side failure worth retrying."""


class LLMQuotaExceeded(LLMError):
    """The configured daily call limit has been reached."""


class LLMBudgetExceeded(LLMError):
    """The per-question LLM call budget has been exhausted."""


@dataclass(frozen=True)
class RawCompletion:
    """What a provider returns for a single call."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Provider(ABC):
    """A concrete LLM backend. Only code under ``llm/`` may talk to a vendor SDK."""

    name: str

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def complete(
        self, system: str, user: str, *, json_mode: bool, temperature: float
    ) -> RawCompletion:
        """Run one completion. Raise LLMRateLimitError/LLMTransientError/LLMError."""

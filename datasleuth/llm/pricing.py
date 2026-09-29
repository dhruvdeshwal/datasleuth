"""Approximate list prices (USD per 1M tokens) used for *estimated* cost reporting.

Free-tier usage costs nothing, but the estimate uses paid-tier list prices so runs
are comparable. Prices change; override via ``PRICES`` if you need exact numbers.
"""

from __future__ import annotations

# model-name prefix -> (input $/1M, output $/1M)
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a call (0.0 for unknown/local models)."""
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            p_in, p_out = PRICES[prefix]
            return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000
    return 0.0

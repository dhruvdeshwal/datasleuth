"""Helpers to turn raw driver output into capped, JSON-safe ExecutionResults."""

from __future__ import annotations

import datetime as dt
import decimal
import math
from typing import Any

from datasleuth.state import ErrorType

# (substring in lowercase message, error type). First match wins.
_SQL_ERROR_PATTERNS: list[tuple[str, ErrorType]] = [
    ("no such column", "unknown_column"),
    ("unknown column", "unknown_column"),
    ("has no column named", "unknown_column"),
    ("no such table", "unknown_table"),
    ("no such function", "syntax_error"),
    ("ambiguous column", "ambiguous_column"),
    ("is ambiguous", "ambiguous_column"),
    ("interrupted", "timeout"),
    ("statement timeout", "timeout"),
    ("max_execution_time", "timeout"),
    ("datatype mismatch", "type_mismatch"),
    ("invalid input syntax for type", "type_mismatch"),
    ("operator does not exist", "type_mismatch"),
    ("cannot be cast", "type_mismatch"),
    ("cannot join", "join_error"),
    ("on clause", "join_error"),
    ("syntax error", "syntax_error"),
    ("incomplete input", "syntax_error"),
    ("misuse of aggregate", "syntax_error"),
    ("misuse of window", "syntax_error"),
    ("you have an error in your sql syntax", "syntax_error"),
]


def classify_sql_error(message: str) -> ErrorType:
    """Map a database error message to the shared error taxonomy."""
    low = message.lower()
    if "column" in low and "does not exist" in low:
        return "unknown_column"
    if ("relation" in low or "table" in low) and (
        "does not exist" in low or "doesn't exist" in low
    ):
        return "unknown_table"
    for needle, kind in _SQL_ERROR_PATTERNS:
        if needle in low:
            return kind
    return "runtime_error"


def jsonable(value: Any, cell_cap: int) -> Any:
    """Convert a cell to a JSON-safe value, truncating long text."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<binary {len(bytes(value))} bytes>"
    text = str(value)
    return text if len(text) <= cell_cap else text[: cell_cap - 1] + "…"

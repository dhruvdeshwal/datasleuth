"""Standalone worker that runs validated Pandas code. Executed in a subprocess or container.

Protocol: one JSON object on stdin
``{"db_path", "code", "tables", "row_cap", "cell_cap", "memory_mb", "cpu_s", "immutable"}``;
one JSON object on stdout ``{"ok", "columns", "rows", "truncated", "error_type", "error_message"}``.

This file deliberately imports nothing from ``datasleuth`` so it can be mounted alone
into the Docker sandbox image.
"""

from __future__ import annotations

import builtins
import contextlib
import decimal
import io
import json
import math
import sqlite3
import sys
import traceback
from typing import Any

_SAFE_BUILTIN_NAMES = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "print",
    "chr",
    "ord",
    "bytes",
    "frozenset",
    "iter",
    "next",
    "hash",
    "callable",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "Exception",
    "IndexError",
    "KeyError",
    "LookupError",
    "NameError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "True",
    "False",
    "None",
]
_ALLOWED_IMPORTS = {"pandas", "numpy", "math"}


def _limit_resources(memory_mb: int, cpu_s: int) -> None:
    if sys.platform == "win32":  # the parent process enforces memory/time there
        return
    import resource

    limit = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _block_network() -> None:
    import socket

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("network access is disabled in the sandbox")

    socket.socket = refuse  # type: ignore[misc,assignment]
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def _safe_import(
    name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0
) -> Any:
    if level or name.split(".")[0] not in _ALLOWED_IMPORTS or "." in name:
        raise ImportError(f"import of '{name}' is not allowed")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _jsonable(value: Any, cap: int) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return None if str(value) == "NaT" else value.isoformat()
    if isinstance(value, bytes | bytearray):
        return f"<binary {len(value)} bytes>"
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item(), cap)
    text = str(value)
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _classify(exc: BaseException) -> str:
    if isinstance(exc, KeyError):
        return "unknown_column"
    if isinstance(exc, NameError):
        return "unknown_table"
    if isinstance(exc, TypeError | ValueError | OverflowError):
        return "type_mismatch"
    if isinstance(exc, SyntaxError):
        return "syntax_error"
    if isinstance(exc, MemoryError):
        return "runtime_error"
    return "runtime_error"


def _describe(exc: BaseException) -> str:
    line = ""
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<agent>":
            line = f" (line {frame.lineno})"
    return f"{type(exc).__name__}: {str(exc)[:400]}{line}"


def _to_frame(pd: Any, value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.rename(name).reset_index()
    if isinstance(value, dict):
        return pd.DataFrame(list(value.items()), columns=["key", "value"])
    if isinstance(value, list | tuple):
        return pd.DataFrame(value)
    return pd.DataFrame({"result": [value]})


def main() -> None:
    payload = json.loads(sys.stdin.read())
    real_stdout = sys.stdout
    out: dict[str, Any]
    try:
        _limit_resources(payload["memory_mb"], payload["cpu_s"])
        _block_network()
        import numpy as np
        import pandas as pd

        suffix = "?mode=ro&immutable=1" if payload.get("immutable") else "?mode=ro"
        conn = sqlite3.connect(f"file:{payload['db_path']}{suffix}", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        wanted = payload.get("tables")
        names = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        ]
        dfs = {
            n: pd.read_sql_query(f'SELECT * FROM "{n}"', conn)
            for n in names
            if not n.startswith("sqlite_") and (not wanted or n in wanted)
        }
        conn.close()
        env: dict[str, Any] = {
            "__builtins__": {
                **{n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)},
                "__import__": _safe_import,
            },
            "pd": pd, "np": np, "math": __import__("math"), "dfs": dfs,
        }  # fmt: skip
        env.update({n: df for n, df in dfs.items() if n.isidentifier() and n not in env})
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(compile(payload["code"], "<agent>", "exec"), env)
            if "result" not in env:
                raise NameError("the code did not assign a variable named `result`")
            frame = _to_frame(pd, env["result"])
            cap = payload["row_cap"]
            head = frame.head(cap + 1)
            cell = payload["cell_cap"]
            rows = [
                [_jsonable(v, cell) for v in row] for row in head.itertuples(index=False, name=None)
            ]
            out = {
                "ok": True,
                "columns": [str(c) for c in frame.columns],
                "rows": rows[:cap],
                "truncated": len(head) > cap,
            }
        except Exception as exc:
            out = {"ok": False, "error_type": _classify(exc), "error_message": _describe(exc)}
    except MemoryError:
        out = {"ok": False, "error_type": "runtime_error", "error_message": "memory limit exceeded"}
    except Exception as exc:
        out = {
            "ok": False,
            "error_type": "runtime_error",
            "error_message": f"sandbox failure: {exc}",
        }
    real_stdout.write(json.dumps(out))
    real_stdout.flush()


if __name__ == "__main__":
    main()

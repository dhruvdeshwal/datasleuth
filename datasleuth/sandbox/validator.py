"""Static validation of generated code. No LLM involved.

* SQL: exactly one read-only query (SELECT / CTE / set operation), checked on the
  sqlglot AST, with denylists for side-effecting nodes, file/network functions and
  system schemas.
* Pandas: an AST allowlist (pandas/numpy/math imports only, no dunder access, no
  file/IO/eval/exec primitives).
"""

from __future__ import annotations

import ast
from typing import Any

import sqlglot
from pydantic import BaseModel
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from datasleuth.state import Dialect, ErrorType


class ValidationResult(BaseModel):
    """Outcome of static validation."""

    ok: bool
    error_type: ErrorType | None = None
    message: str | None = None


_OK = ValidationResult(ok=True)


def _block(message: str, error_type: ErrorType = "validation_error") -> ValidationResult:
    return ValidationResult(ok=False, error_type=error_type, message=message)


# ------------------------------------------------------------------------------ SQL
_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Create", "Drop", "Alter", "AlterTable", "Command", "Pragma",
    "Attach", "Detach", "Copy", "Merge", "Into", "Lock", "Set", "Use", "TruncateTable",
    "Transaction", "Commit", "Rollback", "Grant", "Revoke", "LoadData", "Analyze", "Kill",
    "Refresh", "Declare", "Execute",
)  # fmt: skip
_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

_FORBIDDEN_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile",
    "randomblob", "zeroblob", "read_csv", "read_csv_auto", "read_parquet", "read_json",
    "read_json_auto", "read_text", "read_blob", "read_ndjson", "glob", "parquet_scan",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import",
    "lo_export", "lo_get", "dblink", "dblink_exec", "pg_sleep", "sleep", "benchmark",
    "load_file", "set_config", "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "query_to_xml", "database_to_xml", "xp_cmdshell", "sys_exec",
    "sys_eval", "current_setting", "pg_advisory_lock", "get_lock",
}  # fmt: skip
_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "mysql", "performance_schema", "sys"}


def _function_name(node: Any) -> str | None:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None


def validate_sql(sql: str, dialect: Dialect = "sqlite") -> ValidationResult:
    """Allow a single read-only query; block everything else with a clear reason."""
    if not sql or not sql.strip():
        return _block("The query is empty.", "syntax_error")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, TokenError) as exc:
        return _block(f"SQL could not be parsed: {str(exc).splitlines()[0][:300]}", "syntax_error")
    if not statements:
        return _block("The query is empty.", "syntax_error")
    if len(statements) > 1:
        return _block("Only a single SQL statement is allowed; multiple statements were found.")
    root = statements[0]
    if not isinstance(root, exp.Query):
        return _block(
            f"Only read-only SELECT queries are allowed (got {type(root).__name__.upper()})."
        )
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return _block(f"{type(node).__name__.upper()} is not allowed in read-only queries.")
        name = _function_name(node)
        if name in _FORBIDDEN_FUNCTIONS:
            return _block(f"Function '{name}' is not allowed (file/network/system access).")
        if (
            isinstance(node, exp.Table)
            and str(node.args.get("db") or "").lower() in _SYSTEM_SCHEMAS
        ):
            return _block("Querying database system schemas is not allowed.")
        if isinstance(node, exp.Table) and isinstance(node.this, exp.Func):
            fname = _function_name(node.this)
            if fname and fname not in {"generate_series", "unnest", "json_each", "range"}:
                return _block(f"Table function '{fname}' is not allowed.")
    return _OK


# --------------------------------------------------------------------------- Pandas
_ALLOWED_MODULES = {"pandas", "numpy", "math"}
_FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
    "dir", "type", "super", "classmethod", "staticmethod", "property", "__builtins__",
}  # fmt: skip
_FORBIDDEN_ATTRS = {
    "os", "sys", "subprocess", "builtins", "importlib", "ctypes", "io", "compat", "util",
    "testing", "system", "popen", "eval", "query", "format_map", "load", "loads", "save",
    "savez", "savez_compressed", "savetxt", "fromfile", "tofile", "loadtxt", "genfromtxt",
    "memmap", "ctypeslib", "fromregex", "f2py", "distutils", "lib", "core", "to_csv",
    "to_pickle", "to_excel", "to_parquet", "to_feather", "to_hdf", "to_sql", "to_json",
    "to_html", "to_latex", "to_stata", "to_xml", "to_clipboard", "to_markdown", "to_orc",
    "to_xarray", "plotting", "api", "arrays", "errors",
}  # fmt: skip


class _PandasChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.problems: list[str] = []

    def _flag(self, node: ast.AST, msg: str) -> None:
        self.problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] not in _ALLOWED_MODULES or "." in alias.name:
                self._flag(
                    node, f"import of '{alias.name}' is not allowed (pandas/numpy/math only)"
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module not in _ALLOWED_MODULES:
            self._flag(node, f"import from '{module}' is not allowed (pandas/numpy/math only)")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
            self._flag(node, f"use of '{node.id}' is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") or node.attr.startswith("_"):
            self._flag(node, f"access to private attribute '{node.attr}' is not allowed")
        elif node.attr in _FORBIDDEN_ATTRS or node.attr.startswith("read_"):
            self._flag(node, f"access to '.{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "__" in node.value:
            self._flag(node, "strings containing '__' are not allowed")

    def visit_Global(self, node: ast.Global) -> None:
        self._flag(node, "'global' is not allowed")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._flag(node, "class definitions are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._flag(node, "async code is not allowed")


def validate_pandas(code: str) -> ValidationResult:
    """Check Pandas code against the AST allowlist."""
    if not code or not code.strip():
        return _block("The code is empty.", "syntax_error")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _block(f"Python syntax error: {exc.msg} (line {exc.lineno})", "syntax_error")
    checker = _PandasChecker()
    checker.visit(tree)
    if checker.problems:
        return _block("Blocked by the sandbox policy: " + "; ".join(checker.problems[:5]))
    assigns_result = any(
        isinstance(n, ast.Name) and n.id == "result" and isinstance(n.ctx, ast.Store)
        for n in ast.walk(tree)
    )
    if not assigns_result:
        return _block("The code must assign its final answer to a variable named `result`.")
    return _OK

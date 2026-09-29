"""Sandbox facade: validate, then execute, returning a uniform ExecutionResult."""

from __future__ import annotations

from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.sandbox.pandas_runner import run_pandas_docker, run_pandas_subprocess
from datasleuth.sandbox.sql_executor import run_engine, run_sqlite
from datasleuth.sandbox.validator import ValidationResult, validate_pandas, validate_sql
from datasleuth.state import ExecutionResult, Language


class Sandbox:
    """Runs generated code against one DataSource under the configured isolation."""

    def __init__(self, source: DataSource, settings: Settings) -> None:
        self.source = source
        self.settings = settings

    def validate(self, language: Language, code: str) -> ValidationResult:
        """Static validation only (used by the Validator node)."""
        if language == "sql":
            return validate_sql(code, self.source.dialect)
        return validate_pandas(code)

    def execute(
        self, language: Language, code: str, tables: list[str] | None = None
    ) -> ExecutionResult:
        """Validate then execute. Blocked code never reaches an interpreter or database."""
        verdict = self.validate(language, code)
        if not verdict.ok:
            return ExecutionResult(
                ok=False, error_type=verdict.error_type, error_message=verdict.message
            )
        return self.execute_validated(language, code, tables)

    def execute_validated(
        self, language: Language, code: str, tables: list[str] | None = None
    ) -> ExecutionResult:
        """Execute code that already passed :meth:`validate`."""
        src = self.source
        if language == "sql":
            sql = code.strip().rstrip(";")
            if src.db_path is not None:
                return run_sqlite(src.db_path, sql, self.settings)
            assert src.engine is not None
            return run_engine(src.engine, src.dialect, sql, self.settings)
        if src.db_path is None:
            return ExecutionResult(
                ok=False, error_type="validation_error",
                error_message="Pandas is only available for uploaded files and SQLite databases; use SQL.",
            )  # fmt: skip
        runner = run_pandas_docker if self.settings.sandbox == "docker" else run_pandas_subprocess
        return runner(src.db_path, code, tables, self.settings)

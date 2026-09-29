"""A DataSource describes where a session's data lives and how it may be queried."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from datasleuth.state import Dialect


@dataclass
class DataSource:
    """Either a local SQLite file (uploads, SQLite databases) or an external engine."""

    dialect: Dialect
    db_path: Path | None = None
    engine: Engine | None = None

    def __post_init__(self) -> None:
        if (self.db_path is None) == (self.engine is None):
            raise ValueError("DataSource needs exactly one of db_path or engine.")

    @property
    def supports_pandas(self) -> bool:
        """Pandas execution loads tables from a local SQLite file only."""
        return self.db_path is not None

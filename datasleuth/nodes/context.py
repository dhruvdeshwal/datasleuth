"""Per-question dependencies shared by all nodes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasleuth.config import Settings
from datasleuth.db.source import DataSource
from datasleuth.fewshot import FewShotBank
from datasleuth.llm.client import LLM, CallTracker
from datasleuth.sandbox.executor import Sandbox
from datasleuth.state import SchemaInfo

Event = dict[str, Any]


@dataclass
class RunContext:
    """Everything a node needs besides the graph state. One instance per question."""

    llm: LLM
    settings: Settings
    source: DataSource
    sandbox: Sandbox
    tracker: CallTracker
    schema: SchemaInfo | None = None
    few_shot: FewShotBank | None = None
    artifacts_dir: Path | None = None
    started: float = field(default_factory=time.perf_counter)
    emit: Callable[[Event], None] = field(default=lambda _event: None)

    @property
    def pandas_allowed(self) -> bool:
        """Pandas is available when enabled and the source is a local SQLite file."""
        return self.settings.allow_pandas and self.source.supports_pandas

"""Deterministic backend for the Playwright E2E test (no API key, scripted fake LLM).

Serves the built frontend and the real API. The scripted "model" first writes a query with a
typo'd column, then repairs it when the critic's instruction arrives, so the E2E test also
exercises the visible self-correction path.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import uvicorn

from datasleuth.api.app import create_app
from datasleuth.config import Settings
from datasleuth.llm.client import LLM
from datasleuth.llm.providers import FakeProvider

ROOT = Path(__file__).resolve().parent.parent
BAD = "SELECT region, SUM(amout) AS total FROM sample GROUP BY region ORDER BY region"
GOOD = "SELECT region, SUM(amount) AS total FROM sample GROUP BY region ORDER BY region"


def handler(system: str, user: str) -> str:
    """Pretend to be every LLM-backed node."""
    if "planning stage" in system:
        return json.dumps(
            {
                "status": "answerable",
                "question_types": ["aggregation", "comparison"],
                "needs_chart": True,
                "expected_columns": 2,
                "expects_many_rows": True,
                "steps": "Sum the amount for each region.",
            }
        )
    if "careful data analyst" in system:
        code = GOOD if "REPAIR INSTRUCTION" in user else BAD
        return json.dumps({"language": "sql", "code": code, "rationale": "Total per region."})
    if "chart-selection" in system:
        spec = {"chart_type": "bar", "x": "region", "y": "total", "title": "Total by region"}
        return json.dumps({"use_chart": True, "spec": spec, "reason": "compare categories"})
    return json.dumps(
        {"answer": "East has the highest total at 300, then west at 200 and north at 150."}
    )


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    tmp = Path(tempfile.mkdtemp(prefix="ds_e2e_"))
    settings = Settings(
        llm_provider="fake", data_dir=tmp, cache_dir=tmp / "c", log_dir=tmp / "l", llm_rpm=0,
        use_cache=False, schema_linker_min_tables=99, frontend_dist=ROOT / "frontend" / "dist",
    )  # fmt: skip
    llm = LLM(settings, FakeProvider(handler), sleep=lambda _s: None)
    uvicorn.run(create_app(settings, llm=llm), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

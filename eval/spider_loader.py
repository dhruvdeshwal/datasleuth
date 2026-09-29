"""Spider 1.0 loader. Dev-only code: nothing in ``datasleuth/`` may import this module.

Difficulty labels (easy/medium/hard/extra) come from the *official* evaluator
(``test-suite-sql-eval/evaluation.py``) applied to Spider's parsed ``sql`` field, so
they match the numbers reported in the literature.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).parent / "spider"
DIFFICULTIES = ("easy", "medium", "hard", "extra")


@dataclass(frozen=True)
class SpiderExample:
    """One Spider question."""

    qid: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str


class SpiderNotFound(FileNotFoundError):
    """Spider has not been downloaded yet."""


class SpiderData:
    """Locations and accessors for a downloaded Spider tree."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.environ.get("SPIDER_DIR") or DEFAULT_ROOT)
        self.data = self.root / "spider_data"
        self.official = self.root / "test-suite-sql-eval"
        if not (self.data / "dev.json").exists():
            raise SpiderNotFound(
                f"Spider not found under {self.root}. Run: bash scripts/download_spider.sh"
            )

    # ------------------------------------------------------------------ paths
    def db_path(self, db_id: str) -> Path:
        """Original single-instance SQLite database used for plain execution accuracy."""
        return self.data / "database" / db_id / f"{db_id}.sqlite"

    def test_suite_dir(self) -> Path | None:
        """Directory of test-suite database variants (optional download)."""
        path = self.root / "testsuite_databases"
        return path if path.exists() else None

    # --------------------------------------------------------------- examples
    def _official_evaluator(self) -> Any:
        if not self.official.exists():
            raise SpiderNotFound("Official evaluator missing; run scripts/download_spider.sh")
        if str(self.official) not in sys.path:
            sys.path.append(str(self.official))
        import evaluation

        return evaluation.Evaluator()

    def dev(self) -> list[SpiderExample]:
        """All 1,034 dev questions with official difficulty labels."""
        raw = json.loads((self.data / "dev.json").read_text(encoding="utf-8"))
        evaluator = self._official_evaluator()
        return [
            SpiderExample(
                qid=f"dev-{i:04d}",
                db_id=item["db_id"],
                question=item["question"],
                gold_sql=item["query"],
                difficulty=evaluator.eval_hardness(item["sql"]),
            )
            for i, item in enumerate(raw)
        ]

    def train_pairs(self) -> list[tuple[str, str, str]]:
        """(db_id, question, sql) from Spider *train* only: the leakage-safe few-shot pool."""
        raw = json.loads((self.data / "train_spider.json").read_text(encoding="utf-8"))
        return [(r["db_id"], r["question"], r["query"]) for r in raw]


def stratified_sample(examples: list[SpiderExample], limit: int, seed: int) -> list[SpiderExample]:
    """Pick ``limit`` examples, proportional per difficulty (largest remainder), seeded."""
    if limit >= len(examples):
        return list(examples)
    strata: dict[str, list[SpiderExample]] = defaultdict(list)
    for ex in examples:
        strata[ex.difficulty].append(ex)
    total = len(examples)
    quotas = {k: limit * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    leftovers = sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)
    for k in leftovers[: limit - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen: list[SpiderExample] = []
    for key in sorted(strata):
        pool = sorted(strata[key], key=lambda e: e.qid)
        rng.shuffle(pool)
        chosen.extend(pool[: alloc[key]])
    return sorted(chosen, key=lambda e: e.qid)

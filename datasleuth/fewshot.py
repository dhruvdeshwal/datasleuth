"""Generic few-shot example bank with deterministic BM25 retrieval.

The bank is just ``{question, sql}`` pairs and is entirely optional: the agent works
with zero examples on any schema. Examples come from *other* databases, so they teach
query style, not table names.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
    "what", "which", "who", "how", "many", "much", "that", "with", "by", "at", "from", "as",
    "be", "me", "show", "list", "give", "tell", "all", "each", "do", "does", "have", "has",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords, with naive plural stripping."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok)
    return out


class FewShotExample(BaseModel):
    """One question → SQL pair."""

    question: str
    sql: str


class FewShotBank:
    """BM25 retrieval over example questions."""

    def __init__(self, examples: list[FewShotExample], k1: float = 1.5, b: float = 0.75) -> None:
        self.examples = examples
        self._k1, self._b = k1, b
        self._docs = [Counter(tokenize(e.question)) for e in examples]
        self._lens = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(examples)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> FewShotBank:
        """Load a bank from a JSONL file of ``{"question": ..., "sql": ...}`` lines."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([FewShotExample.model_validate_json(x) for x in lines if x.strip()])

    def save_jsonl(self, path: str | Path) -> None:
        """Write the bank as JSONL."""
        Path(path).write_text(
            "\n".join(json.dumps(e.model_dump()) for e in self.examples) + "\n", encoding="utf-8"
        )

    def retrieve(self, question: str, k: int) -> list[str]:
        """Top-``k`` examples formatted for the prompt (never the identical question)."""
        query = tokenize(question)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            if self.examples[i].question.strip().lower() == question.strip().lower():
                continue
            norm = 1 - self._b + self._b * (self._lens[i] / self._avg if self._avg else 1)
            score = sum(
                self._idf.get(t, 0.0) * doc[t] * (self._k1 + 1) / (doc[t] + self._k1 * norm)
                for t in set(query)
                if t in doc
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [
            f"Question: {self.examples[i].question}\nSQL: {self.examples[i].sql}"
            for _, i in scored[:k]
        ]

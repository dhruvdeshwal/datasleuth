"""Build the few-shot bank from Spider *train* only (leakage guard).

Dev questions/databases are never read here. Spider's train and dev sets use disjoint
databases, and ``FewShotBank.retrieve`` additionally never returns an identical question.
"""

from __future__ import annotations

from pathlib import Path

from datasleuth.fewshot import FewShotBank, FewShotExample
from eval.spider_loader import SpiderData


def build_bank(root: Path | None = None) -> FewShotBank:
    """All 7,000 train_spider question→SQL pairs as a retrieval bank."""
    pairs = SpiderData(root).train_pairs()
    return FewShotBank([FewShotExample(question=q, sql=s) for _, q, s in pairs])


def load_bank(root: Path | None = None) -> FewShotBank:
    """Load (or build and cache) the train-only bank."""
    data = SpiderData(root)
    path = data.root / "fewshot_train.jsonl"
    if path.exists():
        return FewShotBank.from_jsonl(path)
    bank = build_bank(root)
    bank.save_jsonl(path)
    return bank

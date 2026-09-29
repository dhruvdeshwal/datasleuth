"""Build cases + system from an EvalConfig and run it (used by ``datasleuth eval``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasleuth.config import Settings
from datasleuth.llm.client import LLM
from datasleuth.nodes.query_writer import WriterOptions
from eval.runner import EvalConfig, make_ts_scorer, run_eval
from eval.spider_loader import SpiderData, stratified_sample
from eval.systems import Case, SingleShotSystem, System


def spider_cases(limit: int | None, seed: int, root: Path | None = None) -> list[Case]:
    """Stratified (by difficulty) sample of Spider dev as evaluation cases."""
    data = SpiderData(root)
    examples = data.dev()
    if limit:
        examples = stratified_sample(examples, limit, seed)
    return [
        Case(e.qid, data.db_path(e.db_id), e.question, e.gold_sql, e.difficulty, group=e.db_id)
        for e in examples
    ]


def custom_cases(limit: int | None) -> list[Case]:
    """Cases from the hand-made messy-CSV suite; a ``limit`` round-robins across datasets."""
    from eval.custom_suite.build import load_cases

    cases = load_cases()
    if not limit or limit >= len(cases):
        return cases
    by_group: dict[str, list[Case]] = {}
    for c in cases:
        by_group.setdefault(c.group, []).append(c)
    picked: list[Case] = []
    for rank in range(max(len(v) for v in by_group.values())):
        picked += [v[rank] for v in by_group.values() if rank < len(v)]
    return sorted(picked[:limit], key=lambda c: c.qid)


def system_factory(config: EvalConfig) -> Callable[[LLM, Settings], System]:
    """Return a constructor for the configured system."""
    if config.system == "single_shot":
        opts = WriterOptions(
            include_samples=config.include_samples,
            minimal_schema=config.minimal_schema,
            allow_pandas=False,
            force_language="sql",
        )
        return lambda llm, settings: SingleShotSystem(llm, settings, opts)
    from eval.agent_system import AgentSystem

    def build(llm: LLM, settings: Settings) -> System:
        bank = None
        if settings.use_few_shot:
            from eval.fewshot_build import load_bank

            bank = load_bank()
        return AgentSystem(llm, settings, bank)

    return build


def run_config(
    config_path: str | Path,
    *,
    limit: int | None = None,
    seed: int | None = None,
    ts: bool | None = None,
    spider_root: Path | None = None,
    settings_overrides: dict[str, Any] | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Load a YAML config and run it end to end; returns the summary dict."""
    config = EvalConfig.load(config_path)
    if limit is not None:
        config.limit = limit
    if seed is not None:
        config.seed = seed
    if ts is not None:
        config.ts_eval = ts
    settings = config.make_settings(**(settings_overrides or {}))
    if config.suite == "spider":
        cases = spider_cases(config.limit, config.seed, spider_root)
    elif config.suite == "business":
        from eval.business_suite.build import load_cases

        cases = load_cases()[: config.limit or None]
    else:
        cases = custom_cases(config.limit)
    scorer = make_ts_scorer(spider_root) if config.ts_eval and config.suite == "spider" else None
    return run_eval(
        config, cases, system_factory(config), settings=settings, ts_scorer=scorer, out=out
    )

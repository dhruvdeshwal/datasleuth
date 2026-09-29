"""The Spider baselines/ablation ladder, and a runner that executes them on the same subset.

Ladder (each step adds one thing, same questions/seed/temperature 0, cached LLM calls):

1. ``zero_shot``    single call, names/types/keys only
2. ``full_schema``  + profiled schema with sample values
3. ``linker``       + Schema Linker (agent graph, no retries)
4. ``retry_r1..r3`` + execution-error self-correction with 1/2/3 retries
5. ``few_shot``     + few-shot retrieval (Spider *train* only)
6. ``critic``       + semantic Critic checks on suspicious results
7. ``full``         + Router/Planner and SQL→Pandas escalation

Because the first attempt of steps 3-4 uses an identical prompt, the disk cache makes the
extra retry configs cost only the *additional* calls. On a free API tier use a small
``--limit`` and re-run: finished answers are cached and the run resumes.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from eval.metrics import mcnemar
from eval.run_config import run_config
from eval.runner import RESULTS_DIR, EvalConfig

# Settings shared by every agent-based rung (SQL-only, no analyst/chart: the benchmark
# only needs the executed rows).
_AGENT_BASE: dict[str, Any] = {
    "use_router": False,
    "use_schema_linker": True,
    "schema_linker_min_tables": 1,
    "use_few_shot": False,
    "use_critic_semantic": False,
    "allow_pandas": False,
    "use_analyst": False,
    "use_chart": False,
    "max_retries": 0,
    "max_llm_calls": 12,
}


def ladder(model: str, limit: int | None, seed: int = 0) -> list[EvalConfig]:
    """The ordered ablation configs for one model."""

    def cfg(name: str, description: str, system: str, **settings: Any) -> EvalConfig:
        extra: dict[str, Any] = {}
        if system == "single_shot":
            extra = {"minimal_schema": name == "zero_shot", "include_samples": name != "zero_shot"}
        return EvalConfig(
            name=name, description=description, system=system,  # type: ignore[arg-type]
            limit=limit, seed=seed, settings={"llm_model": model, **settings}, **extra,
        )  # fmt: skip

    def agent(**over: Any) -> dict[str, Any]:
        return {**_AGENT_BASE, **over}

    return [
        cfg("zero_shot", "single call, names/types/keys only", "single_shot"),
        cfg("full_schema", "+ profiled schema with sample values", "single_shot"),
        cfg("linker", "+ schema linker", "agent", **agent()),
        cfg("retry_r1", "+ self-correction, 1 retry", "agent", **agent(max_retries=1)),
        cfg("retry_r2", "+ self-correction, 2 retries", "agent", **agent(max_retries=2)),
        cfg("retry_r3", "+ self-correction, 3 retries", "agent", **agent(max_retries=3)),
        cfg("few_shot", "+ few-shot retrieval (train only)", "agent", **agent(max_retries=3, use_few_shot=True)),
        cfg("critic", "+ semantic critic", "agent", **agent(max_retries=3, use_few_shot=True, use_critic_semantic=True)),
        cfg("full", "+ router and pandas escalation", "agent",
            **agent(max_retries=3, use_few_shot=True, use_critic_semantic=True, use_router=True, allow_pandas=True)),
    ]  # fmt: skip


def write_configs(
    directory: Path = Path("configs/ablations"),
    model: str = "gemini-3.1-flash-lite",
    limit: int = 30,
) -> list[Path]:
    """Materialise the ladder as YAML files so every rung is a plain, reviewable config."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in ladder(model, limit):
        data = c.model_dump(exclude_defaults=True, mode="json")
        data["name"], data["description"] = c.name, c.description
        path = directory / f"{c.name}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def estimate_calls(n: int) -> dict[str, int]:
    """Rough upper-ish bound on uncached LLM calls per rung (helps plan a free-tier budget)."""
    per_q = {"zero_shot": 1, "full_schema": 1, "linker": 2, "retry_r1": 0.3, "retry_r2": 0.2,
             "retry_r3": 0.1, "few_shot": 2.5, "critic": 1.5, "full": 3.5}  # fmt: skip
    return {k: max(1, round(v * n)) for k, v in per_q.items()}


def run_ladder(
    model: str,
    limit: int,
    seed: int = 0,
    only: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, dict[str, Any]]:
    """Run (or resume) every rung and return ``name -> summary``."""
    summaries: dict[str, dict[str, Any]] = {}
    plan = estimate_calls(limit)
    print(f"Estimated new LLM calls for n={limit}: ~{sum(plan.values())} ({plan})")
    for c in ladder(model, limit, seed):
        if only and c.name not in only:
            continue
        path = results_dir / "_configs" / f"{c.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(c.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        summaries[c.name] = run_config(path)
        if not summaries[c.name].get("complete"):
            print("Stopping ladder early (quota/errors). Re-run the same command to resume.")
            break
    return summaries


def load_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    """qid -> record for a finished/partial run directory."""
    out: dict[str, dict[str, Any]] = {}
    path = run_dir / "per_question.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["qid"]] = rec
    return out


def paired_tests(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    """McNemar between consecutive rungs and vs the zero-shot baseline, on common qids."""
    recs = {name: load_records(d) for name, d in run_dirs.items()}
    names = list(run_dirs)
    rows: list[dict[str, Any]] = []
    pairs = list(pairwise(names)) + [(names[0], n) for n in names[2:]]
    for a, b in pairs:
        common = sorted(set(recs[a]) & set(recs[b]))
        if not common:
            continue
        res = mcnemar(
            [bool(recs[a][q]["ex"]) for q in common], [bool(recs[b][q]["ex"]) for q in common]
        )
        rows.append({"a": a, "b": b, "n": len(common), **res})
    return rows


def main() -> None:
    """CLI: ``python -m eval.ablations --limit 25 --model gemini-3.1-flash-lite``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--only", nargs="*", help="run only these rungs")
    ap.add_argument(
        "--write-configs", action="store_true", help="write configs/ablations/*.yaml and exit"
    )
    args = ap.parse_args()
    if args.write_configs:
        for p in write_configs(model=args.model, limit=args.limit):
            print("wrote", p)
        return
    run_ladder(args.model, args.limit, args.seed, args.only)


if __name__ == "__main__":
    main()

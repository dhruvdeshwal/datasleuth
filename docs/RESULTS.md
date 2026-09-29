# Results (generated)

## Business case study: Northwind trading database

**business_full__gemini-3.1-flash-lite__n23_s0**: EX 91.3% [78, 100] on 23 questions · first-attempt EX 91.3% · valid SQL 100.0% · 2.9 LLM calls/question

| Difficulty | n | EX |
|---|---:|---:|
| easy | 6 | 100.0% |
| medium | 10 | 90.0% |
| hard | 5 | 80.0% |
| unanswerable | 2 | 100.0% |

Failure notes: [results/business_full__gemini-3.1-flash-lite__n23_s0/error_analysis.md](results/business_full__gemini-3.1-flash-lite__n23_s0/error_analysis.md).

## Spider 1.0 dev: ablation ladder

**Model:** `gemini-3.1-flash-lite` · Spider 1.0 dev, stratified sample of **10** of 1,034 questions (seed 0), temperature 0. \*$/q uses paid-tier list prices (actual free-tier cost is $0).

| # | Configuration | n | EX [95% CI] | Valid SQL | 1st-attempt EX | Avg retries | Retry success | LLM calls/q | $/q* | Latency (s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | zero-shot | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 1.0 | 0.0000 | 16.7 |
| 2 | + full schema | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 1.0 | 0.0000 | 6.6 |
| 3 | + schema linker | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 2.0 | 0.0000 | 14.0 |
| 4a | + 1 retry | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 2.0 | 0.0000 | 0.0 |
| 4b | + 2 retries | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 2.0 | 0.0000 | 0.0 |
| 4c | + 3 retries | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 2.0 | 0.0000 | 0.0 |
| 5 | + few-shot | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.00 | n/a | 2.0 | 0.0000 | 8.1 |
| 6 | + critic | 10 | 80.0% [50, 100] | 100.0% | 80.0% | 0.30 | n/a | 2.6 | 0.0000 | 3.9 |
| 7 | full system | 10 | 70.0% [40, 100] | 100.0% | 80.0% | 0.30 | n/a | 3.8 | 0.0000 | 19.1 |

**Paired significance (exact McNemar on per-question correctness):**

| A | B | n | only A correct | only B correct | p-value |
|---|---|---:|---:|---:|---:|
| zero_shot | full_schema | 10 | 0 | 0 | 1.000 |
| full_schema | linker | 10 | 0 | 0 | 1.000 |
| linker | retry_r1 | 10 | 0 | 0 | 1.000 |
| retry_r1 | retry_r2 | 10 | 0 | 0 | 1.000 |
| retry_r2 | retry_r3 | 10 | 0 | 0 | 1.000 |
| retry_r3 | few_shot | 10 | 0 | 0 | 1.000 |
| few_shot | critic | 10 | 0 | 0 | 1.000 |
| critic | full | 10 | 1 | 0 | 1.000 |
| zero_shot | linker | 10 | 0 | 0 | 1.000 |
| zero_shot | retry_r1 | 10 | 0 | 0 | 1.000 |
| zero_shot | retry_r2 | 10 | 0 | 0 | 1.000 |
| zero_shot | retry_r3 | 10 | 0 | 0 | 1.000 |
| zero_shot | few_shot | 10 | 0 | 0 | 1.000 |
| zero_shot | critic | 10 | 0 | 0 | 1.000 |
| zero_shot | full | 10 | 1 | 0 | 1.000 |

With samples this small most adjacent steps are *not* statistically distinguishable; see Limitations.

**EX by difficulty:**

| Config | easy | medium | hard | extra |
|---|---:|---:|---:|---:|
| 1 zero-shot | 100.0% (n=2) | 75.0% (n=4) | 100.0% (n=2) | 50.0% (n=2) |
| 2 + full schema | 100.0% (n=2) | 75.0% (n=4) | 100.0% (n=2) | 50.0% (n=2) |
| 4c + 3 retries | 100.0% (n=2) | 75.0% (n=4) | 100.0% (n=2) | 50.0% (n=2) |
| 7 full system | 100.0% (n=2) | 75.0% (n=4) | 100.0% (n=2) | 0.0% (n=2) |

**Error types seen across attempts (best agent run):** `runtime_error`: 1

![ablation](results/ablation__gemini-3.1-flash-lite__n10/ablation_bars.png)

![by difficulty](results/ablation__gemini-3.1-flash-lite__n10/ex_by_difficulty.png) ![retries](results/ablation__gemini-3.1-flash-lite__n10/retry_curve.png)

![cost](results/ablation__gemini-3.1-flash-lite__n10/cost_vs_accuracy.png)

## Custom messy-CSV suite

_No custom-suite runs found._ Run:

```bash
datasleuth eval --config configs/custom_full.yaml
python scripts/generate_results.py
```

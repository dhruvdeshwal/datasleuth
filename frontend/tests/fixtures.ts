import type { Report, SchemaInfo } from "../src/api/types";

export const schema: SchemaInfo = {
  dialect: "sqlite",
  tables: [
    {
      name: "sales",
      source: "sales.csv",
      row_count: 3,
      columns: [
        { name: "id", original_name: "ID", dtype: "integer", semantic: null, null_count: 0, distinct_count: 3, min: "1", max: "3", samples: ["1"], primary_key: true },
        { name: "region", original_name: "Region", dtype: "text", semantic: null, null_count: 1, distinct_count: 2, min: "north", max: "south", samples: ["north"], primary_key: false },
        { name: "amount", original_name: "Amount", dtype: "real", semantic: "currency (symbols stripped)", null_count: 0, distinct_count: 3, min: "5", max: "20", samples: ["10"], primary_key: false },
      ],
      foreign_keys: [],
      warnings: ["column 'amount': 1 value(s) could not be parsed as real and were set to NULL"],
    },
  ],
};

const okResult = {
  ok: true,
  rows: [["north", 15.5], ["south", 20]],
  columns: ["region", "total"],
  row_count: 2,
  truncated: false,
  error_type: null,
  error_message: null,
  runtime_ms: 1,
};

export const report: Report = {
  trace_id: "abc123",
  question: "Total by region?",
  status: "success",
  answer: "South leads with 20; north has 15.5.",
  language: "sql",
  code: "SELECT region, SUM(amount) AS total FROM sales GROUP BY region",
  columns: ["region", "total"],
  rows: [["north", 15.5], ["south", 20]],
  row_count: 2,
  truncated: false,
  chart_spec: null,
  chart_json: null,
  has_chart_png: false,
  plan: {
    status: "answerable",
    question_types: ["aggregation"],
    needs_chart: false,
    expected_columns: 2,
    expects_many_rows: true,
    preferred_language: "sql",
    steps: "Group by region.",
    message: null,
  },
  attempts: [
    {
      index: 1,
      language: "sql",
      code: "SELECT region, SUM(amout) FROM sales GROUP BY region",
      rationale: "sum it",
      result: { ...okResult, ok: false, rows: [], columns: [], row_count: 0, error_type: "unknown_column", error_message: "no such column: amout" },
      error: "no such column: amout",
      diagnosis: {
        error_type: "unknown_column",
        explanation: "no such column: amout",
        repair_instruction: 'Column "amout" does not exist. Similar columns: sales.amount.',
        escalate: "none",
      },
    },
    {
      index: 2,
      language: "sql",
      code: "SELECT region, SUM(amount) AS total FROM sales GROUP BY region",
      rationale: "fixed",
      result: okResult,
      error: null,
      diagnosis: null,
    },
  ],
  caveats: [],
  failure_reason: null,
  metrics: { llm_calls: 4, prompt_tokens: 900, completion_tokens: 100, cost_usd: 0.0021, latency_ms: 2500, llm_latency_ms: 2400, attempts: 2, retries: 1 },
};

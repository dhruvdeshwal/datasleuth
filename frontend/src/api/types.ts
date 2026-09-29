import type { components } from "./schema";

type S = components["schemas"];

export type Report = S["Report"];
export type SchemaInfo = S["SchemaInfo"];
export type TableInfo = S["TableInfo"];
export type ColumnInfo = S["ColumnInfo"];
export type Attempt = S["Attempt"];
export type PreviewOut = S["PreviewOut"];

/** Progress events streamed over SSE while the agent runs. */
export interface NodeEvent {
  type: "node_start" | "node_end";
  node: string;
  label: string;
  detail?: Record<string, unknown>;
  llm_calls?: number;
}

import { useState } from "react";
import type { PreviewOut, SchemaInfo, TableInfo } from "../api/types";
import { ResultTable } from "./ResultTable";

interface Props {
  schema: SchemaInfo;
  loadPreview: (table: string) => Promise<PreviewOut>;
}

function TableCard({ table, loadPreview }: { table: TableInfo; loadPreview: Props["loadPreview"] }) {
  const [tab, setTab] = useState<"schema" | "preview">("schema");
  const [preview, setPreview] = useState<PreviewOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function openPreview() {
    setTab("preview");
    if (preview) return;
    try {
      setPreview(await loadPreview(table.name));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load preview.");
    }
  }

  return (
    <details className="group rounded-lg border border-line bg-surface" open>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2">
        <span className="min-w-0">
          <span className="block truncate font-mono text-sm font-medium">{table.name}</span>
          <span className="block truncate text-xs text-ink-3">
            {table.row_count.toLocaleString()} rows · {table.columns.length} columns
            {table.source ? ` · ${table.source}` : ""}
          </span>
        </span>
        <span aria-hidden="true" className="text-ink-3 transition-transform group-open:rotate-90">›</span>
      </summary>
      <div className="border-t border-line px-3 py-2">
        <div role="tablist" aria-label={`${table.name} views`} className="mb-2 flex gap-1 text-xs">
          {(["schema", "preview"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              type="button"
              aria-selected={tab === t}
              onClick={() => (t === "preview" ? openPreview() : setTab("schema"))}
              className={`rounded px-2 py-1 font-medium capitalize ${
                tab === t ? "bg-accent/15 text-accent" : "text-ink-2 hover:bg-surface-2"
              }`}
            >
              {t === "schema" ? "Inferred schema" : "Data preview"}
            </button>
          ))}
        </div>
        {tab === "schema" ? (
          <ul className="space-y-1.5 text-sm">
            {table.columns.map((c) => (
              <li key={c.name} className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono">{c.name}</span>
                <span className="rounded bg-surface-2 px-1.5 text-xs text-ink-2">{c.dtype}</span>
                {c.primary_key && <span className="text-xs font-medium text-accent">PK</span>}
                {c.semantic && <span className="text-xs text-ink-3">{c.semantic}</span>}
                {c.null_count > 0 && <span className="text-xs text-warn">{c.null_count} null</span>}
              </li>
            ))}
            {table.foreign_keys.map((fk) => (
              <li key={fk.columns.join()} className="text-xs text-ink-3">
                FK {fk.columns.join(", ")} → {fk.ref_table}.{fk.ref_columns.join(", ")}
              </li>
            ))}
            {table.warnings.map((w) => (
              <li key={w} className="text-xs text-warn">
                ⚠ {w}
              </li>
            ))}
          </ul>
        ) : error ? (
          <p role="alert" className="text-sm text-bad">
            {error}
          </p>
        ) : preview ? (
          <ResultTable columns={preview.columns} rows={preview.rows} pageSize={5} filename={`${table.name}_preview.csv`} />
        ) : (
          <p className="text-sm text-ink-2">Loading preview…</p>
        )}
      </div>
    </details>
  );
}

export function SchemaView({ schema, loadPreview }: Props) {
  return (
    <section aria-label="Your data" className="space-y-2">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2">Your data</h2>
      {schema.tables.map((t) => (
        <TableCard key={t.name} table={t} loadPreview={loadPreview} />
      ))}
    </section>
  );
}

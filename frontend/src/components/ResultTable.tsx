import { useState } from "react";
import { downloadText, toCsv } from "../lib/csv";

interface Props {
  columns: string[];
  rows: unknown[][];
  truncated?: boolean;
  filename?: string;
  pageSize?: number;
}

function show(v: unknown): string {
  if (v === null || v === undefined) return "NULL";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(6)));
  return String(v);
}

export function ResultTable({ columns, rows, truncated, filename = "result.csv", pageSize = 10 }: Props) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const current = Math.min(page, pages - 1);
  const slice = rows.slice(current * pageSize, (current + 1) * pageSize);

  if (rows.length === 0) {
    return <p className="rounded-lg border border-line px-3 py-4 text-sm text-ink-2">No rows returned.</p>;
  }
  return (
    <div className="rounded-lg border border-line">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2">
        <span className="text-xs text-ink-2">
          {rows.length} row{rows.length === 1 ? "" : "s"}
          {truncated ? " (truncated at the row cap)" : ""}
        </span>
        <button
          type="button"
          onClick={() => downloadText(filename, toCsv(columns, rows))}
          className="rounded px-2 py-1 text-xs font-medium text-accent hover:bg-accent/10"
        >
          Download CSV
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-surface-2 text-xs text-ink-2">
            <tr>
              {columns.map((c) => (
                <th key={c} scope="col" className="whitespace-nowrap px-3 py-2 font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slice.map((row, i) => (
              <tr key={i} className="border-t border-line">
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className={`whitespace-nowrap px-3 py-1.5 tabular-nums ${cell === null ? "text-ink-3 italic" : ""}`}
                  >
                    {show(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <nav aria-label="Result pages" className="flex items-center justify-between border-t border-line px-3 py-2 text-xs">
          <button
            type="button"
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
            className="rounded px-2 py-1 font-medium text-accent enabled:hover:bg-accent/10 disabled:text-ink-3"
          >
            Previous
          </button>
          <span aria-live="polite">
            Page {current + 1} of {pages}
          </span>
          <button
            type="button"
            disabled={current >= pages - 1}
            onClick={() => setPage(current + 1)}
            className="rounded px-2 py-1 font-medium text-accent enabled:hover:bg-accent/10 disabled:text-ink-3"
          >
            Next
          </button>
        </nav>
      )}
    </div>
  );
}

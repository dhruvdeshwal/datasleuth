import { useState } from "react";
import type { SchemaInfo } from "../api/types";

export function suggestQuestions(schema: SchemaInfo | null): string[] {
  const table = schema?.tables[0];
  if (!table) return [];
  const numeric = table.columns.find((c) => (c.dtype === "real" || c.dtype === "integer") && !c.primary_key);
  const category = table.columns.find((c) => c.dtype === "text" && (c.distinct_count ?? 99) <= 20);
  const out = [`How many rows are in ${table.name}?`];
  if (numeric && category) out.push(`What is the average ${numeric.name} for each ${category.name}?`);
  if (numeric) out.push(`What are the top 5 rows by ${numeric.name}?`);
  return out;
}

interface Props {
  onAsk: (question: string) => void;
  disabled: boolean;
  disabledReason?: string;
  suggestions: string[];
}

export function ChatInput({ onAsk, disabled, disabledReason, suggestions }: Props) {
  const [text, setText] = useState("");

  function submit() {
    const q = text.trim();
    if (!q || disabled) return;
    onAsk(q);
    setText("");
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="space-y-2"
    >
      {suggestions.length > 0 && !disabled && (
        <div className="flex flex-wrap gap-2" aria-label="Suggested questions">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => onAsk(s)}
              className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-accent"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <label htmlFor="question" className="sr-only">
          Ask a question about your data
        </label>
        <textarea
          id="question"
          value={text}
          rows={2}
          maxLength={2000}
          disabled={disabled}
          placeholder={disabledReason ?? "Ask a question about your data…"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          className="min-h-[3rem] flex-1 resize-none rounded-xl border border-line bg-surface px-3 py-2 text-sm placeholder:text-ink-3 disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          Ask
        </button>
      </div>
    </form>
  );
}

import type { Report } from "../api/types";
import { CodeViewer } from "./CodeViewer";

export function AgentTrace({ report }: { report: Report }) {
  const m = report.metrics;
  return (
    <details className="rounded-lg border border-line" data-testid="agent-trace">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
        Agent trace · {report.attempts.length} attempt{report.attempts.length === 1 ? "" : "s"}
        {m.retries > 0 && <span className="text-warn"> · self-corrected {m.retries}×</span>}
      </summary>
      <div className="space-y-4 border-t border-line p-3">
        {report.plan && (
          <div className="text-sm">
            <h4 className="font-medium">Plan</h4>
            <p className="text-ink-2">
              {report.plan.question_types?.join(", ") || "n/a"}
              {report.plan.steps ? ` — ${report.plan.steps}` : ""}
            </p>
          </div>
        )}
        {report.attempts.map((a) => {
          const res = a.result;
          return (
            <section key={a.index} aria-label={`Attempt ${a.index}`} className="space-y-2">
              <h4 className="flex flex-wrap items-center gap-2 text-sm font-medium">
                Attempt {a.index}
                <span
                  className={`rounded px-1.5 py-0.5 text-xs ${
                    res?.ok ? "bg-good/15 text-good" : "bg-bad/15 text-bad"
                  }`}
                >
                  {res?.ok ? `ok · ${res.row_count} rows` : `error · ${res?.error_type ?? "unknown"}`}
                </span>
              </h4>
              {a.rationale && <p className="text-sm text-ink-2">{a.rationale}</p>}
              {a.code && <CodeViewer code={a.code} language={a.language} />}
              {res && !res.ok && (
                <p className="rounded bg-bad/10 px-3 py-2 font-mono text-xs text-bad">{res.error_message}</p>
              )}
              {a.diagnosis && (
                <div className="rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-sm">
                  <p className="font-medium text-warn">
                    Diagnosis: {a.diagnosis.error_type}
                    {a.diagnosis.escalate !== "none" && ` · escalation: ${a.diagnosis.escalate.replace("_", " ")}`}
                  </p>
                  <p className="mt-1 text-ink-2">
                    <span className="font-medium text-ink">Fix sent to the writer:</span>{" "}
                    {a.diagnosis.repair_instruction}
                  </p>
                </div>
              )}
            </section>
          );
        })}
        <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
          {[
            ["LLM calls", String(m.llm_calls)],
            ["Tokens", (m.prompt_tokens + m.completion_tokens).toLocaleString()],
            ["Est. cost", `$${m.cost_usd.toFixed(4)}`],
            ["Latency", `${(m.latency_ms / 1000).toFixed(1)} s`],
          ].map(([k, v]) => (
            <div key={k} className="rounded bg-surface-2 px-2 py-1.5">
              <dt className="text-ink-3">{k}</dt>
              <dd className="font-medium tabular-nums">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </details>
  );
}

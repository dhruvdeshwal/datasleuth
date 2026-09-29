import type { NodeEvent, Report } from "../api/types";
import { AgentTrace } from "./AgentTrace";
import { ChartView } from "./ChartView";
import { CodeViewer } from "./CodeViewer";
import { ProgressSteps } from "./ProgressSteps";
import { ResultTable } from "./ResultTable";

export interface Message {
  id: string;
  question: string;
  status: "running" | "done" | "error";
  events: NodeEvent[];
  report?: Report;
  error?: string;
}

function Notice({ tone, title, children }: { tone: "info" | "bad"; title: string; children: string }) {
  const style = tone === "bad" ? "border-bad/40 bg-bad/10" : "border-accent/40 bg-accent/10";
  return (
    <div role={tone === "bad" ? "alert" : "status"} className={`rounded-lg border px-3 py-2 ${style}`}>
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-0.5 text-sm text-ink-2">{children}</p>
    </div>
  );
}

export function AnswerCard({ message, dark }: { message: Message; dark: boolean }) {
  const { report } = message;
  return (
    <article className="space-y-3" aria-label={`Answer to: ${message.question}`}>
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-2 text-sm text-accent-ink">
        {message.question}
      </div>
      <div className="space-y-3 rounded-2xl rounded-bl-sm border border-line bg-surface p-4">
        {message.status === "running" && <ProgressSteps events={message.events} running />}
        {message.status === "error" && (
          <Notice tone="bad" title="Something went wrong">
            {message.error ?? "Unknown error."}
          </Notice>
        )}
        {report?.status === "clarify" && (
          <Notice tone="info" title="I need a bit more information">
            {report.answer ?? ""}
          </Notice>
        )}
        {report?.status === "refused" && (
          <Notice tone="info" title="I can't answer that from this data">
            {report.answer ?? ""}
          </Notice>
        )}
        {report?.status === "failed" && (
          <Notice tone="bad" title="I couldn't answer this">
            {report.answer ?? report.failure_reason ?? "The agent could not produce a working query."}
          </Notice>
        )}
        {report?.status === "success" && (
          <>
            <p className="text-[15px] leading-relaxed">{report.answer}</p>
            {report.caveats.length > 0 && (
              <ul className="space-y-1 text-sm text-warn">
                {report.caveats.map((c) => (
                  <li key={c}>⚠ {c}</li>
                ))}
              </ul>
            )}
            {report.chart_json && <ChartView chartJson={report.chart_json} dark={dark} filename={`chart-${report.trace_id}`} />}
            {report.code && report.language && <CodeViewer code={report.code} language={report.language} />}
            <ResultTable
              columns={report.columns}
              rows={report.rows}
              truncated={report.truncated}
              filename={`result-${report.trace_id}.csv`}
            />
          </>
        )}
        {report && report.attempts.length > 0 && <AgentTrace report={report} />}
        {message.status === "running" && message.events.length > 0 && (
          <span className="sr-only">Working on your question</span>
        )}
      </div>
    </article>
  );
}

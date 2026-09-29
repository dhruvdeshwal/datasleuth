import type { NodeEvent } from "../api/types";

export interface Step {
  node: string;
  label: string;
  status: "running" | "done";
  detail?: Record<string, unknown>;
}

/** Fold the raw start/end event stream into one row per node execution. */
export function buildSteps(events: NodeEvent[]): Step[] {
  const steps: Step[] = [];
  for (const ev of events) {
    if (ev.type === "node_start") {
      steps.push({ node: ev.node, label: ev.label, status: "running" });
    } else {
      const open = [...steps].reverse().find((s) => s.node === ev.node && s.status === "running");
      if (open) {
        open.status = "done";
        open.detail = ev.detail;
      }
    }
  }
  return steps;
}

/** One-line human summary of what a finished step did (shows self-correction live). */
export function describeStep(step: Step): string | null {
  const d = step.detail ?? {};
  switch (step.node) {
    case "write":
      return `attempt ${d.attempt} · ${d.language}`;
    case "validate":
      return d.ok === false ? `blocked: ${d.error}` : "passed the safety policy";
    case "execute":
      return d.ok ? `${d.row_count} row(s)` : `failed (${d.error_type}): ${d.error}`;
    case "critic":
      if (d.status === "success") return "accepted";
      if (d.status === "failed") return "giving up";
      return `retry ${d.retry_count}: ${d.error_type}${d.escalation ? ` → ${String(d.escalation).replace("_", " ")}` : ""}`;
    case "plan":
      return d.status && d.status !== "answerable" ? String(d.status) : null;
    case "link":
      return Array.isArray(d.tables) ? (d.tables as string[]).join(", ") : null;
    case "chart":
      return d.chart_type ? String(d.chart_type) : "no chart needed";
    default:
      return null;
  }
}

export function ProgressSteps({ events, running }: { events: NodeEvent[]; running: boolean }) {
  const steps = buildSteps(events);
  return (
    <ol aria-label="Agent progress" aria-live="polite" className="space-y-1 text-sm">
      {steps.map((s, i) => {
        const note = s.status === "done" ? describeStep(s) : null;
        const failed = s.detail?.ok === false;
        return (
          <li key={i} className="flex items-start gap-2">
            <span
              aria-hidden="true"
              className={`mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full ${
                s.status === "running" ? "animate-pulse bg-accent" : failed ? "bg-bad" : "bg-good"
              }`}
            />
            <span className={s.status === "running" ? "text-ink" : "text-ink-2"}>
              {s.label}
              {note && <span className="text-ink-3"> — {note}</span>}
            </span>
          </li>
        );
      })}
      {running && steps.length === 0 && <li className="text-ink-2">Starting…</li>}
    </ol>
  );
}

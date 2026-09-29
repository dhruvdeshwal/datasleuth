import { render, screen } from "@testing-library/react";
import type { NodeEvent } from "../src/api/types";
import { ProgressSteps, buildSteps, describeStep } from "../src/components/ProgressSteps";

const ev = (type: NodeEvent["type"], node: string, detail?: Record<string, unknown>): NodeEvent => ({
  type,
  node,
  label: `L:${node}`,
  detail,
});

describe("ProgressSteps", () => {
  const events: NodeEvent[] = [
    ev("node_start", "write"),
    ev("node_end", "write", { attempt: 1, language: "sql" }),
    ev("node_start", "execute"),
    ev("node_end", "execute", { ok: false, error_type: "unknown_column", error: "no such column: x" }),
    ev("node_start", "critic"),
    ev("node_end", "critic", { status: "running", retry_count: 1, error_type: "unknown_column", escalation: "widen_schema" }),
    ev("node_start", "write"),
  ];

  it("folds start/end events into one row per execution", () => {
    const steps = buildSteps(events);
    expect(steps.map((s) => [s.node, s.status])).toEqual([
      ["write", "done"],
      ["execute", "done"],
      ["critic", "done"],
      ["write", "running"],
    ]);
  });

  it("makes self-correction visible", () => {
    const steps = buildSteps(events);
    expect(describeStep(steps[1])).toContain("failed (unknown_column)");
    expect(describeStep(steps[2])).toBe("retry 1: unknown_column → widen schema");
    render(<ProgressSteps events={events} running />);
    expect(screen.getByRole("list", { name: /agent progress/i })).toBeInTheDocument();
    expect(screen.getAllByText(/L:write/)).toHaveLength(2);
  });

  it("shows a starting placeholder before the first event", () => {
    render(<ProgressSteps events={[]} running />);
    expect(screen.getByText("Starting…")).toBeInTheDocument();
  });
});

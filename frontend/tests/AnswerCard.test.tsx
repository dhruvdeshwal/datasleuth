import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AnswerCard, type Message } from "../src/components/AnswerCard";
import { report } from "./fixtures";

vi.mock("../src/components/ChartView", () => ({
  ChartView: () => <div data-testid="chart">chart</div>,
}));

const base: Message = { id: "1", question: "Total by region?", status: "done", events: [], report };

describe("AnswerCard", () => {
  it("renders answer, code, result table and chart", () => {
    render(<AnswerCard message={{ ...base, report: { ...report, chart_json: "{}" } }} dark={false} />);
    expect(screen.getByText(report.answer as string)).toBeInTheDocument();
    expect(screen.getAllByLabelText("sql code")[0]).toHaveTextContent("GROUP BY region");
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByTestId("chart")).toBeInTheDocument();
  });

  it("shows the self-correction trace with diagnosis and fix", async () => {
    render(<AnswerCard message={base} dark={false} />);
    expect(screen.getByText(/self-corrected 1×/)).toBeInTheDocument();
    await userEvent.click(screen.getByText(/agent trace/i));
    expect(screen.getByText(/Diagnosis: unknown_column/)).toBeInTheDocument();
    expect(screen.getByText(/Fix sent to the writer/)).toBeInTheDocument();
    expect(screen.getByText("no such column: amout")).toBeInTheDocument();
    expect(screen.getByText("$0.0021")).toBeInTheDocument();
  });

  it("shows a clarification request", () => {
    const r = { ...report, status: "clarify" as const, answer: "Best by revenue or units?", attempts: [] };
    render(<AnswerCard message={{ ...base, report: r }} dark={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("Best by revenue or units?");
  });

  it("shows failures as alerts with the trace still available", () => {
    const r = { ...report, status: "failed" as const, answer: "I couldn't answer this question: boom." };
    render(<AnswerCard message={{ ...base, report: r }} dark={false} />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
    expect(screen.getByTestId("agent-trace")).toBeInTheDocument();
  });

  it("shows live progress while running and errors from the stream", () => {
    const { rerender } = render(
      <AnswerCard
        message={{ id: "2", question: "q", status: "running", events: [{ type: "node_start", node: "plan", label: "Planning" }] }}
        dark={false}
      />,
    );
    expect(screen.getByText("Planning")).toBeInTheDocument();
    rerender(<AnswerCard message={{ id: "2", question: "q", status: "error", events: [], error: "Quota used up." }} dark={false} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Quota used up.");
  });

  it("lists caveats", () => {
    render(<AnswerCard message={{ ...base, report: { ...report, caveats: ["Result may be incomplete."] } }} dark={false} />);
    expect(screen.getByText(/Result may be incomplete/)).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatInput, suggestQuestions } from "../src/components/ChatInput";
import { schema } from "./fixtures";

describe("ChatInput", () => {
  it("submits on Enter, keeps Shift+Enter for newlines, and clears", async () => {
    const onAsk = vi.fn();
    render(<ChatInput onAsk={onAsk} disabled={false} suggestions={[]} />);
    const box = screen.getByLabelText(/ask a question/i);
    await userEvent.type(box, "how many rows{Shift>}{Enter}{/Shift}more");
    expect(onAsk).not.toHaveBeenCalled();
    await userEvent.type(box, "{Enter}");
    expect(onAsk).toHaveBeenCalledWith("how many rows\nmore");
    expect(box).toHaveValue("");
  });

  it("is disabled with a reason, and suggestions are clickable", async () => {
    const { rerender } = render(
      <ChatInput onAsk={vi.fn()} disabled disabledReason="Upload data first…" suggestions={["S1"]} />,
    );
    expect(screen.getByLabelText(/ask a question/i)).toBeDisabled();
    expect(screen.getByPlaceholderText("Upload data first…")).toBeInTheDocument();
    expect(screen.queryByText("S1")).not.toBeInTheDocument();
    const onAsk = vi.fn();
    rerender(<ChatInput onAsk={onAsk} disabled={false} suggestions={["S1"]} />);
    await userEvent.click(screen.getByText("S1"));
    expect(onAsk).toHaveBeenCalledWith("S1");
  });

  it("derives suggestions from the schema", () => {
    const s = suggestQuestions(schema);
    expect(s[0]).toBe("How many rows are in sales?");
    expect(s.some((q) => q.includes("amount"))).toBe(true);
    expect(suggestQuestions(null)).toEqual([]);
  });
});

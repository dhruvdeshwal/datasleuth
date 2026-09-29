import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CodeViewer } from "../src/components/CodeViewer";

describe("CodeViewer", () => {
  it("highlights SQL and copies it", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<CodeViewer code="SELECT 1 FROM t" language="sql" />);
    expect(document.querySelector(".hljs-keyword")).toHaveTextContent("SELECT");
    await userEvent.click(screen.getByRole("button", { name: /copy sql/i }));
    expect(writeText).toHaveBeenCalledWith("SELECT 1 FROM t");
    expect(await screen.findByText(/copied/i)).toBeInTheDocument();
  });

  it("escapes HTML in code instead of injecting it", () => {
    render(<CodeViewer code={"SELECT '<img src=x onerror=alert(1)>'"} language="sql" />);
    expect(document.querySelector("img")).toBeNull();
  });

  it("labels pandas code", () => {
    render(<CodeViewer code="result = 1" language="pandas" />);
    expect(screen.getByRole("button", { name: /copy code/i })).toBeInTheDocument();
  });
});

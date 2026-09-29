import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ResultTable } from "../src/components/ResultTable";

const rows = Array.from({ length: 25 }, (_, i) => [`r${i}`, i === 3 ? null : i * 1.5]);

describe("ResultTable", () => {
  it("paginates and renders NULL distinctly", async () => {
    render(<ResultTable columns={["name", "v"]} rows={rows} pageSize={10} />);
    expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(11);
    expect(screen.getByText("NULL")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Page 2 of 3")).toBeInTheDocument();
    expect(screen.getByText("r10")).toBeInTheDocument();
    expect(screen.queryByText("r0")).not.toBeInTheDocument();
  });

  it("downloads CSV", async () => {
    const create = vi.fn(() => "blob:x");
    Object.assign(URL, { createObjectURL: create, revokeObjectURL: vi.fn() });
    render(<ResultTable columns={["a"]} rows={[[1]]} />);
    await userEvent.click(screen.getByRole("button", { name: /download csv/i }));
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("has an empty state and a truncation notice", () => {
    const { rerender } = render(<ResultTable columns={["a"]} rows={[]} />);
    expect(screen.getByText(/no rows returned/i)).toBeInTheDocument();
    rerender(<ResultTable columns={["a"]} rows={[[1]]} truncated />);
    expect(screen.getByText(/truncated/i)).toBeInTheDocument();
  });

  it("uses proper table semantics", () => {
    render(<ResultTable columns={["name", "v"]} rows={rows} />);
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["name", "v"]);
  });
});

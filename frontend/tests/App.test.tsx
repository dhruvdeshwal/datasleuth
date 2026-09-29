import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../src/App";
import * as api from "../src/api/client";
import { report, schema } from "./fixtures";

vi.mock("../src/api/client");
vi.mock("../src/components/ChartView", () => ({ ChartView: () => null }));

const csv = new File(["a,b\n1,2\n"], "sales.csv", { type: "text/csv" });

beforeEach(() => {
  vi.resetAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(api.createSession).mockResolvedValue("s1");
  vi.mocked(api.uploadFiles).mockResolvedValue(schema);
  vi.mocked(api.deleteSession).mockResolvedValue();
});

async function uploadData() {
  await userEvent.upload(screen.getByTestId("file-input"), csv);
  await screen.findByText("sales");
}

describe("App", () => {
  it("starts in an empty, guided state with input disabled", () => {
    render(<App />);
    expect(screen.getByText(/upload a file to get started/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/ask a question/i)).toBeDisabled();
  });

  it("uploads, shows the inferred schema, then answers a question with live progress", async () => {
    vi.mocked(api.ask).mockImplementation(async (_id, _q, h) => {
      h.onProgress({ type: "node_start", node: "plan", label: "Planning" });
      h.onProgress({ type: "node_end", node: "plan", label: "Planning", detail: {} });
      h.onResult(report);
    });
    render(<App />);
    await uploadData();
    expect(api.uploadFiles).toHaveBeenCalledWith("s1", [csv]);
    expect(screen.getByText("currency (symbols stripped)")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText(/ask a question/i), "Total by region?{Enter}");
    expect(await screen.findByText(report.answer as string)).toBeInTheDocument();
    expect(api.ask).toHaveBeenCalledWith("s1", "Total by region?", expect.any(Object));
  });

  it("shows upload errors from the server", async () => {
    vi.mocked(api.uploadFiles).mockRejectedValue(
      Object.assign(new Error("Unsupported file type '.xlsm'."), { status: 400 }),
    );
    render(<App />);
    await userEvent.upload(screen.getByTestId("file-input"), csv);
    expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported file type");
  });

  it("surfaces stream errors on the message", async () => {
    vi.mocked(api.ask).mockImplementation(async (_i, _q, h) => h.onError("The daily LLM quota is used up."));
    render(<App />);
    await uploadData();
    await userEvent.type(screen.getByLabelText(/ask a question/i), "q{Enter}");
    expect(await screen.findByText(/daily LLM quota/)).toBeInTheDocument();
  });

  it("handles an expired session by asking to re-upload", async () => {
    vi.mocked(api.ask).mockRejectedValue(
      Object.assign(new Error("Session not found or expired."), { status: 404 }),
    );
    render(<App />);
    await uploadData();
    await userEvent.type(screen.getByLabelText(/ask a question/i), "q{Enter}");
    expect(await screen.findByText(/session expired/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/ask a question/i)).toBeDisabled();
  });

  it("New session deletes server data and resets the UI", async () => {
    render(<App />);
    await uploadData();
    await userEvent.click(screen.getByRole("button", { name: /new session/i }));
    await waitFor(() => expect(api.deleteSession).toHaveBeenCalledWith("s1"));
    expect(screen.getByText(/upload a file to get started/i)).toBeInTheDocument();
  });

  it("toggles dark mode", async () => {
    render(<App />);
    const before = document.documentElement.classList.contains("dark");
    await userEvent.click(screen.getByRole("button", { name: /switch to (dark|light) mode/i }));
    expect(document.documentElement.classList.contains("dark")).toBe(!before);
  });

  it("loads the data preview on demand", async () => {
    vi.mocked(api.getPreview).mockResolvedValue({ table: "sales", columns: ["id"], rows: [[1]] });
    render(<App />);
    await uploadData();
    await userEvent.click(screen.getByRole("tab", { name: /data preview/i }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(api.getPreview).toHaveBeenCalledWith("s1", "sales");
  });
});

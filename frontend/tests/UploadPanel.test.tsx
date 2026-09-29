import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UploadPanel } from "../src/components/UploadPanel";

const csv = (name = "sales.csv", body = "a,b\n1,2\n") => new File([body], name, { type: "text/csv" });
const zone = () => screen.getByText(/drop csv or xlsx|profiling/i).closest("div") as HTMLElement;

describe("UploadPanel", () => {
  it("passes valid files chosen through the file input", async () => {
    const onFiles = vi.fn();
    render(<UploadPanel onFiles={onFiles} busy={false} />);
    await userEvent.upload(screen.getByTestId("file-input"), csv());
    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(onFiles.mock.calls[0][0][0].name).toBe("sales.csv");
  });

  it("accepts drag-and-drop", () => {
    const onFiles = vi.fn();
    render(<UploadPanel onFiles={onFiles} busy={false} />);
    fireEvent.drop(zone(), { dataTransfer: { files: [csv("dropped.csv")] } });
    expect(onFiles).toHaveBeenCalled();
  });

  it("rejects unsupported types client-side with a readable message", () => {
    const onFiles = vi.fn();
    render(<UploadPanel onFiles={onFiles} busy={false} />);
    fireEvent.drop(zone(), { dataTransfer: { files: [new File(["x"], "virus.exe")] } });
    expect(onFiles).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/not supported/i);
  });

  it("rejects empty files", () => {
    render(<UploadPanel onFiles={vi.fn()} busy={false} />);
    fireEvent.drop(zone(), { dataTransfer: { files: [csv("e.csv", "")] } });
    expect(screen.getByRole("alert")).toHaveTextContent(/empty/i);
  });

  it("shows server errors and a busy state", () => {
    render(<UploadPanel onFiles={vi.fn()} busy error="Upload failed on the server." />);
    expect(screen.getByRole("alert")).toHaveTextContent("Upload failed on the server.");
    expect(screen.getByText(/profiling/i)).toBeInTheDocument();
    expect(screen.getByTestId("file-input")).toBeDisabled();
  });
});

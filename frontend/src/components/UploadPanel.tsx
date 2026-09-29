import { useId, useRef, useState } from "react";

const ALLOWED = [".csv", ".tsv", ".txt", ".xlsx"];
const MAX_MB = 200;

export function validateFiles(files: File[]): string | null {
  for (const f of files) {
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED.includes(ext)) return `“${f.name}” is not supported. Upload CSV, TSV or XLSX files.`;
    if (f.size === 0) return `“${f.name}” is empty.`;
    if (f.size > MAX_MB * 1024 * 1024) return `“${f.name}” is larger than ${MAX_MB} MB.`;
  }
  return null;
}

interface Props {
  onFiles: (files: File[]) => void;
  busy: boolean;
  error?: string | null;
}

export function UploadPanel({ onFiles, busy, error }: Props) {
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const hintId = useId();

  function submit(list: FileList | null) {
    const files = list ? Array.from(list) : [];
    if (files.length === 0) return;
    const problem = validateFiles(files);
    setLocalError(problem);
    if (!problem) onFiles(files);
    if (inputRef.current) inputRef.current.value = "";
  }

  const shownError = localError ?? error;
  return (
    <section aria-label="Upload data">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          submit(e.dataTransfer.files);
        }}
        className={`relative rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
          dragging ? "border-accent bg-accent/10" : "border-line bg-surface"
        } ${busy ? "opacity-60" : ""}`}
      >
        <input
          ref={inputRef}
          id="file-input"
          data-testid="file-input"
          type="file"
          multiple
          accept={ALLOWED.join(",")}
          disabled={busy}
          aria-describedby={hintId}
          onChange={(e) => submit(e.target.files)}
          className="sr-only"
        />
        <label htmlFor="file-input" className="block cursor-pointer">
          <span className="mx-auto mb-2 block h-8 w-8 text-ink-3" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="block text-sm font-medium">
            {busy ? "Reading and profiling your data…" : "Drop CSV or XLSX files here"}
          </span>
          <span className="mt-1 block text-sm text-accent underline underline-offset-2">
            or browse files
          </span>
        </label>
        <p id={hintId} className="mt-3 text-xs text-ink-3">
          CSV, TSV, XLSX · up to {MAX_MB} MB · stays in your private session and is deleted after it
          expires.
        </p>
      </div>
      {shownError && (
        <p role="alert" className="mt-2 rounded-lg bg-bad/10 px-3 py-2 text-sm text-bad">
          {shownError}
        </p>
      )}
    </section>
  );
}

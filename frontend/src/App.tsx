import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api/client";
import type { NodeEvent, SchemaInfo } from "./api/types";
import { AnswerCard, type Message } from "./components/AnswerCard";
import { ChatInput, suggestQuestions } from "./components/ChatInput";
import { SchemaView } from "./components/SchemaView";
import { ThemeToggle } from "./components/ThemeToggle";
import { UploadPanel } from "./components/UploadPanel";
import { useTheme } from "./lib/theme";

let counter = 0;
const nextId = () => `m${++counter}`;

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [schema, setSchema] = useState<SchemaInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages]);

  const resetSession = useCallback(() => {
    setSessionId(null);
    setSchema(null);
    setMessages([]);
    setUploadError(null);
  }, []);

  async function handleFiles(files: File[]) {
    setUploading(true);
    setUploadError(null);
    try {
      const id = sessionId ?? (await api.createSession());
      setSessionId(id);
      setSchema(await api.uploadFiles(id, files));
    } catch (e) {
      const err = e as api.ApiError;
      setUploadError(err.message || "Upload failed.");
      if (err.status === 404) resetSession();
    } finally {
      setUploading(false);
    }
  }

  async function newSession() {
    if (sessionId) await api.deleteSession(sessionId).catch(() => undefined);
    resetSession();
  }

  const patch = (id: string, fn: (m: Message) => Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? fn(m) : m)));

  async function handleAsk(question: string) {
    if (!sessionId || asking) return;
    const id = nextId();
    setMessages((all) => [...all, { id, question, status: "running", events: [] }]);
    setAsking(true);
    try {
      await api.ask(sessionId, question, {
        onProgress: (ev: NodeEvent) => patch(id, (m) => ({ ...m, events: [...m.events, ev] })),
        onResult: (report) => patch(id, (m) => ({ ...m, status: "done", report })),
        onError: (message) => patch(id, (m) => ({ ...m, status: "error", error: message })),
      });
      patch(id, (m) => (m.status === "running" ? { ...m, status: "error", error: "The connection ended before an answer arrived." } : m));
    } catch (e) {
      const err = e as api.ApiError;
      patch(id, (m) => ({ ...m, status: "error", error: err.message }));
      if (err.status === 404) {
        setSessionId(null);
        setSchema(null);
        setUploadError("Your session expired. Please upload your data again.");
      }
    } finally {
      setAsking(false);
    }
  }

  const hasData = schema !== null;
  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 pb-6">
      <header className="flex items-center justify-between py-4">
        <h1 className="flex items-center gap-2 text-lg font-semibold">
          <span aria-hidden="true" className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-accent-ink">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="3">
              <circle cx="10" cy="10" r="6" />
              <path d="M15 15l6 6" strokeLinecap="round" />
            </svg>
          </span>
          DataSleuth
        </h1>
        <div className="flex items-center gap-2">
          {hasData && (
            <button
              type="button"
              onClick={newSession}
              className="rounded-lg border border-line px-3 py-1.5 text-sm hover:bg-surface-2"
            >
              New session
            </button>
          )}
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      <div className="grid flex-1 gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="space-y-4">
          <UploadPanel onFiles={handleFiles} busy={uploading} error={uploadError} />
          {schema && <SchemaView schema={schema} loadPreview={(t) => api.getPreview(sessionId as string, t)} />}
        </aside>

        <main className="flex min-h-[60vh] min-w-0 flex-col">
          <div className="flex-1 space-y-6 pb-4">
            {messages.length === 0 ? (
              <div className="grid place-items-center rounded-2xl border border-dashed border-line px-6 py-16 text-center">
                <div>
                  <h2 className="text-base font-medium">
                    {hasData ? "Ask your first question" : "Upload a file to get started"}
                  </h2>
                  <p className="mx-auto mt-1 max-w-md text-sm text-ink-2">
                    {hasData
                      ? "DataSleuth writes and runs the SQL for you, fixes its own mistakes, and shows every step."
                      : "Drop a CSV or Excel file on the left. Your data stays in a private session and is deleted after it expires."}
                  </p>
                </div>
              </div>
            ) : (
              messages.map((m) => <AnswerCard key={m.id} message={m} dark={theme === "dark"} />)
            )}
            <div ref={endRef} />
          </div>
          <div className="sticky bottom-0 bg-page/90 pt-2 backdrop-blur">
            <ChatInput
              onAsk={handleAsk}
              disabled={!hasData || asking}
              disabledReason={hasData ? undefined : "Upload data first…"}
              suggestions={messages.length === 0 ? suggestQuestions(schema) : []}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

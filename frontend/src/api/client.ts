import createClient from "openapi-fetch";
import { parseSSE } from "../lib/sse";
import type { paths } from "./schema";
import type { NodeEvent, PreviewOut, Report, SchemaInfo } from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

const client = createClient<paths>({ baseUrl: BASE });

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

function detailOf(error: unknown, fallback: string): string {
  if (error && typeof error === "object" && "detail" in error) {
    const d = (error as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d) && d[0] && typeof d[0] === "object" && "msg" in d[0]) {
      return String((d[0] as { msg: unknown }).msg);
    }
  }
  return fallback;
}

export async function createSession(): Promise<string> {
  const { data, error, response } = await client.POST("/sessions");
  if (!data) throw new ApiError(detailOf(error, "Could not start a session."), response.status);
  return data.session_id;
}

export async function deleteSession(id: string): Promise<void> {
  await client.DELETE("/sessions/{session_id}", { params: { path: { session_id: id } } });
}

/** Upload uses raw fetch: multipart bodies need the browser's FormData boundary handling. */
export async function uploadFiles(id: string, files: File[]): Promise<SchemaInfo> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${BASE}/sessions/${id}/upload`, { method: "POST", body: form });
  const body: unknown = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(detailOf(body, "Upload failed."), res.status);
  return body as SchemaInfo;
}

export async function getPreview(id: string, table: string): Promise<PreviewOut> {
  const { data, error, response } = await client.GET("/sessions/{session_id}/preview/{table}", {
    params: { path: { session_id: id, table }, query: { limit: 20 } },
  });
  if (!data) throw new ApiError(detailOf(error, "Could not load preview."), response.status);
  return data;
}

export interface AskHandlers {
  onStart?: (traceId: string) => void;
  onProgress: (event: NodeEvent) => void;
  onResult: (report: Report) => void;
  onError: (message: string) => void;
}

/** Ask a question and dispatch streamed events. Resolves when the stream ends. */
export async function ask(
  id: string,
  question: string,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${id}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) {
    const body: unknown = await res.json().catch(() => ({}));
    throw new ApiError(detailOf(body, "The question could not be sent."), res.status);
  }
  for await (const msg of parseSSE(res.body)) {
    if (msg.event === "start") handlers.onStart?.((msg.data as { trace_id: string }).trace_id);
    else if (msg.event === "progress") handlers.onProgress(msg.data as NodeEvent);
    else if (msg.event === "result") handlers.onResult(msg.data as Report);
    else if (msg.event === "error") handlers.onError((msg.data as { message: string }).message);
  }
}

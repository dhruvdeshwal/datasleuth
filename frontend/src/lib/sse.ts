export interface SSEMessage {
  event: string;
  data: unknown;
}

/**
 * Parse a text/event-stream body. EventSource only supports GET, but our ask endpoint is a
 * POST, so we read the fetch stream and split on blank lines ourselves.
 */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      let idx: number;
      while ((idx = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx).replace(/^\r?\n\r?\n/, "");
        const msg = decodeBlock(block);
        if (msg) yield msg;
      }
      if (done) break;
    }
    const tail = decodeBlock(buffer.trim());
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function decodeBlock(block: string): SSEMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (data.length === 0) return null;
  const raw = data.join("\n");
  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: raw };
  }
}

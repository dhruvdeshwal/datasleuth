import { parseSSE } from "../src/lib/sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(enc.encode(c)));
      controller.close();
    },
  });
}

async function collect(chunks: string[]) {
  const out = [];
  for await (const m of parseSSE(streamOf(chunks))) out.push(m);
  return out;
}

describe("parseSSE", () => {
  it("parses events split across arbitrary chunk boundaries", async () => {
    const wire = 'event: start\ndata: {"trace_id":"t1"}\n\nevent: progress\ndata: {"node":"plan"}\n\n';
    const msgs = await collect([wire.slice(0, 9), wire.slice(9, 31), wire.slice(31)]);
    expect(msgs).toEqual([
      { event: "start", data: { trace_id: "t1" } },
      { event: "progress", data: { node: "plan" } },
    ]);
  });

  it("handles CRLF, multi-line data and a trailing event without a blank line", async () => {
    const msgs = await collect(['event: result\r\ndata: {"a":\r\ndata: 1}\r\n\r\n', "event: error\ndata: boom"]);
    expect(msgs[0]).toEqual({ event: "result", data: { a: 1 } });
    expect(msgs[1]).toEqual({ event: "error", data: "boom" });
  });

  it("ignores comment/keep-alive blocks", async () => {
    expect(await collect([": ping\n\n"])).toEqual([]);
  });
});

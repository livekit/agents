import { afterEach, describe, expect, it, vi } from "vitest";
import { A2AClient, cutEvents, interpret } from "./a2a";

const task = { task: { id: "AST_1", contextId: "AT_1", status: { state: "TASK_STATE_SUBMITTED" as const } } };
const ending = { statusUpdate: { taskId: "AST_1", contextId: "AT_1", status: { state: "TASK_STATE_COMPLETED" } } };
const frame = (result: unknown) => JSON.stringify({ jsonrpc: "2.0", id: "request", result });
const artifact = { artifactUpdate: { taskId: "AST_1", contextId: "AT_1", artifact: { artifactId: "a", parts: [{ text: "café 🙂" }] }, append: true } };
const wire = `: heartbeat\r\ndata:${frame(task)}\r\n\r\ndata:${frame(artifact)}\r\n\r\ndata: ${frame(ending)}\r\n\r\n`;
afterEach(() => vi.unstubAllGlobals());

function response(chunks: Uint8Array[], cancel = vi.fn()) {
  return new Response(new ReadableStream({
    start(controller) { for (const chunk of chunks) controller.enqueue(chunk); },
    cancel,
  }), { headers: { "content-type": "text/event-stream" } });
}
async function collect(client: A2AClient, signal?: AbortSignal) {
  const frames = [];
  for await (const event of client.stream("message/stream", { message: { messageId: "same-message" } }, signal)) frames.push(event);
  return frames;
}

describe("browser A2A transport", () => {
  it("preserves SSE multiline whitespace and CR, LF, CRLF boundaries", () => {
    for (const newline of ["\n", "\r", "\r\n"]) {
      const parsed = cutEvents(`: comment${newline}data:  first ${newline}data:second${newline}${newline}tail`);
      expect(parsed.events).toEqual([" first \nsecond"]);
      expect(parsed.rest).toBe("tail");
    }
  });
  it("uses an authenticated POST and delivers UTF-8 replies across every byte split", async () => {
    for (const newline of ["\r\n", "\n", "\r"]) {
    const bytes = new TextEncoder().encode(wire.replaceAll("\r\n", newline));
    for (let split = 0; split <= bytes.length; split++) {
      const cancel = vi.fn();
      const fetcher = vi.fn<typeof fetch>().mockResolvedValue(response([bytes.slice(0, split), bytes.slice(split)], cancel));
      vi.stubGlobal("fetch", fetcher);
      const result = await collect(new A2AClient("https://agent", "scoped"));
      expect(result).toHaveLength(3);
      expect(interpret(result[1])).toEqual([{ type: "artifact", text: "café 🙂", append: true }]);
      expect(cancel).toHaveBeenCalledOnce();
      const options = fetcher.mock.calls[0][1];
      expect(options?.method).toBe("POST");
      expect(options?.headers).toMatchObject({ "A2A-Version": "1.0", Authorization: "Bearer scoped" });
      expect(options?.headers).not.toHaveProperty("A2A-Extensions");
      expect(options?.headers).not.toHaveProperty("X-LiveKit-Session-Id");
    }
    }
  });
  it("reports premature EOF and malformed records and releases the reader", async () => {
    for (const input of [frame(task), '{"result":{"task":{}}}', 'not JSON']) {
      const cancel = vi.fn();
      const stream = response([new TextEncoder().encode(`data:${input}\n\n`)], cancel);
      // EOF is intentionally delivered after a submitted task.
      const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(`data:${input}\n\n`)); c.close(); }, cancel });
      vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { headers: stream.headers })));
      await expect(collect(new A2AClient("https://agent", "scoped"))).rejects.toThrow();
      expect(body.locked).toBe(false);
    }
  });
  it("rejects malformed agent payloads before displaying or completing a reply", async () => {
    for (const payload of [
      { task: { id: "AST_1", status: { state: "TASK_STATE_COMPLETED" } } },
      { statusUpdate: { taskId: "AST_1", contextId: "AT_1", status: {} } },
      { artifactUpdate: { taskId: "AST_1", contextId: "AT_1", artifact: { artifactId: "a", parts: [{ text: 42 }] } } },
      { ...task, ...ending },
    ]) {
      const cancel = vi.fn();
      const stream = response([new TextEncoder().encode(`data:${frame(payload)}\n\n`)], cancel);
      vi.stubGlobal("fetch", vi.fn(async () => stream));
      await expect(collect(new A2AClient("https://agent", "scoped"))).rejects.toThrow();
      expect(cancel).toHaveBeenCalledOnce();
      expect(stream.body?.locked).toBe(false);
    }
  });
  it("aborts a pending read and cancels a reader when iteration stops", async () => {
    const cancel = vi.fn();
    const stream = response([new TextEncoder().encode(`data:${frame(task)}\n\n`)], cancel);
    vi.stubGlobal("fetch", vi.fn(async () => stream));
    const controller = new AbortController();
    const iterator = new A2AClient("https://agent", "scoped").stream("message/stream", {}, controller.signal);
    await iterator.next();
    const pending = iterator.next();
    controller.abort();
    await expect(pending).rejects.toThrow();
    expect(cancel).toHaveBeenCalledOnce();
    expect(stream.body?.locked).toBe(false);
    const stopped = response([new TextEncoder().encode(`data:${frame(task)}\n\n`)], cancel);
    vi.stubGlobal("fetch", vi.fn(async () => stopped));
    for await (const value of new A2AClient("https://agent", "scoped").stream("message/stream", {})) { expect(value).toBeTruthy(); break; }
    expect(stopped.body?.locked).toBe(false);
  });
  it("keeps HTTP authorization failures distinct from task failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ error: "denied" }, { status: 403 })));
    await expect(collect(new A2AClient("https://agent", "scoped"))).rejects.toThrow(/403/);
    expect(() => new A2AClient("https://agent", "")).toThrow();
  });
  it("does not finish a submitted task, and finishes replayed completed tasks", () => {
    expect(interpret({ result: task }).some((e) => e.type === "ended")).toBe(false);
    expect(interpret({ result: { task: { ...task.task, status: { state: "TASK_STATE_COMPLETED" }, artifacts: [{ artifactId: "a", parts: [{ text: "Done" }] }] } } })).toContainEqual({ type: "ended", taskId: "AST_1", state: "TASK_STATE_COMPLETED", meta: {} });
  });
});

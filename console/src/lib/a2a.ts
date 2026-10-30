/**
 * The wire to the agent's A2A endpoint: JSON-RPC over fetch and SSE.
 * The grant is an ordinary LiveKit token: the node verifies it in
 * Authorization, strips it, and asserts the grant it carried in the preamble
 * the worker reads; the worker is what enforces it.
 *
 * No library: this is what the minimal browser client of plan 10.1 demo 2
 * is, kept DOM-free so the frame interpretation is unit-tested.
 */

export const EXT = "https://livekit.io/a2a/ext/agents/v1";
export const key = (name: string) => `${EXT}/${name}`;


export const WIRE_TERMINAL = new Set([
  "TASK_STATE_COMPLETED",
  "TASK_STATE_CANCELED",
  "TASK_STATE_FAILED",
  "TASK_STATE_REJECTED",
]);

type Part = { kind?: string; text?: string };
type Message = { messageId?: string; role?: string; parts?: Part[]; metadata?: Record<string, unknown> };
type Status = { state: string; message?: Message };
type Task = { id: string; contextId: string; status: Status; artifacts?: { parts: Part[] }[] };

export type Frame =
  | { error: { code: number; message: string } }
  | {
      result: {
        task?: Task;
        statusUpdate?: { taskId: string; status: Status; final?: boolean; metadata?: Record<string, unknown> };
        artifactUpdate?: { taskId: string; artifact: { parts: Part[] }; append?: boolean; lastChunk?: boolean };
      };
    };

export type WireEvent =
  | { type: "task"; taskId: string; contextId: string; state: string; artifactText?: string }
  | { type: "progress"; text: string }
  | { type: "artifact"; text: string; append: boolean }
  | { type: "question"; text: string }
  | { type: "ended"; taskId: string; state: string; meta: Record<string, unknown> }
  | { type: "error"; code: number; message: string };

const partsText = (parts: Part[] | undefined) =>
  (parts ?? []).map((part) => part.text ?? "").join("");

/** The `livekit/*` entries of a status update's metadata, short keys. */
export function extensionMeta(meta: Record<string, unknown> | undefined) {
  const out: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(meta ?? {}))
    if (name.startsWith(EXT)) out[name.slice(EXT.length + 1)] = value;
  return out;
}

/**
 * One frame, as the events the stage reacts to. Stateless: the turn keeps
 * the reply it is assembling.
 */
export function interpret(frame: Frame): WireEvent[] {
  if ("error" in frame) return [{ type: "error", code: frame.error.code, message: frame.error.message }];
  const r = frame.result;
  const out: WireEvent[] = [];
  if (r.task) {
    const state = r.task.status.state;
    const terminal = WIRE_TERMINAL.has(state);
    out.push({
      type: "task",
      taskId: r.task.id,
      contextId: r.task.contextId,
      state,
      /* A replayed terminal task carries its answer as an artifact. */
      artifactText:
        terminal && r.task.artifacts
          ? r.task.artifacts.map((artifact) => partsText(artifact.parts)).join("")
          : undefined,
    });
  }
  if (r.statusUpdate) {
    const update = r.statusUpdate;
    const state = update.status.state;
    if (state === "TASK_STATE_WORKING" && update.status.message)
      out.push({ type: "progress", text: partsText(update.status.message.parts) });
    if (update.final || state !== "TASK_STATE_WORKING") {
      /* A tool waiting on the person (plan 7.4): the question rides on the
         status, not an artifact. */
      if (state === "TASK_STATE_INPUT_REQUIRED" && update.status.message)
        out.push({ type: "question", text: partsText(update.status.message.parts) });
      out.push({ type: "ended", taskId: update.taskId, state, meta: extensionMeta(update.metadata) });
    }
  }
  if (r.artifactUpdate)
    out.push({
      type: "artifact",
      text: partsText(r.artifactUpdate.artifact.parts),
      append: Boolean(r.artifactUpdate.append),
    });
  return out;
}

/** Cuts complete SSE events off a buffer: their `data:` payloads, and the rest. */
export function cutEvents(buffer: string): { events: string[]; rest: string } {
  const events: string[] = [];
  let index: number;
  while ((index = buffer.indexOf("\n\n")) >= 0) {
    const event = buffer.slice(0, index);
    buffer = buffer.slice(index + 2);
    const data = event
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice(6))
      .join("");
    if (data) events.push(data);
  }
  return { events, rest: buffer };
}

export const uuid = () => `msg_${Math.random().toString(36).slice(2, 12)}`;

/** One client per grant: the endpoint and the token. */
export class A2AClient {
  constructor(
    readonly endpoint: string,
    readonly token: string,
  ) {}

  private headers(sessionId: string | null) {
    return {
      "Content-Type": "application/json",
      "A2A-Extensions": EXT,
      Authorization: `Bearer ${this.token}`,
      ...(sessionId ? { "X-LiveKit-Session-Id": sessionId } : {}),
    };
  }

  async *stream(
    method: string,
    params: unknown,
    sessionId: string | null,
  ): AsyncGenerator<Frame> {
    const id = uuid();
    const response = await fetch(`${this.endpoint}/v1`, {
      method: "POST",
      headers: this.headers(sessionId),
      body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
    });
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
    if (!response.headers.get("content-type")?.startsWith("text/event-stream")) {
      yield (await response.json()) as Frame;
      return;
    }
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const cut = cutEvents(buffer);
      buffer = cut.rest;
      for (const data of cut.events) yield JSON.parse(data) as Frame;
    }
  }


}

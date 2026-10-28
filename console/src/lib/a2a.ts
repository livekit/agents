import { record } from "./decode";

/** A2A JSON fields consumed by the conversation and replay views. */
export type Part = { text?: string; data?: unknown };
export type Message = {
  messageId: string;
  contextId?: string;
  taskId?: string;
  role: "ROLE_UNSPECIFIED" | "ROLE_USER" | "ROLE_AGENT";
  parts: Part[];
  referenceTaskIds?: string[];
};
export type Artifact = { artifactId: string; parts: Part[] };
type TaskStatus = { state: string; message?: Message };
export type StreamResponse = {
  task?: {
    id: string;
    contextId?: string;
    status: TaskStatus;
    artifacts?: Artifact[];
    metadata?: Record<string, unknown>;
  };
  message?: Message;
  statusUpdate?: {
    taskId: string;
    contextId: string;
    status: TaskStatus;
    metadata?: Record<string, unknown>;
  };
  artifactUpdate?: { taskId: string; contextId: string; artifact: Artifact; append?: boolean };
};
export type AgentCard = {
  name: string;
  description: string;
  version?: string;
  capabilities?: { streaming?: boolean };
  supportedInterfaces?: { url: string; protocolBinding: string; protocolVersion: string }[];
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  skills?: unknown[];
};

function parts(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.every(
      (part) => record(part) && (part.text === undefined || typeof part.text === "string"),
    )
  );
}
export function decodeMessage(value: unknown): Message {
  if (
    !record(value) ||
    typeof value.messageId !== "string" ||
    !parts(value.parts) ||
    !["ROLE_UNSPECIFIED", "ROLE_USER", "ROLE_AGENT"].includes(String(value.role)) ||
    [value.taskId, value.contextId].some((id) => id !== undefined && typeof id !== "string") ||
    (value.referenceTaskIds !== undefined &&
      (!Array.isArray(value.referenceTaskIds) ||
        !value.referenceTaskIds.every((id) => typeof id === "string")))
  )
    throw new Error("The saved message could not be read");
  return value as Message;
}
export function decodeCard(value: unknown): AgentCard {
  if (!record(value) || typeof value.name !== "string" || typeof value.description !== "string")
    throw new Error("The agent's details could not be read");
  return value as AgentCard;
}
function status(value: unknown): boolean {
  if (!record(value) || typeof value.state !== "string") return false;
  if (value.message !== undefined) decodeMessage(value.message);
  return true;
}
function artifact(value: unknown): boolean {
  return record(value) && typeof value.artifactId === "string" && parts(value.parts);
}
function streamResponse(value: unknown): StreamResponse {
  if (
    !record(value) ||
    [value.task, value.message, value.statusUpdate, value.artifactUpdate].filter(
      (item) => item !== undefined,
    ).length !== 1
  )
    throw new Error("The agent's reply could not be read");
  const task = value.task;
  if (
    task !== undefined &&
    (!record(task) ||
      typeof task.id !== "string" ||
      typeof task.contextId !== "string" ||
      !task.contextId ||
      !status(task.status) ||
      (task.artifacts !== undefined &&
        (!Array.isArray(task.artifacts) || !task.artifacts.every(artifact))))
  )
    throw new Error("The agent's task could not be read");
  if (value.message !== undefined) decodeMessage(value.message);
  for (const update of [value.statusUpdate, value.artifactUpdate]) {
    if (update === undefined) continue;
    if (
      !record(update) ||
      typeof update.taskId !== "string" ||
      typeof update.contextId !== "string"
    )
      throw new Error("The agent's update could not be read");
  }
  if (record(value.statusUpdate) && !status(value.statusUpdate.status))
    throw new Error("The agent's status could not be read");
  if (
    record(value.artifactUpdate) &&
    (!artifact(value.artifactUpdate.artifact) ||
      (value.artifactUpdate.append !== undefined &&
        typeof value.artifactUpdate.append !== "boolean"))
  )
    throw new Error("The agent's answer could not be read");
  return value as StreamResponse;
}
export const EXT = "https://livekit.io/a2a/ext/agents/v1";
export const key = (name: string) => `${EXT}/${name}`;


export const WIRE_TERMINAL = new Set([
  "TASK_STATE_COMPLETED",
  "TASK_STATE_CANCELED",
  "TASK_STATE_FAILED",
  "TASK_STATE_REJECTED",
]);

export type Frame =
  | { error: { code: number; message: string } }
  | { result: StreamResponse; cursor?: string };

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
    if (name.startsWith(`${EXT}/`)) out[name.slice(EXT.length + 1)] = value;
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
      contextId: r.task.contextId ?? "",
      state,
      /* A replayed terminal task carries its answer as an artifact. */
      artifactText:
        terminal && r.task.artifacts
          ? r.task.artifacts.map((artifact) => partsText(artifact.parts)).join("")
          : undefined,
    });
    if (terminal || state === "TASK_STATE_INPUT_REQUIRED" || state === "TASK_STATE_AUTH_REQUIRED") {
      if (state === "TASK_STATE_INPUT_REQUIRED" && r.task.status.message)
        out.push({ type: "question", text: partsText(r.task.status.message.parts) });
      out.push({ type: "ended", taskId: r.task.id, state, meta: extensionMeta(r.task.metadata) });
    }
  }
  if (r.statusUpdate) {
    const update = r.statusUpdate;
    const state = update.status.state;
    if (state === "TASK_STATE_WORKING" && update.status.message)
      out.push({ type: "progress", text: partsText(update.status.message.parts) });
    if (WIRE_TERMINAL.has(state) || state === "TASK_STATE_INPUT_REQUIRED" || state === "TASK_STATE_AUTH_REQUIRED") {
      /* Questions are status messages, separate from answer artifacts. */
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

/** Consume complete SSE records without normalizing significant data whitespace. */
export function cutEvents(buffer: string): { events: string[]; rest: string; cursors: (string | undefined)[] } {
  const events: string[] = [];
  const cursors: (string | undefined)[] = [];
  let start = 0;
  let at = 0;
  let data: string[] = [];
  let cursor: string | undefined;
  while (at < buffer.length) {
    let end = at;
    while (end < buffer.length && buffer[end] !== "\r" && buffer[end] !== "\n") end++;
    if (end === buffer.length) break;
    const line = buffer.slice(at, end);
    at = end + (buffer[end] === "\r" && buffer[end + 1] === "\n" ? 2 : 1);
    if (!line) {
      if (data.length) { events.push(data.join("\n")); cursors.push(cursor); }
      start = at;
      data = [];
      cursor = undefined;
      continue;
    }
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") data.push(value);
    if (field === "id" && !value.includes("\0")) cursor = value;
  }
  return { events, cursors, rest: buffer.slice(start) };
}

export const uuid = () => `msg_${crypto.randomUUID()}`;

function decodeFrame(value: unknown, cursor?: string): Frame {
  if (!record(value) || value.jsonrpc !== "2.0" || !(value.id === null || typeof value.id === "string" || typeof value.id === "number") || (("error" in value) === ("result" in value))) throw new Error("The agent's reply could not be read");
  if ("error" in value) {
    if (!record(value.error) || typeof value.error.code !== "number" || typeof value.error.message !== "string")
      throw new Error("The agent's error could not be read");
    return { error: { code: value.error.code, message: value.error.message } };
  }
  const result = streamResponse(value.result);
  return { result, cursor };
}

export class A2AClient {
  private controllers = new Set<AbortController>();
  constructor(readonly endpoint: string, readonly token: string) {
    if (!token.trim()) throw new Error("A bearer token is required");
  }
  close() {
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
  }
  async *stream(method: string, params: unknown, signal?: AbortSignal): AsyncGenerator<Frame> {
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    this.controllers.add(controller);
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    const cancelRead = () => { void reader?.cancel().catch(() => undefined); };
    controller.signal.addEventListener("abort", cancelRead, { once: true });
    try {
      const response = await fetch(`${this.endpoint}/v1`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream, application/json", "A2A-Version": "1.0", Authorization: `Bearer ${this.token}` },
        body: JSON.stringify({ jsonrpc: "2.0", id: uuid(), method, params }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Agent request failed (HTTP ${response.status})`);
      if (!response.headers.get("content-type")?.startsWith("text/event-stream")) {
        const frame = decodeFrame(await response.json());
        yield frame;
        if (!interpret(frame).some((event) => event.type === "ended" || event.type === "error"))
          throw new Error("Reply ended before the task finished; resume the reply");
        return;
      }
      if (!response.body) throw new Error("Agent returned no response stream");
      reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8", { fatal: true });
      let buffer = "";
      for (;;) {
        controller.signal.throwIfAborted();
        const { value, done } = await reader.read();
        controller.signal.throwIfAborted();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        // Commit complete records immediately, including a final bare CR. A leading
        // LF in the next chunk is an empty record and cannot duplicate a delivery.
        const cut = cutEvents(buffer);
        buffer = cut.rest;
        for (let index = 0; index < cut.events.length; index++) {
          const frame = decodeFrame(JSON.parse(cut.events[index]), cut.cursors[index]);
          yield frame;
          if (interpret(frame).some((event) => event.type === "ended" || event.type === "error")) return;
        }
        if (done) throw new Error("Reply ended before the task finished; resume the reply");
      }
    } finally {
      signal?.removeEventListener("abort", abort);
      controller.signal.removeEventListener("abort", cancelRead);
      await reader?.cancel().catch(() => undefined);
      reader?.releaseLock();
      controller.abort();
      this.controllers.delete(controller);
    }
  }
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Empty, Json, Panel } from "../chrome/panel";
import { Icon } from "../chrome/icon";
import { streamUrl } from "@/lib/api";
import { short, text } from "@/lib/format";

type StreamFrame = { index: number; summary: string; raw: unknown };
type StreamState = "idle" | "connecting" | "live" | "closed" | "failed";

/**
 * One line of a frame: an A2A event (status, artifact, message, task) or
 * the keys of anything else.
 */
export function summarize(frame: unknown): string {
  if (typeof frame !== "object" || frame === null) return String(frame);
  const f = frame as Record<string, unknown>;
  const r = (f.result ?? f) as Record<string, unknown>;
  const su = r.statusUpdate as { status?: { state?: string; message?: unknown }; final?: boolean } | undefined;
  if (su || r.kind === "status-update") {
    const status = (su?.status ?? (r.status as { state?: string; message?: unknown })) ?? {};
    return `status ${(status.state ?? "").replace("TASK_STATE_", "").toLowerCase()} ${status.message ? text(status.message) : ""}${
      su?.final || r.final ? " (final)" : ""
    }`.trim();
  }
  const au = r.artifactUpdate as { artifact?: unknown; append?: boolean } | undefined;
  if (au || r.kind === "artifact-update") return `artifact ${au?.append ? "+" : ""}${text(au?.artifact ?? r.artifact).slice(0, 200)}`;
  const task = r.task as { status?: { state?: string } } | undefined;
  if (task || r.kind === "task") return `task ${(task?.status?.state ?? (r.status as { state?: string })?.state ?? "").replace("TASK_STATE_", "").toLowerCase()}`;
  if (r.kind === "message" || r.parts) return `${String(r.role ?? "message")} ${text(r).slice(0, 200)}`;
  return Object.keys(r).join(", ");
}

/** Decodes one `data` event: a JSON array of frames, or base64 length-prefixed records. */
export function decodeData(data: string): unknown[] {
  try {
    const parsed = JSON.parse(data) as unknown;
    if (Array.isArray(parsed)) return parsed;
  } catch {
    /* not JSON: length-prefixed records follow */
  }
  try {
    const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
    const decoder = new TextDecoder();
    const out: unknown[] = [];
    let i = 0;
    while (i + 4 <= bytes.length) {
      const length = ((bytes[i] << 24) | (bytes[i + 1] << 16) | (bytes[i + 2] << 8) | bytes[i + 3]) >>> 0;
      i += 4;
      const chunk = decoder.decode(bytes.slice(i, i + length));
      i += length;
      try {
        out.push(JSON.parse(chunk));
      } catch {
        out.push(chunk);
      }
    }
    return out;
  } catch {
    return [data];
  }
}

/**
 * The selected task's answer stream, replayed from the start and followed
 * live through the debug API's SSE relay. A terminal task's stream is
 * closed; a running one stays open and the frames keep coming.
 */
type Replay = { key: string; frames: StreamFrame[]; state: StreamState; note: string; offset: string };
const fresh = (key: string, state: StreamState): Replay => ({ key, frames: [], state, note: "", offset: "" });

export function StreamPanel({ taskId, available }: { taskId: string | null; available: boolean }) {
  const [follow, setFollow] = useState(true);
  const [run, setRun] = useState(0);
  /* One replay per task and run, keyed, so a new selection starts clean
     without an effect resetting state. */
  const key = taskId ? `${taskId}:${run}` : "";
  const [replayState, setReplay] = useState<Replay>(() => fresh("", "idle"));
  const list = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!taskId || !available) return;
    const update = (change: (current: Replay) => Partial<Replay>) =>
      setReplay((current) => {
        const base = current.key === key ? current : fresh(key, "connecting");
        return { ...base, ...change(base) };
      });
    let count = 0;
    const source = new EventSource(streamUrl(taskId));
    source.onopen = () => update(() => ({ state: "live" }));
    source.addEventListener("data", (event) => {
      /* Numbered here, not in the updater: React may run an updater twice. */
      const decoded = decodeData((event as MessageEvent).data as string).map((raw) => ({
        index: ++count,
        summary: summarize(raw),
        raw,
      }));
      update((current) => ({ frames: [...current.frames, ...decoded] }));
    });
    let closed = false;
    source.addEventListener("control", (event) => {
      let control: { streamNextOffset?: string; streamClosed?: boolean; upToDate?: boolean; error?: string } = {};
      try {
        control = JSON.parse((event as MessageEvent).data as string);
      } catch {
        /* an unreadable control frame changes nothing */
      }
      /* A closed stream ends the follow here: left open, EventSource would
         reconnect when the relay returns and replay every frame again. */
      if (control.streamClosed) {
        closed = true;
        source.close();
      }
      update((current) => {
        const next: Partial<Replay> = {};
        if (control.streamNextOffset) next.offset = control.streamNextOffset;
        if (control.error) {
          next.note = control.error;
          next.state = "failed";
        }
        if (control.streamClosed) next.state = "closed";
        else if (control.upToDate) next.note = current.note || "up to date";
        return next;
      });
    });
    source.onerror = () => {
      if (closed) return;
      update(() => (source.readyState === EventSource.CLOSED ? { state: "failed", note: "not available" } : { note: "reconnecting" }));
    };
    return () => source.close();
  }, [taskId, available, run, key]);

  const replayed = replayState.key === key && key ? replayState : fresh(key, taskId && available ? "connecting" : "idle");
  const { frames, state, note, offset } = replayed;

  useEffect(() => {
    if (!follow) return;
    const element = list.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [frames, follow]);

  const replay = useCallback(() => setRun((n) => n + 1), []);

  return (
    <Panel
      id="stream"
      title="Stream"
      meta={
        taskId ? (
          <span className="inline-flex items-center gap-2">
            <span title={taskId}>{short(taskId)}</span>
            <span data-stream-state={state} className={state === "failed" ? "text-failed-text" : undefined}>
              {state}
            </span>
            <span data-stream-frames={frames.length} title={offset ? `next offset ${offset}` : undefined}>
              {frames.length} frames
            </span>
            {note ? <span>{note}</span> : null}
          </span>
        ) : undefined
      }
      actions={
        taskId ? (
          <>
            <button
              type="button"
              className="btn btnQuiet h-row-sm px-2"
              aria-pressed={follow}
              title="Follow the newest frame"
              onClick={() => setFollow((value) => !value)}
            >
              follow
            </button>
            <button type="button" className="btn btnQuiet h-row-sm px-2" title="Replay from the start" onClick={replay} data-action="replay">
              <Icon name="replay" size={12} />
              replay
            </button>
          </>
        ) : undefined
      }
    >
      {!available ? (
        <Empty>Streams are not available on this agent-db.</Empty>
      ) : !taskId ? (
        <Empty>Select a task in Tasks, or open one from the id under a turn, to replay its stream here.</Empty>
      ) : frames.length === 0 ? (
        <Empty>{state === "connecting" ? "Connecting." : "No frames yet."}</Empty>
      ) : (
        <div ref={list} className="h-full overflow-auto px-3 py-2">
          {/* A timeline: a line down the left, a dot per frame, the last one filled while the stream is live. */}
          {frames.map((frame, index) => (
            <div
              key={frame.index}
              className="grid grid-cols-[var(--pad-4)_var(--w-index)_minmax(0,1fr)] gap-2 py-1"
              data-frame={frame.index}
            >
              <span className="relative" aria-hidden="true">
                <span
                  className="absolute top-0 bottom-0 left-1/2 w-px bg-line data-[first=true]:top-1/2 data-[last=true]:bottom-1/2"
                  data-first={index === 0 || undefined}
                  data-last={index === frames.length - 1 || undefined}
                />
                <span
                  className="absolute top-1/2 left-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-control data-[live=true]:bg-accent"
                  data-live={(index === frames.length - 1 && state === "live") || undefined}
                />
              </span>
              <span className="meta pt-px">{frame.index}</span>
              <span className="grid min-w-0 gap-px">
                <span className="truncate font-mono text-micro text-ink">{frame.summary}</span>
                <Json value={frame.raw} label="frame" />
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

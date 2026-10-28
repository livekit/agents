"use client";

import { useEffect, useRef, useState } from "react";
import { Empty, Json, Panel } from "../chrome/panel";
import { A2AClient, interpret, type Frame } from "@/lib/a2a";
import type { Grant } from "@/lib/types";

export function summarize(frame: Frame): string {
  return interpret(frame).map((event) => {
    switch (event.type) {
      case "error": return event.message;
      case "progress": case "question": case "artifact": return `${event.type}: ${event.text}`;
      case "task": case "ended": return `${event.type}: ${event.state}`;
    }
  }).join("; ");
}

type Replay = { key: string; frames: { index: number; raw: Frame }[]; state: string; note: string };
const fresh = (key: string): Replay => ({ key, frames: [], state: "connecting", note: "" });

/** Task replay uses the same authorized, cursor-aware POST transport as chat. */
export function StreamPanel({ taskId, sessionId, grant }: { taskId: string | null; sessionId: string | null; grant: Grant | null }) {
  const [run, setRun] = useState(0);
  const key = `${sessionId}:${taskId}:${run}`;
  const [replay, setReplay] = useState<Replay>(() => fresh(""));
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!taskId || !sessionId || !grant) return;
    const client = new A2AClient(grant.endpoint, grant.token);
    const controller = new AbortController();
    let index = 0;
    void (async () => {
      try {
        for await (const raw of client.stream("tasks/resubscribe", { id: taskId, contextId: sessionId }, controller.signal)) {
          const frame = { index: ++index, raw };
          setReplay((current) => {
            const base = current.key === key ? current : fresh(key);
            return { ...base, frames: [...base.frames, frame].slice(-500), state: "live" };
          });
        }
        if (!controller.signal.aborted) setReplay((current) => ({ ...current, state: "closed" }));
      } catch (cause) {
        if (!controller.signal.aborted) setReplay((current) => ({ ...(current.key === key ? current : fresh(key)), state: "failed", note: cause instanceof Error ? cause.message : "Replay unavailable" }));
      }
    })();
    return () => { controller.abort(); client.close(); };
  }, [key, taskId, sessionId, grant]);
  const current = replay.key === key ? replay : fresh(key);
  return (
    <Panel id="stream" title="Stream" meta={`${current.state} · ${current.frames.length} frames`} actions={<button className="btn btnQuiet" onClick={() => setRun((n) => n + 1)}>Replay</button>}>
      {!taskId ? <Empty>Select a task to replay its reply.</Empty> : current.note ? <Empty>{current.note}</Empty> : (
        <div ref={list} className="h-full overflow-auto px-3 py-2">
          {current.frames.map((frame) => <div key={frame.index} data-frame={frame.index} className="py-1"><span className="font-mono text-micro">{frame.index} {summarize(frame.raw)}</span><Json value={frame.raw} label="frame" /></div>)}
        </div>
      )}
    </Panel>
  );
}

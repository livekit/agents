"use client";

import { useEffect, useState } from "react";
import { watchSession } from "./api";
import { text } from "./format";
import type { Progress, SessionDump } from "./types";

/** What the dumps said about one task over time: when it was first seen, and every status message, in order. */
export type TaskLog = { firstSeen: number; log: Progress[] };

/** The next log, with what a dump adds: a task seen for the first time, a status message not seen before. */
export function fold(current: Map<string, TaskLog>, dump: SessionDump, at: number): Map<string, TaskLog> {
  const tasks = dump.database?.tasks ?? [];
  let next: Map<string, TaskLog> | null = null;
  for (const task of tasks) {
    const known = current.get(task.task_id);
    const message = text(task.status_message).trim();
    if (known && (!message || known.log.some((line) => line.text === message))) continue;
    next ??= new Map(current);
    const entry = known ? { ...known, log: [...known.log] } : { firstSeen: at, log: [] };
    if (message) entry.log.push({ at, text: message });
    next.set(task.task_id, entry);
  }
  return next ?? current;
}

/**
 * The selected session's database, live: the debug API pushes a dump every
 * two seconds over SSE. `skew` is the server's clock minus ours, so lease
 * countdowns and "ago" labels read on the server's time.
 */
export function useSession(sessionId: string | null) {
  /* Keyed by the session it came from, so switching sessions shows nothing
     stale without an effect resetting state. */
  const [latest, setLatest] = useState<{ sessionId: string; dump: SessionDump | null; error: string | null }>({
    sessionId: "",
    dump: null,
    error: null,
  });
  const [skew, setSkew] = useState(0);
  /* Keyed by session as well: another session's tasks are its own. */
  const [logs, setLogs] = useState<{ sessionId: string; tasks: Map<string, TaskLog> }>({ sessionId: "", tasks: new Map() });

  useEffect(() => {
    if (!sessionId) return;
    return watchSession(
      sessionId,
      (next) => {
        setSkew(next.fetchedAt - Date.now());
        setLatest({ sessionId, dump: next, error: null });
        setLogs((current) => {
          const base = current.sessionId === sessionId ? current.tasks : new Map<string, TaskLog>();
          const tasks = fold(base, next, Date.now());
          return tasks === base && current.sessionId === sessionId ? current : { sessionId, tasks };
        });
      },
      (error) => setLatest((current) => ({ sessionId, dump: current.sessionId === sessionId ? current.dump : null, error })),
    );
  }, [sessionId]);

  const current = sessionId && latest.sessionId === sessionId ? latest : null;
  const taskLog = sessionId && logs.sessionId === sessionId ? logs.tasks : EMPTY_LOG;
  return { dump: current?.dump ?? null, skew, watchError: current?.error ?? null, taskLog };
}

const EMPTY_LOG: Map<string, TaskLog> = new Map();

/** A clock that ticks four times a second, for the countdowns between dumps. */
export function useNow(skew: number, every = 250) {
  const [now, setNow] = useState(() => Date.now() + skew);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() + skew), every);
    return () => window.clearInterval(timer);
  }, [every, skew]);
  return now;
}

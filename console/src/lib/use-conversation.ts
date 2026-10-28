"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { A2AClient, decodeCard, decodeMessage, interpret, uuid } from "./a2a";
import {
  HttpError,
  getConfig,
  getGrant,
  getHistory,
  listSessions,
  nameSession,
} from "./api";
import { listExamples, sameExample, type Example } from "./examples";
import type { Message } from "./a2a";
import { text } from "./format";
import type {
  Entry,
  Grant,
  SessionRow,
  ConsoleConfig,
  Timing,
} from "./types";

export const SESSION_KEY = "console.session";
const PENDING_KEY = "console.pending";
const QUESTION_KEY = "console.question.";

type Pending = { taskId?: string; sessionId?: string; message: Message; cursor?: string; answer?: string };
function pendingFrom(value: string | null): Pending | null {
  if (!value) return null;
  try {
    const raw: unknown = JSON.parse(value);
    if (!raw || typeof raw !== "object" || !("message" in raw)) return null;
    return {
      message: decodeMessage(raw.message),
      taskId: "taskId" in raw && typeof raw.taskId === "string" ? raw.taskId : undefined,
      sessionId: "sessionId" in raw && typeof raw.sessionId === "string" ? raw.sessionId : undefined,
      cursor: "cursor" in raw && typeof raw.cursor === "string" ? raw.cursor : undefined,
      answer: "answer" in raw && typeof raw.answer === "string" ? raw.answer : undefined,
    };
  } catch { return null; }
}
/** An entry before it is given its id: a distributive Omit, so each kind keeps its fields. */
type Draft = Entry extends infer E
  ? E extends Entry
    ? Omit<E, "id" | "at">
    : never
  : never;

const read = (name: string) =>
  typeof window === "undefined" ? null : window.localStorage.getItem(name);
const write = (name: string, value: string | null) => {
  if (value === null) window.localStorage.removeItem(name);
  else window.localStorage.setItem(name, value);
};

let counter = 0;
const nextId = () => `e${++counter}`;

/**
 * The conversation with the agent, as the stage shows it, and what it needs
 * from outside: the grant and the endpoint from the debug API, the subject's
 * sessions from the registry through it, the agent card for its name. The conversation itself goes to the agent's endpoint directly.
 *
 * Persisted across reloads: the session id, and a running task so the reply resumes through tasks/resubscribe.
 *
 * What goes wrong is told in plain words, once, where it belongs: a turn
 * that failed under the turn. A session that no longer exists is not a
 * fault at all: it is dropped and the page starts a new conversation,
 * quietly, and `onSessionLost` lets the room close what it had opened for it.
 */
export function useConversation({
  onSessionLost,
}: { onSessionLost?: () => void } = {}) {
  const [config, setConfig] = useState<ConsoleConfig | null>(null);
  const [grant, setGrant] = useState<Grant | null>(null);
  /** Every example the API knows, and the one this conversation talks to. */
  const [examples, setExamples] = useState<Example[]>([]);
  const [example, setExample] = useState<Example | null>(null);
  const grantRef = useRef<Grant | null>(null);
  const [card, setCard] = useState<{
    name: string;
    description: string;
  } | null>(null);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);
  /** The debug API or the endpoint did not answer at boot: nothing can be sent. */
  const [unreachable, setUnreachable] = useState(false);
  /* The history of the session being opened is on its way: the stage keeps the conversation's layout. */
  const [restoring, setRestoring] = useState(false);
  /** The registry did not answer: the list is empty for that reason, not because there is nothing. */
  const [registryDown, setRegistryDown] = useState(false);
  const client = useRef<A2AClient | null>(null);
  const connection = useRef<AbortController | null>(null);
  const sessionRef = useRef<string | null>(null);
  /* Sessions known to carry a title, or asked for one: each is named once. */
  const titled = useRef(new Set<string>());
  const lost = useRef(onSessionLost);
  useEffect(() => {
    lost.current = onSessionLost;
  }, [onSessionLost]);

  /* The first thing the person said in this conversation: what a title is made from. */
  const firstUserText = useRef<string | null>(null);
  const append = useCallback((entry: Draft) => {
    const id = nextId();
    if (entry.kind === "user" && firstUserText.current === null)
      firstUserText.current = entry.text;
    setEntries((list) => [...list, { ...entry, id, at: Date.now() } as Entry]);
    return id;
  }, []);
  const patch = useCallback((id: string, update: (entry: Entry) => Entry) => {
    setEntries((list) =>
      list.map((entry) => (entry.id === id ? update(entry) : entry)),
    );
  }, []);

  const refreshSessions = useCallback(
    async (agent: string, endpoint: string, sub: string) => {
      try {
        const rows = await listSessions(agent, endpoint, sub);
        for (const row of rows)
          if (row.attributes?.title) titled.current.add(row.sessionId);
        setSessions(rows);
        setRegistryDown(false);
      } catch {
        setRegistryDown(true);
      }
    },
    [],
  );

  /**
   * Names a session after its first message, through the console's backend,
   * once; the list shows the title once written. Called after a text turn,
   * after a voice turn, and for any listed session still without a title.
   */
  const name = useCallback(
    async (sessionId: string, firstMessage: string) => {
      if (!grant || titled.current.has(sessionId)) return;
      titled.current.add(sessionId);
      try {
        await nameSession(sessionId, firstMessage);
      } catch {
        /* unnamed: the list falls back to the first message */
        titled.current.delete(sessionId);
        return;
      }
      await refreshSessions(grant.agent, grant.endpointName, grant.sub);
    },
    [grant, refreshSessions],
  );

  const setSession = useCallback(
    (id: string | null, { fresh = false } = {}) => {
      sessionRef.current = id;
      setSessionId(id);
      write(SESSION_KEY, id);
      setRestoring(false);
      if (!fresh) {
        /* Switching: one connection serves one session, and the log is the session's. */
        setEntries([]);
        firstUserText.current = null;
      }
    },
    [],
  );

  /** The committed history, from the session API under the grant. */
  const restore = useCallback(
    async (id: string) => {
      const a2a = client.current;
      if (!a2a) return;
      setEntries([]);
      setRestoring(true);
      try {
        const { items } = await getHistory(id, grantRef.current?.agent ?? "");
        const restored: Entry[] = [];
        for (const item of items) {
          if (item.type !== "message" || (item.role !== "user" && item.role !== "assistant")) continue;
          restored.push({
            kind: item.role === "user" ? "user" : "agent", id: nextId(),
            at: (item.created_at ?? Date.now() / 1000) * 1000, text: text(item.content),
          });
        }
        if (sessionRef.current !== id) return;
        firstUserText.current =
          restored.find((entry) => entry.kind === "user")?.text ?? null;
        setEntries(restored);
      } catch (cause) {
        if (sessionRef.current !== id) return;
        if (cause instanceof HttpError && cause.status === 404) {
          /* The conversation expired.
             Forget it and start a new conversation, quietly. */
          if (sessionRef.current === id) {
            setSession(null, { fresh: true });
            write(PENDING_KEY, null);
            lost.current?.();
          }
          return;
        }
        setEntries([
          {
            kind: "status",
            id: nextId(),
            at: Date.now(),
            text: "The conversation's history could not be loaded.",
          },
        ]);
      } finally {
        if (sessionRef.current === id) setRestoring(false);
      }
    },
    [setSession],
  );

  /** One turn's frames, from message/stream or tasks/resubscribe. */
  const consume = useCallback(
    async (
      frames: AsyncGenerator<Parameters<typeof interpret>[0]>,
      pending: Pending,
      signal: AbortSignal,
    ) => {
      let reply: string | null = pending.answer ? append({ kind: "agent", text: pending.answer, taskId: pending.taskId, streaming: true }) : null;
      let artifactText = pending.answer ?? "";
      let taskId: string | null = null;
      const sentAt = performance.now();
      const timing: Timing = {};
      const stamp = (mark: keyof Timing) => {
        if (timing[mark] === undefined)
          timing[mark] = Math.round(performance.now() - sentAt);
      };
      setBusy(true);
      try {
        for await (const frame of frames) {
          if (signal.aborted) return;
          if ("cursor" in frame && frame.cursor !== undefined) pending.cursor = frame.cursor;
          for (const event of interpret(frame)) {
            switch (event.type) {
              case "error":
                write(PENDING_KEY, null);
                append({
                  kind: "status",
                  text: `This turn failed: ${event.message}`,
                  taskId: taskId ?? undefined,
                  failed: true,
                });
                return;
              case "task":
                stamp("ack");
                taskId = event.taskId;
                pending.taskId = taskId;
                pending.sessionId = event.contextId;
                /* The first message made the session. */
                if (!sessionRef.current)
                  setSession(event.contextId, { fresh: true });
                write(PENDING_KEY, JSON.stringify(pending));
                if (event.artifactText !== undefined) {
                  artifactText = event.artifactText;
                  if (reply) patch(reply, (entry) => entry.kind === "agent" ? { ...entry, text: artifactText } : entry);
                }
                break;
              case "progress":
                append({ kind: "progress", text: event.text });
                break;
              case "artifact":
                if (event.text) stamp("first");
                if (!reply)
                  reply = append({
                    kind: "agent",
                    text: "",
                    taskId: taskId ?? undefined,
                    streaming: true,
                  });
                pending.answer = event.append ? (pending.answer ?? "") + event.text : event.text;
                patch(reply, (entry) =>
                  entry.kind === "agent"
                    ? {
                        ...entry,
                        text: pending.answer ?? "",
                        timing: { ...timing },
                      }
                    : entry,
                );
                break;
              case "question":
                append({ kind: "question", text: event.text });
                break;
              case "ended": {
                write(PENDING_KEY, null);
                if (pending.sessionId) write(QUESTION_KEY + pending.sessionId,
                  event.state === "TASK_STATE_INPUT_REQUIRED" ? event.taskId : null);
                stamp("done");
                if (artifactText && !reply)
                  reply = append({
                    kind: "agent",
                    text: artifactText,
                    taskId: event.taskId,
                  });
                if (reply)
                  patch(
                    reply,
                    (entry) =>
                      ({
                        ...entry,
                        streaming: false,
                        timing: { ...timing },
                      }) as Entry,
                  );
                const state = event.state.replace("TASK_STATE_", "").toLowerCase().replace(/_/g, "-");
                const labels: Record<string, string> = {
                  completed: "Reply complete", "input-required": "Waiting for your answer",
                  failed: "The agent could not finish this request", canceled: "Request stopped",
                  rejected: "The agent could not accept this request", "auth-required": "Sign in to continue",
                };
                append({
                  kind: "status",
                  text: labels[state] ?? "Reply complete",
                  taskId: event.taskId,
                  state,
                  failed:
                    state === "failed" ||
                    state === "canceled" ||
                    state === "rejected",
                });
                break;
              }
            }
          }
          if (!interpret(frame).some((event) => event.type === "ended" || event.type === "error"))
            write(PENDING_KEY, JSON.stringify(pending));
        }
      } catch (cause) {
        if (signal.aborted) return;
        append({
          kind: "status",
          text: `The reply did not arrive: ${(cause as Error).message}`,
          taskId: taskId ?? undefined,
          failed: true,
        });
      } finally {
        if (!signal.aborted) setBusy(false);
        if (grant && !signal.aborted) {
          await refreshSessions(grant.agent, grant.endpointName, grant.sub);
          const id = sessionRef.current;
          if (id) void name(id, firstUserText.current ?? "");
        }
      }
    },
    [
      append,
      grant,
      name,
      patch,
      refreshSessions,
      setSession,
    ],
  );

  const active = useRef<AbortController | null>(null);
  const stop = useCallback(() => {
    connection.current?.abort();
    active.current?.abort();
    active.current = null;
    client.current?.close();
    setBusy(false);
  }, []);
  useEffect(() => () => { connection.current?.abort(); active.current?.abort(); client.current?.close(); }, []);

  const runPending = useCallback(async (pending: Pending) => {
    const a2a = client.current;
    if (!a2a || active.current) return;
    const controller = new AbortController();
    active.current = controller;
    write(PENDING_KEY, JSON.stringify(pending));
    try {
      const method = pending.taskId && pending.sessionId ? "tasks/resubscribe" : "message/stream";
      const params = pending.taskId && pending.sessionId
        ? { id: pending.taskId, contextId: pending.sessionId, ...(pending.cursor !== undefined ? { offset: pending.cursor } : {}) }
        : { message: pending.message };
      await consume(a2a.stream(method, params, controller.signal), pending, controller.signal);
    } finally {
      if (active.current === controller) active.current = null;
    }
  }, [consume]);

  const send = useCallback(async (textToSend: string) => {
    if (active.current) return;
    const saved = pendingFrom(read(PENDING_KEY));
    // A lost first acknowledgment must retry the same message, including messageId.
    const message: Message = saved && !saved.taskId && text(saved.message.parts) === textToSend
      ? saved.message
      : { messageId: uuid(), role: "ROLE_USER", parts: [{ text: textToSend }], ...(sessionRef.current ? { contextId: sessionRef.current, ...(read(QUESTION_KEY + sessionRef.current) ? { taskId: read(QUESTION_KEY + sessionRef.current)! } : {}) } : {}) };
    append({ kind: "user", text: textToSend });
    await runPending({ message, sessionId: sessionRef.current ?? undefined });
  }, [append, runPending]);

  const resume = useCallback(async (pending: Pending) => {
    append({ kind: "status", text: "Resuming the reply", taskId: pending.taskId });
    await runPending(pending);
  }, [append, runPending]);

  /** Talks to one example from here on: its grant, its card, its sessions. */
  const connect = useCallback(
    async (chosen: Example, sub: string) => {
      stop();
      const controller = new AbortController();
      connection.current = controller;
      const minted = await getGrant(sub, chosen.agent, chosen.endpoint, controller.signal);
      controller.signal.throwIfAborted();
      const cardResponse = await fetch(
        `${minted.endpoint}/.well-known/agent-card.json`,
        {
          headers: { Authorization: `Bearer ${minted.token}`, "A2A-Version": "1.0" },
          signal: controller.signal,
        },
      );
      if (!cardResponse.ok) throw new Error(`Agent card failed (HTTP ${cardResponse.status})`);
      const agentCard = decodeCard(await cardResponse.json());
      controller.signal.throwIfAborted();
      grantRef.current = minted;
      setGrant(minted);
      setExample(chosen);
      setCard({
        name: agentCard.name,
        /* The card's one-line description, as a title: no trailing full stop. */
        description: (agentCard.description ?? agentCard.name).replace(
          /\.\s*$/,
          "",
        ),
      });
      client.current = new A2AClient(minted.endpoint, minted.token);
      await refreshSessions(minted.agent, minted.endpointName, minted.sub);
    },
    [refreshSessions, stop],
  );

  /* Boot: the examples, the first one running, then the saved session and any
     reply that was streaming when the page went away. */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cfg = await getConfig();
        const found = (await listExamples()).examples;
        if (cancelled) return;
        setConfig(cfg);
        setExamples(found);
        const live = found.find((candidate) => candidate.live);
        if (!live) {
          setExample(found[0] ?? null);
          return;
        }
        await connect(live, cfg.grant.subs[0]);
        if (cancelled) return;
        const saved = read(SESSION_KEY);
        if (saved) {
          sessionRef.current = saved;
          setSessionId(saved);
          await restore(saved);
          /* restore() dropped a session that no longer exists. */
          if (sessionRef.current !== saved) return;
          const pending = pendingFrom(read(PENDING_KEY));
          if (pending && pending.sessionId === saved) void resume(pending);
        } else {
          const pending = pendingFrom(read(PENDING_KEY));
          if (pending && !pending.sessionId) void resume(pending);
        }
      } catch {
        if (!cancelled) setUnreachable(true);
      }
    })();
    return () => {
      cancelled = true;
      connection.current?.abort();
      active.current?.abort();
      client.current?.close();
    };
    /* Boot runs once; the callbacks it uses are stable for the page's life. */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Another example: a fresh conversation on it. One that is not running is shown as such,
      with what to start, and connected once it answers. */
  const selectExample = useCallback(
    async (chosen: Example) => {
      if (!config) return;
      stop();
      write(PENDING_KEY, null);
      setSession(null);
      setEntries([]);
      setUnreachable(false);
      if (!chosen.live) {
        client.current = null;
        grantRef.current = null;
        setGrant(null);
        setCard(null);
        setSessions([]);
        setExample(chosen);
        return;
      }
      try {
        await connect(chosen, config.grant.subs[0]);
      } catch {
        setUnreachable(true);
      }
    },
    [config, connect, setSession, stop],
  );

  /* Check again after each response; only one discovery request is in flight. */
  useEffect(() => {
    if (!config || !example || grant) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const found = (await listExamples(controller.signal)).examples;
        if (controller.signal.aborted) return;
        setExamples(found);
        const now = found.find((candidate) => sameExample(candidate, example));
        if (now?.live) await connect(now, config.grant.subs[0]);
      } catch { /* Discovery retries on its next scheduled refresh. */ }
      finally { if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 3000); }
    };
    timer = setTimeout(() => void poll(), 3000);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [config, example, grant, connect]);

  const select = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      stop();
      write(PENDING_KEY, null);
      setSession(id);
      void restore(id);
    },
    [restore, setSession, stop],
  );

  const fresh = useCallback(() => { stop(); write(PENDING_KEY, null); setSession(null); }, [setSession, stop]);

  /** A session made elsewhere (a call with no conversation yet) becomes this one. */
  const adopt = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      setSession(id, { fresh: !sessionRef.current });
      if (grant)
        void refreshSessions(grant.agent, grant.endpointName, grant.sub);
    },
    [grant, refreshSessions, setSession],
  );

  return {
    config,
    grant,
    examples,
    example,
    selectExample,
    card,
    sessions,
    sessionId,
    entries,
    restoring,
    busy,
    unreachable,
    registryDown,
    send,
    select,
    fresh,
    adopt,
    name,
  };
}

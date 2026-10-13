"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { A2AClient, interpret, uuid } from "./a2a";
import { HttpError, createSession, getConfig, getGrant, getHistory, listSessions, nameSession } from "./api";
import { listExamples, sameExample, type Example } from "./examples";
import { text } from "./format";
import type { Delegation, Entry, Grant, SessionRow, ConsoleConfig } from "./types";

export const SESSION_KEY = "console.session";
const PENDING_KEY = "console.pending";

type Pending = { taskId: string; sessionId: string; text?: string };
/** An entry before it is given its id: a distributive Omit, so each kind keeps its fields. */
type Draft = Entry extends infer E ? (E extends Entry ? Omit<E, "id" | "at"> : never) : never;

const read = (name: string) => (typeof window === "undefined" ? null : window.localStorage.getItem(name));
const write = (name: string, value: string | null) => {
  if (value === null) window.localStorage.removeItem(name);
  else window.localStorage.setItem(name, value);
};

let counter = 0;
const nextId = () => `e${++counter}`;

/** `FunctionCall.extra` keys the delegating call carries (delegation/tool.py). */
export const DELEGATION_KEYS = {
  taskId: "lk.delegation.task_id",
  endpoint: "lk.delegation.endpoint",
  transport: "lk.delegation.transport",
  profile: "lk.delegation.profile",
} as const;
const DELEGATE_TOOL = "delegate";

type HistoryItem = Record<string, unknown> & { extra?: Record<string, unknown> };

/** Whether a history item is a call that delegated: the delegate tool, or any tool that recorded the expert's task. */
const delegated = (item: HistoryItem) => item.name === DELEGATE_TOOL || typeof item.extra?.[DELEGATION_KEYS.taskId] === "string";

/** The call a frame belongs to: a non-blocking tool's later frames are `<call>_update_<n>` and `<call>_final` (voice/events.py). */
export const baseCall = (callId: string) => callId.replace(/_(update_\d+|final)$/, "");

/** The `lk.delegation.*` a frame recorded, without the ones it did not. */
const recorded = (item: HistoryItem) => Object.fromEntries(Object.entries(delegationExtra(item)).filter(([, value]) => value));

/** The request as the call sent it: the delegate tool's `task`, else the arguments as `key value`. */
function requestOf(args: unknown): string {
  try {
    const parsed = typeof args === "string" ? (JSON.parse(args) as unknown) : args;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const record = parsed as Record<string, unknown>;
      if (typeof record.request === "string") return record.request;
      if (typeof record.task === "string") return record.task;
      return Object.entries(record)
        .map(([key, value]) => `${key} ${typeof value === "string" ? value : JSON.stringify(value)}`)
        .join(", ");
    }
  } catch {
    /* not JSON */
  }
  return typeof args === "string" ? args : "";
}

/** What a history item says about where its delegation went (`lk.delegation.*`). */
function delegationExtra(item: HistoryItem): Pick<Delegation, "taskId" | "endpoint" | "transport" | "profile"> {
  const extra = item.extra ?? {};
  const read = (key: string) => (typeof extra[key] === "string" ? (extra[key] as string) : undefined);
  return { taskId: read(DELEGATION_KEYS.taskId), endpoint: read(DELEGATION_KEYS.endpoint), transport: read(DELEGATION_KEYS.transport), profile: read(DELEGATION_KEYS.profile) };
}

/** What a tool said while it ran, as the framework words it to the model (voice/tool_executor.py UPDATE_TEMPLATE). */
const UPDATE_PREFIX = /^The tool `[^`]*` has updated, message: /;
const UPDATE_TAIL = /\nThe task is still running.*$/s;
/** The dispatch note of the delegate tool (delegation/tool.py): an instruction to the model, never shown. */
const DISPATCH_NOTE = "Acknowledge in a few natural words";

/**
 * A function_call_output of the history, sorted: `undefined` for a real
 * output, the message for a progress update, `null` for the delegate's
 * dispatch note, which is the model's business and nobody else's.
 */
export function toolUpdate(output: string): string | null | undefined {
  if (!UPDATE_PREFIX.test(output)) return undefined;
  const message = output.replace(UPDATE_PREFIX, "").replace(UPDATE_TAIL, "").trim();
  if (message.includes(DISPATCH_NOTE)) return null;
  return message;
}

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
export function useConversation({ onSessionLost }: { onSessionLost?: () => void } = {}) {
  const [config, setConfig] = useState<ConsoleConfig | null>(null);
  const [grant, setGrant] = useState<Grant | null>(null);
  /** Every example the API knows, and the one this conversation talks to. */
  const [examples, setExamples] = useState<Example[]>([]);
  const [example, setExample] = useState<Example | null>(null);
  const grantRef = useRef<Grant | null>(null);
  const [card, setCard] = useState<{ name: string; description: string } | null>(null);
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
    if (entry.kind === "user" && firstUserText.current === null) firstUserText.current = entry.text;
    setEntries((list) => [...list, { ...entry, id, at: Date.now() } as Entry]);
    return id;
  }, []);
  const patch = useCallback((id: string, update: (entry: Entry) => Entry) => {
    setEntries((list) => list.map((entry) => (entry.id === id ? update(entry) : entry)));
  }, []);

  const refreshSessions = useCallback(async (agent: string, endpoint: string, sub: string) => {
    try {
      const rows = await listSessions(agent, endpoint, sub);
      for (const row of rows) if (row.attributes?.title) titled.current.add(row.sessionId);
      setSessions(rows);
      setRegistryDown(false);
    } catch {
      setRegistryDown(true);
    }
  }, []);

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
        /* One card per delegating call: its frames (progress, then the answer) join it by call id. */
        const byCall = new Map<string, Extract<Entry, { kind: "delegation" }>>();
        for (const item of items as HistoryItem[]) {
          const kind = item.type as string;
          const at = typeof item.created_at === "number" ? item.created_at * 1000 : Date.now();
          const callId = typeof item.call_id === "string" ? baseCall(item.call_id) : undefined;
          if (kind === "agent_handoff") {
            /* Which agent answers from here: the voice door hands the session to its
               voice agent for a call. */
            const agent = typeof item.new_agent_id === "string" ? item.new_agent_id : "";
            if (agent) restored.push({ kind: "status", id: nextId(), at, text: `${agent} answers from here` });
          } else if (kind === "message") {
            const role = item.role === "user" ? "user" : "agent";
            restored.push(
              role === "user"
                ? { kind: "user", id: nextId(), at, text: text(item.content) }
                : { kind: "agent", id: nextId(), at, text: text(item.content), interrupted: item.interrupted === true || undefined },
            );
          } else if (kind === "function_call") {
            const args = typeof item.arguments === "string" ? item.arguments : JSON.stringify(item.arguments ?? {});
            const known = callId ? byCall.get(callId) : undefined;
            /* A later frame: the final one carries the expert's task id. */
            if (known) Object.assign(known, recorded(item));
            else if (delegated(item)) {
              const entry: Extract<Entry, { kind: "delegation" }> = {
                kind: "delegation",
                id: nextId(),
                at,
                name: String(item.name ?? ""),
                task: requestOf(args),
                progress: [],
                callId,
                ...delegationExtra(item),
              };
              if (callId) byCall.set(callId, entry);
              restored.push(entry);
            } else restored.push({ kind: "tool", id: nextId(), at, name: String(item.name ?? ""), args });
          } else if (kind === "function_call_output") {
            const output = text(item.output);
            const update = toolUpdate(output);
            /* The dispatch note is the model's, not the person's. */
            if (update === null) continue;
            const owner = callId ? byCall.get(callId) : undefined;
            if (owner) {
              if (update !== undefined) owner.progress.push({ at, text: update });
              else {
                owner.answer = output;
                owner.failed = Boolean(item.is_error) || undefined;
                owner.endedAt = at;
              }
            } else if (update !== undefined) restored.push({ kind: "progress", id: nextId(), at, text: update });
            else restored.push({ kind: "tool", id: nextId(), at, name: String(item.name ?? ""), output, isError: Boolean(item.is_error) });
          }
        }
        firstUserText.current = restored.find((entry) => entry.kind === "user")?.text ?? null;
        setEntries(restored);
      } catch (cause) {
        if (cause instanceof HttpError && cause.status === 404) {
          /* The session is gone (expired, or another stack): not a fault.
             Forget it and start a new conversation, quietly. */
          if (sessionRef.current === id) {
            setSession(null, { fresh: true });
            write(PENDING_KEY, null);
            lost.current?.();
          }
          return;
        }
        setEntries([{ kind: "status", id: nextId(), at: Date.now(), text: "The conversation's history could not be loaded." }]);
      } finally {
        if (sessionRef.current === id) setRestoring(false);
      }
    },
    [setSession],
  );

  /**
   * Once a turn is over the history knows what the wire did not say about its
   * delegations: the call, the expert's task id, where it was reached and how,
   * the request as sent, the answer as returned. Each card without a call
   * takes the next unclaimed delegating call's, in order.
   */
  const settleDelegations = useCallback(async () => {
    const id = sessionRef.current;
    if (!id) return;
    let items: HistoryItem[];
    try {
      items = (await getHistory(id, grantRef.current?.agent ?? "")).items as HistoryItem[];
    } catch {
      return;
    }
    const calls = new Map<string, { call: HistoryItem; extra: ReturnType<typeof recorded>; output?: HistoryItem }>();
    for (const item of items) {
      if (typeof item.call_id !== "string") continue;
      const callId = baseCall(item.call_id);
      const known = calls.get(callId);
      if (item.type === "function_call") {
        if (known) Object.assign(known.extra, recorded(item));
        else if (delegated(item)) calls.set(callId, { call: item, extra: recorded(item) });
      } else if (item.type === "function_call_output" && known && toolUpdate(text(item.output)) === undefined) known.output = item;
    }
    if (!calls.size) return;
    setEntries((list) => {
      const claimed = new Set(list.flatMap((entry) => (entry.kind === "delegation" && entry.callId ? [entry.callId] : [])));
      const unclaimed = [...calls].filter(([callId]) => !claimed.has(callId));
      let changed = false;
      const next = list.map((entry) => {
        if (entry.kind !== "delegation" || entry.callId) return entry;
        const claim = unclaimed.shift();
        if (!claim) return entry;
        const [callId, { call, extra, output }] = claim;
        changed = true;
        return {
          ...entry,
          ...extra,
          callId,
          name: String(call.name ?? entry.name),
          task: requestOf(call.arguments) || entry.task,
          answer: output ? text(output.output) : entry.answer,
          failed: output ? Boolean(output.is_error) || undefined : entry.failed,
          endedAt: output && typeof output.created_at === "number" ? output.created_at * 1000 : (entry.endedAt ?? Date.now()),
        };
      });
      return changed ? next : list;
    });
  }, []);

  /** One turn's frames, from message/stream or tasks/resubscribe. */
  const consume = useCallback(
    async (frames: AsyncGenerator<Parameters<typeof interpret>[0]>, userText?: string) => {
      let reply: string | null = null;
      let artifactText = "";
      let taskId: string | null = null;
      setBusy(true);
      try {
        for await (const frame of frames) {
          for (const event of interpret(frame)) {
            switch (event.type) {
              case "error":
                append({ kind: "status", text: `This turn failed: ${event.message}`, taskId: taskId ?? undefined, failed: true });
                return;
              case "task":
                taskId = event.taskId;
                /* The first message made the session. */
                if (!sessionRef.current) setSession(event.contextId, { fresh: true });
                write(PENDING_KEY, JSON.stringify({ taskId, sessionId: event.contextId, text: userText } satisfies Pending));
                if (event.artifactText) artifactText = event.artifactText;
                break;
              case "progress":
                /* The expert's progress belongs to the delegation of this turn: the card opens on its first line. */
                setEntries((list) => {
                  const now = Date.now();
                  for (let index = list.length - 1; index >= 0; index--) {
                    const entry = list[index];
                    if (entry.kind === "user") break;
                    if (entry.kind === "delegation" && entry.answer === undefined)
                      return list.map((candidate, at) => (at === index ? { ...entry, progress: [...entry.progress, { at: now, text: event.text }] } : candidate));
                  }
                  return [...list, { kind: "delegation", id: nextId(), at: now, name: "", task: userText ?? "", progress: [{ at: now, text: event.text }] }];
                });
                break;
              case "artifact":
                if (!reply) reply = append({ kind: "agent", text: "", taskId: taskId ?? undefined, streaming: true });
                patch(reply, (entry) =>
                  entry.kind === "agent"
                    ? { ...entry, text: event.append ? entry.text + event.text : event.text || entry.text }
                    : entry,
                );
                break;
              case "question":
                append({ kind: "question", text: event.text });
                break;
              case "ended": {
                write(PENDING_KEY, null);
                /* The turn's delegation, if any, is over with it. */
                setEntries((list) => list.map((entry) => (entry.kind === "delegation" && entry.endedAt === undefined ? { ...entry, endedAt: Date.now() } : entry)));
                if (artifactText && !reply) reply = append({ kind: "agent", text: artifactText, taskId: event.taskId });
                if (reply) patch(reply, (entry) => ({ ...entry, streaming: false }) as Entry);
                /* The plan's spelling of the A2A states: input-required, not INPUT_REQUIRED. */
                const state = event.state.replace("TASK_STATE_", "").toLowerCase().replace(/_/g, "-");
                const meta = Object.entries(event.meta)
                  .map(([name, value]) => `${name}=${JSON.stringify(value)}`)
                  .join(" ");
                append({
                  kind: "status",
                  text: state,
                  taskId: event.taskId,
                  state,
                  meta: meta || undefined,
                  failed: state === "failed" || state === "canceled" || state === "rejected",
                });
                break;
              }
            }
          }
        }
      } catch (cause) {
        append({
          kind: "status",
          text: `The reply did not arrive: ${(cause as Error).message}`,
          taskId: taskId ?? undefined,
          failed: true,
        });
      } finally {
        setBusy(false);
        if (grant) {
          await refreshSessions(grant.agent, grant.endpointName, grant.sub);
          const id = sessionRef.current;
          if (id && userText) void name(id, firstUserText.current ?? userText);
        }
        void settleDelegations();
      }
    },
    [append, grant, name, patch, refreshSessions, setSession, settleDelegations],
  );

  const send = useCallback(
    async (textToSend: string) => {
      const a2a = client.current;
      if (!a2a) return;
      const message: Record<string, unknown> = {
        messageId: uuid(),
        role: "user",
        parts: [{ kind: "text", text: textToSend }],
      };
      if (!sessionRef.current && grant) {
        /* The thread is the subject's: made here so the list shows it. */
        const created = await createSession(grant.sub, grant.agent, grant.endpointName);
        setSession(created.sessionId, { fresh: true });
      }
      if (sessionRef.current) message.contextId = sessionRef.current;
      append({ kind: "user", text: textToSend });
      await consume(a2a.stream("message/stream", { message }, sessionRef.current), textToSend);
    },
    [append, consume, grant, setSession],
  );

  const resume = useCallback(
    async (pending: Pending) => {
      const a2a = client.current;
      if (!a2a) return;
      if (pending.text) append({ kind: "user", text: pending.text });
      append({ kind: "status", text: "resuming the reply", taskId: pending.taskId });
      await consume(
        a2a.stream("tasks/resubscribe", { id: pending.taskId, contextId: pending.sessionId }, pending.sessionId),
        pending.text,
      );
    },
    [append, consume],
  );

  /** Talks to one example from here on: its grant, its card, its sessions. */
  const connect = useCallback(
    async (chosen: Example, sub: string) => {
      const minted = await getGrant(sub, chosen.agent, chosen.endpoint);
      const cardResponse = await fetch(`${minted.endpoint}/.well-known/agent-card.json`);
      const agentCard = (await cardResponse.json()) as { name: string; description?: string };
      grantRef.current = minted;
      setGrant(minted);
      setExample(chosen);
      setCard({
        name: agentCard.name,
        /* The card's one-line description, as a title: no trailing full stop. */
        description: (agentCard.description ?? agentCard.name).replace(/\.\s*$/, ""),
      });
      client.current = new A2AClient(minted.endpoint, minted.token);
      await refreshSessions(minted.agent, minted.endpointName, minted.sub);
    },
    [refreshSessions],
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
          const pendingRaw = read(PENDING_KEY);
          const pending = pendingRaw ? (JSON.parse(pendingRaw) as Pending) : null;
          if (pending && pending.sessionId === saved) void resume(pending);
        }
      } catch {
        if (!cancelled) setUnreachable(true);
      }
    })();
    return () => {
      cancelled = true;
    };
    /* Boot runs once; the callbacks it uses are stable for the page's life. */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Another example: a fresh conversation on it. One that is not running is shown as such,
      with what to start, and connected once it answers. */
  const selectExample = useCallback(
    async (chosen: Example) => {
      if (!config) return;
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
    [config, connect, setSession],
  );

  /* An example that is not running is checked for every few seconds, and connected when it is. */
  useEffect(() => {
    if (!config || !example || grant) return;
    const chosen = example;
    const timer = window.setInterval(() => {
      void (async () => {
        const found = (await listExamples()).examples;
        setExamples(found);
        const now = found.find((candidate) => sameExample(candidate, chosen));
        if (now?.live) await connect(now, config.grant.subs[0]);
      })();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [config, example, grant, connect]);

  const select = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      setSession(id);
      void restore(id);
    },
    [restore, setSession],
  );

  const fresh = useCallback(() => setSession(null), [setSession]);

  /** A session made elsewhere (a call with no conversation yet) becomes this one. */
  const adopt = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      setSession(id, { fresh: !sessionRef.current });
      if (grant) void refreshSessions(grant.agent, grant.endpointName, grant.sub);
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

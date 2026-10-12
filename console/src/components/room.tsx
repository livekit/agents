"use client";

import { ExamplePicker } from "./switcher";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SessionProvider } from "@livekit/components-react";
import { getSession } from "@/lib/api";
import { useCall } from "@/lib/use-call";
import { useConversation } from "@/lib/use-conversation";
import { useNow, useSession } from "@/lib/use-session";
import { text } from "@/lib/format";
import type { DatabaseDump, SessionRow } from "@/lib/types";
import { Drawer, useDrawer } from "./drawer";
import { ConversationPanel } from "./panels/conversation";
import { InfoPanel } from "./panels/info";
import { InboxPanel } from "./panels/inbox";
import { JournalPanel } from "./panels/journal";
import { LanesPanel } from "./panels/lanes";
import { sessionTitle } from "./panels/sessions";
import { Chips } from "./chips";
import { Swatches } from "./chrome/swatches";
import { StreamPanel } from "./panels/stream";
import { TasksPanel } from "./panels/tasks";

type Detail = { lanes: number; firstMessage: string | null };

/** What a session's database says about it for the list: its lanes, and the first thing the person said. */
function detailOf(database: DatabaseDump | undefined, agent: string): Detail {
  if (!database) return { lanes: 0, firstMessage: null };
  const first = database.items.find((item) => {
    const payload = item.payload as { role?: string } | null;
    return item.type === "message" && item.lane === agent && payload?.role === "user";
  });
  return {
    lanes: database.lanes.length,
    firstMessage: first ? text((first.payload as { content?: unknown }).content) : null,
  };
}

/**
 * What the list shows beside the registry row: the lane count and, for a
 * session the console has not named yet, its first message. The row does
 * not say, so each session is read once through the debug API, a few at a
 * time; the selected one follows its live dump.
 */
function useSessionDetails(sessions: SessionRow[], agent: string, selected: string | null, selectedDump: DatabaseDump | null) {
  const [details, setDetails] = useState<Map<string, Detail>>(() => new Map());
  const inFlight = useRef(new Set<string>());
  useEffect(() => {
    const missing = sessions.map((session) => session.sessionId).filter((id) => !details.has(id) && !inFlight.current.has(id));
    if (!missing.length) return;
    let cancelled = false;
    (async () => {
      for (let start = 0; start < missing.length && !cancelled; start += 3) {
        const batch = missing.slice(start, start + 3);
        for (const id of batch) inFlight.current.add(id);
        const dumps = await Promise.allSettled(batch.map((id) => getSession(id)));
        if (cancelled) return;
        setDetails((current) => {
          const next = new Map(current);
          dumps.forEach((result, index) => {
            next.set(batch[index], detailOf(result.status === "fulfilled" ? result.value.database : undefined, agent));
          });
          return next;
        });
        for (const id of batch) inFlight.current.delete(id);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [agent, details, sessions]);
  return useMemo(
    () => (selected && selectedDump ? new Map(details).set(selected, detailOf(selectedDump, agent)) : details),
    [agent, details, selected, selectedDump],
  );
}

/**
 * The room: the conversation, by text or by voice, the accent swatches in
 * its corner; under it the conversations as a strip of chips, then the
 * debug drawer (the lanes, tasks, journal, inbox and stream of the
 * session, and what it talks to).
 */
export function Room() {
  const panel = useDrawer();
  const chat = useConversation();
  const { dump, skew, watchError, taskLog } = useSession(chat.sessionId);
  const now = useNow(skew);
  const database = dump?.database ?? null;
  const agent = chat.grant?.agent ?? "";
  const sub = chat.grant?.sub ?? "demo-user";
  const details = useSessionDetails(chat.sessions, agent, chat.sessionId, database);
  const [splitting, setSplitting] = useState(false);

  /* A call with no conversation yet makes one; the page adopts it. */
  const adopt = chat.adopt;
  const call = useCall({ sessionId: chat.sessionId, sub, agent, onSession: adopt });

  /* A selection belongs to its session: another session has its own tasks. */
  const [selection, setSelection] = useState<{ sessionId: string | null; taskId: string } | null>(null);
  const selectedTask = selection && selection.sessionId === chat.sessionId ? selection.taskId : null;
  /* A row selected in Tasks; the Stream view follows the selection. */
  const selectTask = useCallback(
    (taskId: string) => {
      setSelection({ sessionId: chat.sessionId, taskId });
    },
    [chat.sessionId],
  );
  /* The task id under a turn opens Tasks on that row. */
  const openTask = useCallback(
    (taskId: string) => {
      setSelection({ sessionId: chat.sessionId, taskId });
      panel.setView("tasks");
    },
    [chat.sessionId, panel],
  );
  const openStream = useCallback(() => panel.setView("stream"), [panel]);
  /* The experts' tasks: every task on a lane that is not the agent's own. */
  const expertTasks = useMemo(() => (database?.tasks ?? []).filter((task) => task.lane !== agent), [agent, database?.tasks]);

  /* A session with a first message and no title gets one: the one a voice
     call just made, and any older one the list still shows by its message. */
  const nameSession = chat.name;
  useEffect(() => {
    for (const session of chat.sessions) {
      if (session.attributes?.title) continue;
      const first = details.get(session.sessionId)?.firstMessage;
      if (first) void nameSession(session.sessionId, first);
    }
  }, [chat.sessions, details, nameSession]);
  /* A session nothing was said in is not worth a row; the open one, if so, is the strip's "New conversation". */
  const listed = useMemo(
    () => chat.sessions.filter((session) => session.attributes?.title || details.get(session.sessionId)?.firstMessage !== null),
    [chat.sessions, details],
  );

  /* The conversation's title: the registry's, else the first message, else nothing yet. */
  const firstMessage = chat.entries.find((entry) => entry.kind === "user")?.text ?? details.get(chat.sessionId ?? "")?.firstMessage;
  const title = sessionTitle(
    chat.sessions.find((session) => session.sessionId === chat.sessionId),
    firstMessage,
  );

  /* The tab says what the conversation is called, else which agent this is. */
  const tabTitle = chat.sessionId && title !== "New conversation" ? title : agent;
  useEffect(() => {
    document.title = tabTitle;
  }, [tabTitle]);

  /* Cmd+K (Ctrl+K): a new conversation; a call in progress ends with it. */
  const fresh = chat.fresh;
  const endCall = call.end;
  const resetVoice = call.reset;
  const startNew = useCallback(() => {
    if (call.connected) void endCall();
    resetVoice();
    fresh();
  }, [call.connected, endCall, fresh, resetVoice]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        startNew();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [startNew]);

  /* Switching sessions: the call, if any, belongs to the old one. */
  const select = useCallback(
    (id: string) => {
      if (id === chat.sessionId) return;
      if (call.connected) void endCall();
      resetVoice();
      chat.select(id);
    },
    [call.connected, chat, endCall, resetVoice],
  );

  /* The speech button starts a call into this conversation; nothing connects
     before it, not on load, not on a selection, not from anything saved. */
  const startCall = useCallback(() => {
    void call.start().catch(() => undefined);
  }, [call]);
  const hangUp = useCallback(() => {
    void endCall();
  }, [endCall]);

  const ready = Boolean(chat.card) && !chat.unreachable;
  /* Which agent this is, the same for every demo: the agent and the endpoint the grant names. */
  const agentId = chat.grant ? `${chat.grant.agent} / ${chat.grant.endpoint.split("/").pop()}` : "";
  const streamsOk = dump?.streams.available ?? chat.config?.streams.available ?? false;
  /* What a view says when there is no database to read: one line, never a raw error. */
  const empty = !chat.sessionId
    ? ""
    : watchError && !database
      ? "The session's database is not answering."
      : "Loading the session.";
  /* Booting says nothing (the composer waits, disabled); an agent that cannot be reached is told,
     and an example that is not running shows what to start, from its folder. */
  const stopped = chat.example && !chat.grant ? chat.example : null;
  const note = chat.unreachable
    ? "Can't reach the agent."
    : stopped
      ? "Not running. Start it from the repository root; this page connects once it is up."
      : "";

  const view = (() => {
    switch (panel.view) {
      case "lanes":
        return <LanesPanel lanes={database?.lanes ?? null} now={now} tip={database?.tip} sessionId={chat.sessionId} empty={empty} />;
      case "tasks":
        return <TasksPanel tasks={database?.tasks ?? null} selected={selectedTask} now={now} empty={empty} onSelect={selectTask} onStream={openStream} />;
      case "journal":
        return <JournalPanel events={database?.events ?? null} lanes={database?.lanes ?? []} now={now} empty={empty} />;
      case "inbox":
        return <InboxPanel inbox={database?.inbox ?? null} lanes={database?.lanes ?? []} now={now} empty={empty} />;
      case "stream":
        return <StreamPanel taskId={selectedTask} available={streamsOk} />;
      case "info":
        return <InfoPanel grant={chat.grant} streamsOk={streamsOk} />;
      default:
        return null;
    }
  })();

  return (
    <SessionProvider session={call.session}>
      <div className="flex h-dvh overflow-hidden bg-room" data-splitting={splitting || undefined}>
        <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="absolute top-4 left-4 z-10">
            <Swatches />
          </div>
          <div className="absolute top-4 right-4 z-10">
            <ExamplePicker examples={chat.examples} selected={chat.example} onSelect={(chosen) => void chat.selectExample(chosen)} />
          </div>
          <main className="min-h-0 min-w-0 flex-1" data-pane="conversation">
            <ConversationPanel
              entries={chat.entries}
              restoring={chat.restoring}
              voice={call.turns}
              busy={chat.busy}
              ready={ready}
              agentName={stopped ? stopped.name : agentId}
              note={note}
              command={stopped ? `cd ${stopped.dir} && ${stopped.start}` : undefined}
              call={call}
              expertTasks={expertTasks}
              taskLog={taskLog}
              now={now}
              onCall={startCall}
              onHangUp={hangUp}
              onSend={(message) => void chat.send(message)}
              onOpenTask={openTask}
            />
          </main>
          <Chips sessions={listed} selected={chat.sessionId} details={details} onSelect={select} onNew={startNew} />
          <Drawer
            view={panel.view}
            height={panel.height}
            disabled={selectedTask ? {} : { stream: "Select a task in Tasks to open its stream" }}
            onToggle={panel.toggle}
            onHeight={panel.setHeight}
            onDragging={setSplitting}
          >
            {view}
          </Drawer>
        </div>
      </div>
    </SessionProvider>
  );
}

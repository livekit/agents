"use client";

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { RoomAudioRenderer, useLocalParticipant, useMediaDeviceSelect } from "@livekit/components-react";
import { CopyId } from "../chrome/copy";
import { Icon } from "../chrome/icon";
import { Orb, type OrbState } from "../chrome/orb";
import { Disclosure, Document } from "../chrome/panel";
import { DelegationCard } from "./delegation";
import { tail } from "@/lib/format";
import { elapsed, type VoiceTurn, type useCall } from "@/lib/use-call";
import { useStick } from "@/lib/use-stick";
import type { TaskLog } from "@/lib/use-session";
import type { Entry, TaskRow } from "@/lib/types";

export type Call = ReturnType<typeof useCall>;

/** How long before a card its expert's task may have been first seen: the dump lags the wire. */
const SETTLE_SLACK_MS = 10_000;

/** The one tool a delegate is reached through (delegation/tool.py). */
const DELEGATE = "delegate";

/** What the mark and its line say, from the thread, the turn in flight and the call. */
export function stateLine(entries: Entry[], busy: boolean, call: Pick<Call, "connected" | "connecting" | "agent">): { state: OrbState; line: string } {
  if (call.connecting) return { state: "connecting", line: "Connecting the call" };
  if (call.connected) {
    const agent = call.agent.state;
    if (agent === "speaking") return { state: "speaking", line: "On a call, speaking" };
    if (agent === "thinking") return { state: "thinking", line: "On a call, thinking" };
    if (agent === "listening") return { state: "listening", line: "On a call, listening" };
    return { state: "connecting", line: "On a call, waiting for the agent" };
  }
  if (busy) {
    /* The latest progress of the turn in flight takes the line's place. */
    for (let index = entries.length - 1; index >= 0; index--) {
      const entry = entries[index];
      if (entry.kind === "user") break;
      if (entry.kind === "progress") return { state: "thinking", line: entry.text };
      if (entry.kind === "delegation" && entry.progress.length) return { state: "thinking", line: entry.progress[entry.progress.length - 1].text };
    }
    return { state: "thinking", line: "Thinking" };
  }
  for (let index = entries.length - 1; index >= 0; index--) {
    const entry = entries[index];
    if (entry.kind === "user") break;
    if (entry.kind === "status" && entry.state) return entry.state === "input-required" ? { state: "asked", line: "Asked you a question" } : { state: "listening", line: "Listening" };
  }
  return { state: "listening", line: "Listening" };
}

/** The thread: the text entries and the call's turns, in the order they happened. */
function merge(entries: Entry[], voice: VoiceTurn[]): Entry[] {
  const spoken: Entry[] = voice.map((turn) => ({ kind: turn.role, id: `v-${turn.id}`, at: turn.at, text: turn.text, voice: true }));
  const sorted = [...entries, ...spoken].sort((a, b) => a.at - b.at);
  /* Nothing said is nothing shown; a handoff before anything was said divides nothing. */
  const said = (entry: Entry) => (entry.kind === "user" && entry.text.trim() !== "") || (entry.kind === "agent" && (entry.text.trim() !== "" || entry.streaming === true));
  return sorted.filter((entry, index) => {
    if (entry.kind === "user" || entry.kind === "agent") return said(entry);
    if (entry.kind === "status" && !entry.taskId && !entry.failed) return sorted.slice(0, index).some(said);
    return true;
  });
}

/**
 * The stage: the chat with the agent, on the room grey. Over it the mark
 * (the orb, which hears the agent's voice during a call), the agent's name
 * and one line saying what it is doing; the thread as a measure, centred,
 * that sticks to its foot; a hairline, then the composer at the foot, the
 * call bar over it while a call is up. Before the first message the mark
 * is large and the composer sits with it in the middle of the stage; it is
 * the same element in both places, so it keeps its focus.
 */
export function ConversationPanel({
  entries,
  restoring,
  voice,
  busy,
  ready,
  agentName,
  note,
  command,
  call,
  expertTasks,
  taskLog,
  now,
  onCall,
  onHangUp,
  onSend,
  onOpenTask,
}: {
  entries: Entry[];
  /** The history is loading: keep the conversation's layout, not the empty stage. */
  restoring: boolean;
  voice: VoiceTurn[];
  busy: boolean;
  ready: boolean;
  agentName: string;
  note: string;
  /** What to run, under the note, one click to copy: an example that is not running. */
  command?: string;
  call: Call;
  /** The experts' tasks in the session's database: every task on a lane that is not the agent's. */
  expertTasks: TaskRow[];
  /** What the dumps said about each task over time. */
  taskLog: Map<string, TaskLog>;
  now: number;
  /** The speech button: start a call into this conversation. */
  onCall: () => void;
  onHangUp: () => void;
  onSend: (text: string) => void;
  onOpenTask: (taskId: string) => void;
}) {
  const { thread, taskOf } = useMemo(() => {
    const cards = entries.filter((entry): entry is Extract<Entry, { kind: "delegation" }> => entry.kind === "delegation");
    const byId = new Map(expertTasks.map((task) => [task.task_id, task]));
    const taskOf = new Map<string, TaskRow>();
    const claimed = new Set<string>();
    /* A card takes the task its call recorded (`lk.delegation.task_id`). */
    for (const card of cards)
      if (card.taskId && byId.has(card.taskId)) {
        taskOf.set(card.id, byId.get(card.taskId)!);
        claimed.add(card.taskId);
      }
    const seen = (task: TaskRow) => taskLog.get(task.task_id)?.firstSeen ?? task.updated_at * 1000;
    /* A card the history has not settled yet (the turn in flight) takes the newest unclaimed task seen since it started. */
    for (const card of cards) {
      if (card.callId || card.taskId) continue;
      const task = expertTasks.filter((candidate) => !claimed.has(candidate.task_id) && seen(candidate) >= card.at - SETTLE_SLACK_MS).sort((a, b) => b.seq - a.seq)[0];
      if (!task) continue;
      taskOf.set(card.id, task);
      claimed.add(task.task_id);
    }
    const spare = expertTasks.filter((task) => !claimed.has(task.task_id));
    /* An expert's task no call accounts for (a call delegating right now): a card of its own, where it was first seen. */
    const orphans: Entry[] = spare.map((task) => ({
      kind: "delegation",
      id: `t-${task.task_id}`,
      at: seen(task),
      name: "",
      task: "",
      taskId: task.task_id,
      progress: [],
    }));
    for (const orphan of orphans) taskOf.set(orphan.id, byId.get((orphan as Extract<Entry, { kind: "delegation" }>).taskId!)!);
    return { thread: merge([...entries, ...orphans].sort((a, b) => a.at - b.at), voice), taskOf };
  }, [entries, expertTasks, taskLog, voice]);
  const empty = thread.length === 0 && !restoring;
  const { state, line } = stateLine(entries, busy, call);
  /* The dots at the foot of the thread: while a turn runs with nothing streaming and no card already breathing. */
  const lastEntry = entries[entries.length - 1];
  const thinking = busy && !entries.some((entry) => entry.kind === "agent" && entry.streaming) && !(lastEntry?.kind === "delegation" && lastEntry.endedAt === undefined);
  const inCall = call.connected || call.connecting;
  const agentTrack = call.connected && call.agent.isConnected ? call.agent.microphoneTrack : undefined;
  const last = thread[thread.length - 1];
  const { attach, following, follow, onScroll } = useStick([thread.length, last?.kind === "agent" ? last.text : last?.kind === "delegation" ? last.progress.length : null, thinking]);
  /* The mark: large on the empty stage, beside the name once there is a thread. */
  const size: "large" | "medium" = empty ? "large" : "medium";
  /* The line under the name: the note on the empty stage, else only while something happens. */
  const stated = empty ? Boolean(note) : inCall || state !== "listening";

  const head = (
    <div key="head" className="mx-auto w-full max-w-measure px-4 pt-3 pb-2 data-[size=large]:pb-6" data-head="true" data-size={size}>
      <div
        className="grid items-center gap-x-3 gap-y-1 data-[size=large]:justify-items-center data-[size=large]:text-center data-[size=medium]:grid-cols-[auto_minmax(0,1fr)] data-[size=small]:grid-cols-[auto_minmax(0,1fr)]"
        data-size={size}
      >
        <span className="flex items-center justify-center data-[stated=true]:row-span-2 data-[size=large]:h-mark-lg data-[size=medium]:h-mark" data-size={size} data-stated={stated}>
          <Orb state={state} size={size} audioTrack={agentTrack} />
        </span>
        <h1 className="m-0 truncate font-mono text-[length:var(--fs-title-sm)] text-ink data-[size=large]:mt-3 data-[size=large]:text-[length:var(--fs-title)] data-[stated=true]:self-end" data-size={size} data-stated={stated}>
          {agentName || "Agent"}
        </h1>
        {!stated ? null : (
          <p className="m-0 flex min-h-row-sm items-center gap-2 self-start text-state text-label" role="status" data-state-line={state}>
            <span className="truncate">{empty ? note : line}</span>
            {!empty && state === "thinking" ? <Dots /> : null}
          </p>
        )}
      </div>
      {empty && command ? (
        <div className="mt-3 flex justify-center" data-start-command>
          <CopyId id={command} className="max-w-full" />
        </div>
      ) : null}
    </div>
  );

  const composer = (
    <div key="composer" className="grid" data-empty={empty} data-foot="true">
      <div className="grid bg-room">
        <div className="mx-auto grid w-full max-w-measure gap-2 px-4 py-3 data-[empty=true]:py-0" data-empty={empty}>
          {call.connected ? <CallPill call={call} onHangUp={onHangUp} /> : call.connecting ? <ConnectingPill onHangUp={onHangUp} /> : null}
          {call.error ? (
            <p className="m-0 px-3 text-failed-text" role="alert">
              The call could not start. Keep chatting in text.
            </p>
          ) : null}
          <Composer
            ready={ready}
            placeholder={call.connected ? "Type during the call" : "Send a message"}
            calling={call.connecting ? "connecting" : call.connected ? "connected" : "idle"}
            onSend={onSend}
            onCall={onCall}
          />
        </div>
      </div>
    </div>
  );

  return (
    <section
      className="grid h-full min-h-0 min-w-0 overflow-hidden bg-room data-[empty=false]:grid-rows-[auto_minmax(0,1fr)_auto] data-[empty=true]:grid-rows-[minmax(0,1fr)_auto_auto_minmax(0,1.4fr)]"
      data-panel="conversation"
      data-empty={empty}
      data-mode={inCall ? "voice" : "text"}
    >
      {empty ? (
        <>
          <span key="top" />
          <div key="invitation" data-invitation="true" className="contents">
            {head}
          </div>
          {composer}
          <span key="bottom" />
        </>
      ) : (
        <>
          {head}
          <div key="thread" className="relative min-h-0 min-w-0">
            <div ref={attach} className="h-full min-h-0 overflow-x-hidden overflow-y-auto px-4 py-3" data-log="conversation" onScroll={onScroll}>
              <div className="mx-auto grid max-w-measure grid-cols-[minmax(0,1fr)] gap-[var(--pad-6)]">
                {thread.map((entry) =>
                  entry.kind === "delegation" ? (
                    <DelegationCard key={entry.id} entry={entry} task={taskOf.get(entry.id)} log={taskLog.get(taskOf.get(entry.id)?.task_id ?? "")} now={now} onOpenTask={onOpenTask} />
                  ) : (
                    <EntryView key={entry.id} entry={entry} agentName={agentName} onOpenTask={onOpenTask} />
                  ),
                )}
                {thinking ? (
                  <div className="flex h-row items-center gap-2 px-3 text-label" role="status" aria-live="polite" data-thinking="true">
                    <Dots />
                    <span className="truncate">{line}</span>
                  </div>
                ) : null}
              </div>
            </div>
            {!following ? (
              <button type="button" className="btn absolute bottom-3 left-1/2 -translate-x-1/2 shadow-[var(--shadow)]" onClick={follow} data-action="latest">
                <Icon name="disclosureDown" size={12} />
                Latest
              </button>
            ) : null}
          </div>
          {composer}
        </>
      )}
    </section>
  );
}

/**
 * The composer: the field grown to hold more than one line, the speech
 * button and the send glyph inside the well as quiet buttons; the send
 * glyph takes the accent once there is text. Enter sends, Shift+Enter
 * breaks the line, Escape clears; Cmd+K (the room's) starts a new
 * conversation. The speech button is the one way a call starts: nothing
 * connects before it is clicked; while the call is up the same spot ends it.
 */
function Composer({
  ready,
  placeholder,
  calling,
  onSend,
  onCall,
}: {
  ready: boolean;
  placeholder: string;
  calling: "idle" | "connecting" | "connected";
  onSend: (text: string) => void;
  onCall: () => void;
}) {
  const [draft, setDraft] = useState("");
  const area = useRef<HTMLTextAreaElement | null>(null);
  const send = () => {
    const text = draft.trim();
    if (!text || !ready) return;
    setDraft("");
    onSend(text);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    } else if (event.key === "Escape" && draft) {
      event.preventDefault();
      setDraft("");
    }
  };
  /* The textarea grows with its content; WebKit has no field-sizing, so it is measured. */
  useEffect(() => {
    const element = area.current;
    if (!element) return;
    element.style.height = "0";
    element.style.height = `${element.scrollHeight}px`;
  }, [draft]);
  const hasText = Boolean(draft.trim());
  return (
    <div className="composer" data-composer="true" data-calling={calling}>
      <textarea
        ref={area}
        rows={1}
        value={draft}
        placeholder={placeholder}
        aria-label="Message"
        disabled={!ready}
        autoFocus
        onKeyDown={onKeyDown}
        onChange={(event) => setDraft(event.target.value)}
      />
      <div className="flex items-center gap-3">
        {calling === "idle" ? (
          <button type="button" className="btn btnQuiet composerBtn" disabled={!ready} title="Call the agent" aria-label="Call the agent" data-action="speech" onClick={onCall}>
            <Icon name="voice" size={16} />
          </button>
        ) : null}
        <button
          type="button"
          className={`btn composerBtn ${hasText ? "" : "btnQuiet"}`}
          data-primary={hasText ? "true" : undefined}
          disabled={!ready || !hasText}
          title="Send (Enter)"
          aria-label="Send"
          onClick={send}
        >
          <Icon name="send" size={16} />
        </button>
      </div>
    </div>
  );
}

/** The three dots breathing in turn (base.css). */
function Dots() {
  return (
    <span className="flex items-center gap-1" aria-hidden="true">
      <span className="thinkingDot" />
      <span className="thinkingDot" />
      <span className="thinkingDot" />
    </span>
  );
}

/**
 * The call, a pill of its own over the composer while it is up: a live dot,
 * what the agent is doing, the elapsed time, the mic toggle with the
 * microphone picker behind its chevron (when there is more than one), and
 * the end button. Text stays available beside it (plan 3.3.7: a text during
 * a call lands in the same session).
 */
function CallPill({ call, onHangUp }: { call: Call; onHangUp: () => void }) {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const agent = call.agent;
  const { devices, activeDeviceId } = useMediaDeviceSelect({ kind: "audioinput", room: call.session.room });
  const microphones = devices.filter((device) => device.deviceId !== "");
  return (
    <div className="callPill" data-call-bar="true" data-agent-state={agent.state} data-live={agent.isConnected}>
      <RoomAudioRenderer />
      <span className="callDot" aria-hidden="true" />
      <span className="max-w-[var(--w-chip)] truncate">
        {agent.isConnected ? (agent.state === "speaking" ? "The agent is speaking" : agent.state === "thinking" ? "Thinking" : "Listening") : "Waiting for the agent"}
      </span>
      <span className="text-label tabular-nums" data-call-elapsed="true">
        {elapsed(call.startedAt, now)}
      </span>
      <span className="relative flex items-center">
        <button
          type="button"
          className="callBtn"
          aria-pressed={isMicrophoneEnabled}
          title={isMicrophoneEnabled ? "Mute the microphone" : "Unmute the microphone"}
          aria-label={isMicrophoneEnabled ? "Mute the microphone" : "Unmute the microphone"}
          data-action="mic"
          data-muted={!isMicrophoneEnabled}
          onClick={() => void localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled)}
        >
          <Icon name={isMicrophoneEnabled ? "mic" : "micOff"} size={14} />
        </button>
        {microphones.length > 1 ? (
          <label className="callBtn callDevice" title="Microphone">
            <Icon name="disclosureDown" size={10} />
            <select
              className="absolute inset-0 cursor-pointer opacity-0"
              aria-label="Microphone"
              data-action="microphone"
              value={activeDeviceId}
              onChange={(event) => void call.setMicrophone(event.target.value).catch(() => undefined)}
            >
              {microphones.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || "Microphone"}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </span>
      <button type="button" className="callBtn callEnd" title="End the call" aria-label="End the call" data-action="hang-up" onClick={onHangUp}>
        <Icon name="phoneOff" size={14} />
      </button>
    </div>
  );
}

/** The pill while the call is still connecting: no room to read yet, only the way out. */
function ConnectingPill({ onHangUp }: { onHangUp: () => void }) {
  return (
    <div className="callPill" data-call-bar="true" data-live="false">
      <Icon name="loader" size={14} className="animate-spin text-label" />
      <span className="text-label">Connecting</span>
      <button type="button" className="callBtn callEnd" title="Cancel the call" aria-label="Cancel the call" data-action="hang-up" onClick={onHangUp}>
        <Icon name="phoneOff" size={14} />
      </button>
    </div>
  );
}

/** A tool's arguments as `key value` pairs, when they are a JSON object; the text as given otherwise. */
function argumentPairs(args: string | undefined): { pairs: [string, string][] | null; raw: unknown } {
  if (!args) return { pairs: [], raw: undefined };
  try {
    const parsed = JSON.parse(args) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
      return {
        pairs: Object.entries(parsed as Record<string, unknown>).map(([key, value]) => [key, typeof value === "string" ? value : JSON.stringify(value)]),
        raw: parsed,
      };
    return { pairs: null, raw: parsed };
  } catch {
    return { pairs: null, raw: args };
  }
}

/**
 * A tool call as one compact row: the wrench, the name and its `key value`
 * arguments, wrapping inside the measure; its output an indented quiet
 * line under it; the document behind either folds open. A `delegate` call
 * reads as what it is to the person: the expert was asked, and this is
 * what it was asked.
 */
function ToolRow({ entry }: { entry: Extract<Entry, { kind: "tool" }> }) {
  const isCall = entry.output === undefined;
  const delegate = entry.name === DELEGATE;
  const { pairs, raw } = isCall ? argumentPairs(entry.args) : { pairs: null, raw: entry.output };
  const hasDocument = raw !== undefined && raw !== "";
  const task = delegate && pairs ? pairs.find(([key]) => key === "task")?.[1] : undefined;
  return (
    <article className="grid min-w-0 gap-1 px-3 text-label data-[tool=output]:pl-[calc(var(--pad-3)+var(--pad-5))]" data-entry="tool" data-tool={isCall ? "call" : "output"} data-delegate={delegate || undefined}>
      <div className="flex min-w-0 items-start gap-2">
        {isCall ? <Icon name={delegate ? "lanes" : "tool"} size={12} className="mt-px flex-none" /> : null}
        <span className="min-w-0 wrap-break-word whitespace-pre-wrap">
          {isCall ? (
            delegate ? (
              <>
                <span className="text-ink">Asked the expert</span>
                {task ? <span className="ml-2">{task}</span> : null}
              </>
            ) : (
              <>
                <span className="font-mono text-ink">{entry.name}</span>
                {pairs && pairs.length ? (
                  <span className="ml-2">
                    {pairs.map(([key, value], index) => (
                      <span key={key}>
                        {index ? ", " : ""}
                        {key} <span className="text-ink">{value}</span>
                      </span>
                    ))}
                  </span>
                ) : pairs === null ? (
                  <span className="ml-2 font-mono text-micro">{entry.args}</span>
                ) : null}
              </>
            )
          ) : (
            <span className={entry.isError ? "text-failed-text" : undefined}>{entry.output}</span>
          )}
        </span>
      </div>
      {hasDocument ? (
        <Disclosure label={isCall ? (delegate ? "details" : "arguments") : "output"} data={isCall ? (delegate ? "details" : "arguments") : "output"}>
          <Document value={raw} />
        </Disclosure>
      ) : null}
    </article>
  );
}

function EntryView({ entry, agentName, onOpenTask }: { entry: Entry; agentName: string; onOpenTask: (taskId: string) => void }) {
  switch (entry.kind) {
    case "user":
      return (
        <div className="grid min-w-0 justify-items-end gap-1" data-entry="user" data-voice={entry.voice || undefined}>
          {entry.voice ? <VoiceMark /> : null}
          <div className="bubble" data-side="user">
            {entry.text}
          </div>
        </div>
      );
    case "agent":
      return (
        <div
          className="grid min-w-0 justify-items-start gap-1"
          data-entry="agent"
          data-streaming={entry.streaming ? "true" : undefined}
          data-voice={entry.voice || undefined}
          data-interrupted={entry.interrupted || undefined}
        >
          {entry.voice ? <VoiceMark /> : null}
          <div className="bubble" data-side="agent">
            {entry.text || (entry.streaming ? <Dots /> : "")}
            {entry.interrupted ? (
              <>
                {"..."}
                <span className="ml-2 align-baseline text-tag text-label" data-interrupted-tag="true">
                  interrupted
                </span>
              </>
            ) : null}
          </div>
        </div>
      );
    case "question":
      return (
        <div className="grid min-w-0 justify-items-start gap-1" data-entry="question">
          <span className="px-3 window-label text-label">{agentName ? `${agentName.split(" / ")[0]} asks` : "asks"}</span>
          <div className="bubble" data-side="agent">
            {entry.text}
          </div>
        </div>
      );
    case "tool":
      return <ToolRow entry={entry} />;
    case "delegation":
      return null;
    case "progress":
      return (
        <p className="m-0 flex min-h-row-sm min-w-0 items-start gap-2 px-3 text-label italic" data-entry="progress">
          <Icon name="activity" size={12} className="mt-px flex-none" />
          <span className="min-w-0 wrap-break-word">{entry.text}</span>
        </p>
      );
    case "status":
      if (!entry.taskId && !entry.failed)
        return (
          <p className="m-0 flex min-w-0 items-center gap-3 py-1 text-micro text-label before:h-px before:flex-1 before:bg-line-weak after:h-px after:flex-1 after:bg-line-weak" data-entry="status" data-divider="true">
            {entry.text}
          </p>
        );
      return (
        <p
          className="m-0 flex min-h-row-sm min-w-0 flex-wrap items-center gap-x-2 px-2 meta data-[failed=true]:text-failed-text"
          data-entry="status"
          data-state={entry.state}
          data-failed={entry.failed ? "true" : undefined}
        >
          {entry.taskId ? (
            <span className="inline-flex items-center">
              <CopyId id={entry.taskId} short={tail(entry.taskId)} className={entry.failed ? "text-failed-text" : ""} />
              <button
                type="button"
                className="btn btnQuiet size-row-sm p-0"
                title="Open this task in Tasks"
                aria-label="Open this task in Tasks"
                data-task-link={entry.taskId}
                onClick={() => onOpenTask(entry.taskId!)}
              >
                <Icon name="tasks" size={11} />
              </button>
            </span>
          ) : null}
          <span>{entry.text}</span>
          {entry.meta ? <span className="wrap-break-word">{entry.meta}</span> : null}
        </p>
      );
  }
}

/** A turn that was spoken, marked as such over its bubble. */
function VoiceMark() {
  return (
    <span className="flex items-center gap-1 px-3 window-label text-label">
      <Icon name="phone" size={8} />
      Voice
    </span>
  );
}

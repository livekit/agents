"use client";

import { useState } from "react";
import { Icon } from "../chrome/icon";
import { Document } from "../chrome/panel";
import { TERMINAL, tail, text } from "@/lib/format";
import { stateDot } from "@/lib/state";
import type { TaskLog } from "@/lib/use-session";
import type { Delegation, Progress, TaskRow } from "@/lib/types";

/** The A2A extension's directive on a task's status metadata, when the expert gave one. */
function directiveOf(task: TaskRow | undefined): string | null {
  const status = task?.status_message as { metadata?: Record<string, unknown> } | null;
  const meta = { ...(task?.data?.metadata as Record<string, unknown> | null), ...status?.metadata };
  for (const [key, value] of Object.entries(meta)) {
    if (key.endsWith("/directive") && value && typeof value === "object") {
      const kind = (value as { kind?: string; type?: string }).kind ?? (value as { type?: string }).type;
      if (kind) return kind;
    }
    if (key.endsWith("/directiveType") && typeof value === "string") return value;
  }
  return null;
}

/** mm:ss or s.s for an elapsed span in ms. */
function span(ms: number) {
  const seconds = Math.max(0, ms / 1000);
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

/**
 * What the card knows, merged from the two sources: the entry (the wire and
 * the history) and the expert's task row with its log (the session's
 * database, every two seconds). Either may be missing for a while.
 */
export function merge(entry: Delegation & { at: number }, task: TaskRow | undefined, log: TaskLog | undefined, now: number) {
  const state = task?.status ?? (entry.endedAt ? (entry.failed ? "failed" : "completed") : entry.progress.length ? "working" : "submitted");
  const terminal = TERMINAL.has(state);
  /* The wire's lines and the database's, as one list, each text once. */
  const progress: Progress[] = [];
  for (const line of [...entry.progress, ...(log?.log ?? [])].sort((a, b) => a.at - b.at)) if (!progress.some((known) => known.text === line.text)) progress.push(line);
  const artifacts = text(task?.data?.artifacts).trim();
  const answer = entry.answer ?? (terminal && artifacts ? artifacts : undefined);
  /* The final status message is the answer itself: it reads once, as the answer. */
  if (answer) progress.splice(0, progress.length, ...progress.filter((line) => line.text.trim() !== answer.trim()));
  const request = entry.task || text((task?.data?.history as unknown[] | null)?.[0]).trim();
  const endedAt = entry.endedAt ?? (terminal && task ? task.updated_at * 1000 : undefined);
  const startedAt = entry.at;
  return {
    state,
    terminal,
    progress,
    answer,
    request,
    taskId: entry.taskId ?? task?.task_id,
    lane: task?.lane,
    /* Unknown when the start is only when the console first saw an already ended task. */
    elapsed: endedAt !== undefined && endedAt < startedAt ? undefined : (endedAt ?? now) - startedAt,
    directive: directiveOf(task),
  };
}

/**
 * The delegation card: one quiet line in the thread (the expert, an A2A
 * tag, its state as a dot and a word, the elapsed time) that opens onto
 * what was asked (the task id and a JSON toggle at its right), the progress
 * lines as they arrive and the answer; the task's JSON opens under it all.
 * Open while it runs, folded once it ended unless the person opened it.
 */
export function DelegationCard({
  entry,
  task,
  log,
  now,
  onOpenTask,
}: {
  entry: Delegation & { id: string; at: number };
  task: TaskRow | undefined;
  log: TaskLog | undefined;
  now: number;
  onOpenTask: (taskId: string) => void;
}) {
  const view = merge(entry, task, log, now);
  const [opened, setOpened] = useState<boolean | null>(null);
  const [json, setJson] = useState(false);
  const open = opened ?? !view.terminal;
  const expert = view.lane ?? entry.endpoint?.split("/").filter(Boolean).pop() ?? "expert";
  /* While it works, only the latest line: the rest is history once it ended. */
  const progress = view.terminal ? view.progress : view.progress.slice(-1);

  return (
    <article
      className="-mb-[var(--pad-4)] grid min-w-0 overflow-hidden bg-panel data-[open=false]:justify-self-start data-[open=false]:rounded-full data-[open=true]:rounded-blender"
      data-entry="delegation"
      data-delegation-state={view.state}
      data-delegation-task={view.taskId ?? ""}
      data-open={open}
    >
      <button
        type="button"
        className="flex h-row min-w-0 items-center gap-2 px-3 text-left text-label hover:text-ink"
        aria-expanded={open}
        data-action={open ? "fold" : "unfold"}
        onClick={() => setOpened(!open)}
      >
        <Icon name="lanes" size={12} className="flex-none" />
        <span className="truncate text-ink" title={entry.endpoint ?? view.lane ?? ""} data-expert={expert}>
          {expert}
        </span>
        <span className="flex-none text-tag" data-badge={entry.profile ?? "a2a"}>
          A2A
        </span>
        {open ? <span className="flex-1" /> : null}
        <span className="inline-flex flex-none items-center gap-1" data-state={view.state}>
          <span className={`size-2 flex-none rounded-full ${stateDot(view.state)}`} aria-hidden="true" />
          {view.state}
        </span>
        {view.elapsed === undefined ? null : (
          <span className="flex-none tabular-nums" data-elapsed="true">
            {span(view.elapsed)}
          </span>
        )}
        <Icon name={open ? "disclosureDown" : "disclosureRight"} size={12} className="flex-none" />
      </button>
      {open ? (
        <div className="grid gap-2 px-3 pb-3">
          <div className="flex min-w-0 items-baseline gap-3">
            <p className="m-0 min-w-0 flex-1 wrap-break-word text-label">
              {view.request ? (
                <>
                  <span className="text-ink">Asked: </span>
                  {view.request}
                </>
              ) : null}
            </p>
            {view.taskId ? (
              <button type="button" className="quietLink flex-none font-mono text-micro" title="Open this task in Tasks" data-task-link={view.taskId} onClick={() => onOpenTask(view.taskId!)}>
                {tail(view.taskId)}
              </button>
            ) : null}
            {task ? (
              <button
                type="button"
                className="inline-flex h-row-sm flex-none items-center gap-1 self-center rounded-full border border-line px-2 text-micro text-ink hover:bg-raised"
                aria-expanded={json}
                data-disclosure="task"
                onClick={() => setJson(!json)}
              >
                JSON
                <Icon name={json ? "disclosureUp" : "disclosureDown"} size={10} />
              </button>
            ) : null}
          </div>
          {progress.map((line, index) => (
            <p key={`${line.at}-${index}`} className="m-0 flex min-w-0 items-center gap-2 text-label italic" data-progress={index}>
              <span className="min-w-0 wrap-break-word">{line.text}</span>
              {!view.terminal ? (
                <span className="flex items-center gap-1" aria-hidden="true">
                  <span className="thinkingDot" />
                  <span className="thinkingDot" />
                  <span className="thinkingDot" />
                </span>
              ) : null}
            </p>
          ))}
          {view.answer ? (
            <p className={`m-0 min-w-0 wrap-break-word whitespace-pre-wrap ${entry.failed ? "text-failed-text" : "text-ink"}`} data-answer="true">
              {view.answer}
            </p>
          ) : null}
          {view.directive ? (
            <p className="m-0 text-attention" data-directive={view.directive}>
              {view.directive.replace(/_/g, " ")}
            </p>
          ) : null}
          {task && json ? <Document value={task.data} /> : null}
        </div>
      ) : null}
    </article>
  );
}

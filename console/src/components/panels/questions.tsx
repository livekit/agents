"use client";

import { useMemo } from "react";
import { CopyId } from "../chrome/copy";
import { Panel } from "../chrome/panel";
import { DataTable, type Column } from "../chrome/table";
import { ago, tail, text } from "@/lib/format";
import type { TaskRow } from "@/lib/types";

/** One open question of a lane, as its `input-required` task row holds it. */
export type OpenQuestion = {
  task: TaskRow;
  question: string;
  /** In front of the person: the one their next input answers. */
  front: boolean;
  /** Waits unasked behind the one in front (`ask.queued`). */
  queued: boolean;
  /** The pending call that asked, when a conversation tool asked. */
  callId?: string;
  /** The tool that takes the answer, when the question named one: a delegate continues its expert's task. */
  answerTool?: string;
};

type Ask = { question?: unknown; queued?: unknown; call_id?: unknown; answer_tool?: unknown };

/**
 * A lane's open questions, oldest first (docs/delegation.md): its
 * `input-required` rows. The one in front of the person is the oldest that is
 * not queued; the person's next input continues its task.
 */
export function openQuestions(tasks: TaskRow[], lane: string): OpenQuestion[] {
  const open = tasks.filter((task) => task.lane === lane && task.status === "input-required").sort((a, b) => a.seq - b.seq);
  let fronted = false;
  return open.map((task) => {
    const ask = ((task.internal as { ask?: Ask } | null)?.ask ?? {}) as Ask;
    const queued = ask.queued === true;
    const front = !queued && !fronted;
    if (front) fronted = true;
    return {
      task,
      question: typeof ask.question === "string" ? ask.question : text(task.status_message),
      front,
      queued,
      callId: typeof ask.call_id === "string" ? ask.call_id : undefined,
      answerTool: typeof ask.answer_tool === "string" ? ask.answer_tool : undefined,
    };
  });
}

/**
 * The questions waiting on the person, lane by lane: the agent's own first,
 * with the one in front of the person marked, then the experts' questions,
 * which reach the person through it.
 */
export function QuestionsPanel({
  tasks,
  agent,
  now,
  empty,
  onSelect,
}: {
  tasks: TaskRow[] | null;
  agent: string;
  now: number;
  empty: string;
  onSelect: (taskId: string) => void;
}) {
  const rows = useMemo(() => {
    if (!tasks) return null;
    const lanes = [agent, ...new Set(tasks.map((task) => task.lane).filter((lane) => lane !== agent))];
    return lanes.flatMap((lane) => openQuestions(tasks, lane));
  }, [agent, tasks]);

  const columns = useMemo<Column<OpenQuestion>[]>(
    () => [
      {
        id: "place",
        label: "place",
        tip: "In front of the person (their next input answers it), or waiting behind it",
        width: 80,
        cell: (row) => (
          <span className={row.front ? "text-ink" : "text-label"} data-question={row.front ? "front" : row.queued ? "queued" : "open"}>
            {row.task.lane !== agent ? "expert's" : row.front ? "in front" : "waiting"}
          </span>
        ),
        text: (row) => (row.front ? "in front" : "waiting"),
      },
      { id: "question", label: "question", tip: "What the person is asked", width: 240, cell: (row) => row.question },
      {
        id: "task",
        label: "task_id",
        tip: "The task the answer continues",
        width: 84,
        cell: (row) => <CopyId id={row.task.task_id} short={tail(row.task.task_id)} />,
        text: (row) => row.task.task_id,
      },
      { id: "lane", label: "lane", tip: "Whose question it is", width: 68, mono: true, cell: (row) => row.task.lane },
      {
        id: "tool",
        label: "answer goes to",
        tip: "The tool that takes the answer as said; a delegate sends it to the expert's task",
        width: 110,
        mono: true,
        cell: (row) => row.answerTool ?? (row.callId ? "the call's output" : "the task's model"),
      },
      { id: "asked", label: "asked", tip: "When the row last changed", width: 64, mono: true, cell: (row) => ago(row.task.updated_at, now) },
    ],
    [agent, now],
  );

  return (
    <Panel id="questions" title="Questions" meta={rows ? `${rows.filter((row) => row.task.lane === agent).length} open` : undefined}>
      {!rows ? (
        empty ? <p className="m-0 flex h-row items-center px-4 text-label">{empty}</p> : null
      ) : (
        <DataTable
          id="questions"
          rows={rows}
          columns={columns}
          rowKey={(row) => row.task.task_id}
          rowData={(row) => ({ task: row.task.task_id, front: row.front || undefined })}
          selected={null}
          onSelect={(row) => onSelect(row.task.task_id)}
          empty="No open questions. When an agent asks the person something, its question waits here until it is answered, one in front of the person at a time."
          now={now}
        />
      )}
    </Panel>
  );
}

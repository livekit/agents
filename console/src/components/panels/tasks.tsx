"use client";

import { useEffect, useMemo } from "react";
import { CopyId } from "../chrome/copy";
import { Icon } from "../chrome/icon";
import { Document, Panel } from "../chrome/panel";
import { DataTable, type Column } from "../chrome/table";
import { FAILED, ago, clock, tail, taskFlags, text } from "@/lib/format";
import { stateDot } from "@/lib/state";
import type { TaskRow } from "@/lib/types";

/**
 * The A2A tasks of the session as a dense table: every run, seen from the
 * wire. The state is a dot and a word; a canceled, failed or rejected task
 * takes the failed colour; the merge and the interrupt are read off the
 * task's metadata. Clicking a row selects the task (the Stream view then
 * replays it); the chevron opens the task's status message, timing and
 * document under the row instead. A task opened from a turn's link is
 * selected and scrolled into view.
 */
export function TasksPanel({
  tasks,
  selected,
  now,
  empty,
  onSelect,
  onStream,
}: {
  tasks: TaskRow[] | null;
  selected: string | null;
  now: number;
  /** The one line shown when there is no database to read. */
  empty: string;
  onSelect: (taskId: string) => void;
  /** Open the selected task's stream. */
  onStream: () => void;
}) {
  useEffect(() => {
    if (!selected) return;
    document.querySelector(`[data-table='tasks'] [data-row='${selected}']`)?.scrollIntoView({ block: "nearest" });
  }, [selected, tasks]);

  const columns = useMemo<Column<TaskRow>[]>(
    () => [
      {
        id: "task",
        label: "task_id",
        tip: "The A2A task of the turn: the id the client and the stream are keyed by",
        width: 84,
        cell: (task) => <CopyId id={task.task_id} short={tail(task.task_id)} />,
        text: (task) => task.task_id,
      },
      { id: "lane", label: "lane", tip: "The agent whose lane the task ran on", width: 68, mono: true, cell: (task) => task.lane },
      {
        id: "state",
        label: "status",
        tip: "The A2A state the task is in: submitted, working, input-required, completed, failed, canceled, rejected",
        width: 110,
        cell: (task) => {
          const flags = taskFlags(task.data);
          return (
            <span className={`inline-flex items-center gap-1 ${FAILED.has(task.status) ? "text-failed-text" : ""}`} data-task-state={task.status}>
              <span className={`size-2 flex-none rounded-full ${stateDot(task.status)}`} aria-hidden="true" />
              {task.status}
              {flags.map((flag) => (
                <span key={flag} className="text-label" data-flag={flag.split(" ")[0]}>
                  {" "}
                  {flag}
                </span>
              ))}
            </span>
          );
        },
        text: (task) => [task.status, ...taskFlags(task.data)].join(" "),
      },
      {
        id: "updated",
        label: "updated_at",
        tip: "When the row last changed",
        width: 72,
        mono: true,
        cell: (task) => <span title={clock(task.updated_at)}>{ago(task.updated_at, now)}</span>,
        text: (task) => clock(task.updated_at),
        sort: (a, b) => a.updated_at - b.updated_at,
      },
      {
        id: "message",
        label: "status_message",
        tip: "What the task last said about itself: a progress line, a question, a reason",
        width: 160,
        cell: (task) => text(task.status_message).replace(/\s+/g, " "),
      },
    ],
    [now],
  );

  return (
    <Panel
      id="tasks"
      title="Tasks"
      meta={tasks ? `${tasks.length} tasks` : undefined}
      actions={
        selected ? (
          <button type="button" className="btn btnQuiet h-row-sm px-2" title="Replay the selected task's stream" data-action="open-stream" onClick={onStream}>
            <Icon name="stream" size={12} />
            stream
          </button>
        ) : undefined
      }
    >
      {!tasks ? (
        empty ? <p className="m-0 flex h-row items-center px-4 text-label">{empty}</p> : null
      ) : (
        <DataTable
          id="tasks"
          rows={tasks}
          columns={columns}
          rowKey={(task) => task.task_id}
          rowData={(task) => ({
            task: task.task_id,
            lane: task.lane,
            status: task.status,
            merged: taskFlags(task.data).some((flag) => flag.startsWith("merged")) || undefined,
          })}
          selected={selected}
          onSelect={(task) => onSelect(task.task_id)}
          fold={(task) => (
            <div className="grid gap-2">
              {text(task.status_message) ? <p className="m-0 whitespace-pre-wrap text-ink">{text(task.status_message)}</p> : null}
              <p className="m-0 meta">
                {[
                  `updated ${clock(task.updated_at)}`,
                  task.name ? `background ${task.name}` : "",
                  task.message_id ? `message ${task.message_id}` : "",
                ]
                  .filter(Boolean)
                  .join(", ")}
              </p>
              <Document value={task.data} />
            </div>
          )}
          empty="No tasks yet. Send a message and its task appears here, with the task of every agent it asks."
          now={now}
        />
      )}
    </Panel>
  );
}

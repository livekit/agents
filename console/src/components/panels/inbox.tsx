"use client";

import { useMemo } from "react";
import { CopyId } from "../chrome/copy";
import { Document, Panel } from "../chrome/panel";
import { DataTable, type Column } from "../chrome/table";
import { ago, clock, tail, text } from "@/lib/format";
import type { InboxRow, Lane } from "@/lib/types";

type Row = InboxRow & { drained: boolean };

/**
 * Messages received while a turn ran, waiting for the lane's lease holder to
 * drain them as one turn after a debounce (plan 3.3.4), as a table. A row
 * past the lane's drained_id has been answered; the pending ones are what
 * the panel is for. A row opens its message under it.
 */
export function InboxPanel({
  inbox,
  lanes,
  now,
  empty,
}: {
  inbox: InboxRow[] | null;
  lanes: Lane[];
  now: number;
  /** The one line shown when there is no database to read. */
  empty: string;
}) {
  const rows = useMemo<Row[]>(() => {
    const drainedSeq = new Map(lanes.map((lane) => [lane.lane, Number(lane.drained_id)]));
    return (inbox ?? []).map((row) => ({ ...row, drained: (drainedSeq.get(row.lane) ?? 0) >= Number(row.id) }));
  }, [inbox, lanes]);
  const pending = rows.filter((row) => !row.drained).length;

  const columns = useMemo<Column<Row>[]>(
    () => [
      { id: "id", label: "id", tip: "The message's place in the lane's inbox", width: 44, align: "right", mono: true, cell: (row) => row.id, sort: (a, b) => a.id - b.id },
      { id: "lane", label: "lane", tip: "The agent whose inbox it waited in", width: 64, mono: true, cell: (row) => row.lane },
      {
        id: "task",
        label: "task_id",
        tip: "The task the message was received as",
        width: 80,
        cell: (row) => <CopyId id={row.task_id} short={tail(row.task_id)} />,
        text: (row) => row.task_id,
      },
      {
        id: "state",
        label: "state",
        tip: "pending: not answered yet; drained: answered by a turn",
        width: 60,
        cell: (row) => (
          <span className={row.drained ? "text-label" : undefined} data-inbox-state={row.drained ? "drained" : "pending"}>
            {row.drained ? "drained" : "pending"}
          </span>
        ),
        text: (row) => (row.drained ? "drained" : "pending"),
      },
      {
        id: "received",
        label: "received_at",
        tip: "When it arrived",
        width: 72,
        mono: true,
        cell: (row) => <span title={clock(row.received_at)}>{ago(row.received_at, now)}</span>,
        text: (row) => clock(row.received_at),
        sort: (a, b) => a.received_at - b.received_at,
      },
      { id: "message", label: "message", tip: "The A2A message's text", width: 160, cell: (row) => text(row.message).replace(/\s+/g, " ") },
    ],
    [now],
  );

  return (
    <Panel
      id="inbox"
      title="Inbox"
      meta={inbox ? `${pending} pending of ${inbox.length}` : undefined}
    >
      {!inbox ? (
        empty ? <p className="m-0 flex h-row items-center px-4 text-label">{empty}</p> : null
      ) : (
        <DataTable
          id="inbox"
          rows={rows}
          columns={columns}
          rowKey={(row) => `${row.lane}:${row.id}`}
          rowData={(row) => ({ inbox: row.id, drained: row.drained, task: row.task_id })}
          fold={(row) => <Document value={row.message} />}
          group={(row) => (row.drained ? "drained" : "pending")}
          empty="Nothing has waited here. A message that arrives while a turn runs waits here."
          now={now}
        />
      )}
    </Panel>
  );
}

"use client";

import { useMemo } from "react";
import { CopyId } from "../chrome/copy";
import { Document, Panel } from "../chrome/panel";
import { DataTable, type Column, type Rail } from "../chrome/table";
import { ago, clockMs, lease, tail } from "@/lib/format";
import type { JournalEvent, Lane } from "@/lib/types";

type Row = JournalEvent & {
  /** run_started and run_ended carry the same mark, so the pair reads as one. */
  mark: { text: string; kind: "ended" | "running" | "open" } | null;
  /** Where the row sits in its run: a run reads as a bracket from run_started to run_ended. */
  rail: Rail;
};

/**
 * The lane's journal as a dense table, newest last: run_started and
 * run_ended paired by run id and drawn as one bracket down the left, with
 * every event between them inside it. An open run on a lane nobody leases
 * is a crash or a run the worker never closed, and takes the attention
 * colour; an open run under a held lease is a run in flight. The columns
 * are the row's own fields; the payload is in the fold under the row, not
 * summarised into a column.
 */
export function JournalPanel({
  events,
  lanes,
  now,
  empty,
}: {
  events: JournalEvent[] | null;
  lanes: Lane[];
  now: number;
  /** The one line shown when there is no database to read. */
  empty: string;
}) {
  const rows = useMemo<Row[]>(() => {
    const list = events ?? [];
    const started = new Map(list.filter((event) => event.type === "run_started").map((event) => [event.run_id, event]));
    const ended = new Map(list.filter((event) => event.type === "run_ended").map((event) => [event.run_id, event]));
    const heldLanes = new Set(lanes.filter((lane) => lease(lane.lease_until, now).state === "held").map((lane) => lane.lane));
    const kindOf = (runId: string, lane: string): "ended" | "running" | "open" => (ended.has(runId) ? "ended" : heldLanes.has(lane) ? "running" : "open");
    const open = new Set<string>();
    return list.map((event): Row => {
      let mark: Row["mark"] = null;
      let rail: Rail = null;
      if (event.type === "run_started") {
        const kind = kindOf(event.run_id, event.lane);
        const end = ended.get(event.run_id);
        mark = { text: kind === "ended" ? `#${event.id} to #${end!.id}` : kind, kind };
        rail = { place: kind === "ended" || kind === "running" ? "start" : "single", tone: kind === "open" ? "attention" : undefined };
        if (kind !== "open") open.add(event.run_id);
      } else if (event.type === "run_ended") {
        const kind = kindOf(event.run_id, event.lane);
        const start = started.get(event.run_id);
        mark = { text: `#${start?.id ?? "?"} to #${event.id}`, kind };
        rail = { place: start ? "end" : "single" };
        open.delete(event.run_id);
      } else if (open.size) {
        /* Inside a bracket; the attention tone follows an open run's start. */
        const inOpen = [...open].some((runId) => kindOf(runId, event.lane) === "open");
        rail = { place: "mid", tone: inOpen ? "attention" : undefined };
      }
      return { ...event, mark, rail };
    });
  }, [events, lanes, now]);

  const columns = useMemo<Column<Row>[]>(
    () => [
      { id: "id", label: "id", tip: "The event's place in the journal", width: 44, align: "right", mono: true, cell: (row) => row.id, sort: (a, b) => a.id - b.id },
      {
        id: "time",
        label: "ts",
        tip: "When it happened, on the server's clock; the exact time on hover",
        width: 64,
        mono: true,
        cell: (row) => <span title={clockMs(row.ts_ms)}>{ago(row.ts_ms / 1000, now)}</span>,
        text: (row) => clockMs(row.ts_ms),
        sort: (a, b) => a.ts_ms - b.ts_ms,
      },
      { id: "type", label: "type", tip: "What happened: a run starting or ending, usage, a tool, a handoff", width: 90, mono: true, cell: (row) => row.type },
      {
        id: "run",
        label: "run_id",
        tip: "The run the event belongs to: one turn of one agent",
        width: 76,
        cell: (row) => (row.run_id ? <CopyId id={row.run_id} short={tail(row.run_id)} /> : ""),
        text: (row) => row.run_id,
      },
      {
        id: "state",
        label: "run",
        tip: "Whether the run ended, is running under a held lease, or was left open",
        width: 92,
        cell: (row) =>
          row.mark ? (
            <span className={row.mark.kind === "open" ? "text-attention" : "text-label"} data-run-mark={row.mark.kind}>
              {row.mark.text}
            </span>
          ) : (
            ""
          ),
        text: (row) => row.mark?.text ?? "",
      },
      { id: "lane", label: "lane", tip: "The agent whose lane the event is on", width: 68, mono: true, cell: (row) => row.lane },
    ],
    [now],
  );

  return (
    <Panel
      id="journal"
      title="Journal"
      meta={events ? `${events.length} events` : undefined}
    >
      {!events ? (
        empty ? <p className="m-0 flex h-row items-center px-4 text-label">{empty}</p> : null
      ) : (
        <DataTable
          id="journal"
          rows={rows}
          columns={columns}
          rowKey={(row) => String(row.id)}
          rowData={(row) => ({ event: row.type, run: row.run_id })}
          fold={(row) => (row.payload == null ? null : <Document value={row.payload} />)}
          rail={(row) => row.rail}
          empty="No events yet. Send a message and its run appears here, from run_started to run_ended."
          now={now}
        />
      )}
    </Panel>
  );
}

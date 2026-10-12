"use client";

import { useMemo } from "react";
import { CopyId } from "../chrome/copy";
import { Document, Empty, Panel } from "../chrome/panel";
import { DataTable, type Column } from "../chrome/table";
import { clock, lease } from "@/lib/format";
import type { Lane } from "@/lib/types";

/**
 * The session's lanes as a table: one row per agent in the session, its
 * lease as a state (held and expiring, free, or lapsed, the one thing here
 * that asks for attention), who holds or last held it, who waits, how far
 * its inbox is drained. A row opens on its state and agent stack. The
 * session's id and the database's tip sit in the header.
 */
export function LanesPanel({
  lanes,
  now,
  tip,
  sessionId,
  empty,
}: {
  lanes: Lane[] | null;
  now: number;
  tip?: number;
  sessionId: string | null;
  /** The one line shown when there is no database to read. */
  empty: string;
}) {
  const columns = useMemo<Column<Lane>[]>(
    () => [
      { id: "lane", label: "lane", tip: "The agent this lane belongs to", width: 120, cell: (lane) => lane.lane, sort: (a, b) => a.lane.localeCompare(b.lane) },
      {
        id: "lease",
        label: "lease",
        tip: "Held while a worker writes the lane; lapsed when its holder stopped renewing",
        width: 140,
        cell: (lane) => {
          const held = lease(lane.lease_until, now);
          return (
            <span className="inline-flex items-center gap-1 data-[state=lapsed]:text-attention" data-state={held.state} data-lease-state={held.state} title={lane.lease_until ? `lease until ${clock(lane.lease_until)}` : undefined}>
              <span className="size-2 flex-none rounded-full bg-control data-[state=held]:bg-label data-[state=lapsed]:bg-attention" data-state={held.state} aria-hidden="true" />
              {held.state === "free" ? "free" : held.label}
            </span>
          );
        },
        text: (lane) => lease(lane.lease_until, now).label,
      },
      {
        id: "holder",
        label: "holder",
        tip: "The worker that holds or last held the lease",
        width: 200,
        mono: true,
        cell: (lane) => <span data-lease-holder={lane.lease_holder ?? ""}>{lane.lease_holder ?? "none"}</span>,
        text: (lane) => lane.lease_holder ?? "",
      },
      { id: "waiter", label: "waiter", tip: "A worker waiting for the lease", width: 120, mono: true, cell: (lane) => lane.waiter ?? "none" },
      { id: "drained", label: "drained", tip: "The last inbox message this lane answered", width: 64, align: "right", mono: true, cell: (lane) => lane.drained_id },
      { id: "version", label: "version", tip: "How many times the lane was written", width: 64, align: "right", mono: true, cell: (lane) => lane.version, sort: (a, b) => a.version - b.version },
    ],
    [now],
  );

  return (
    <Panel
      id="lanes"
      title="Lanes"
      meta={
        sessionId ? (
          <span className="inline-flex items-center gap-3" data-session-summary="true">
            <span className="inline-flex items-center gap-1">
              session
              <span data-session-id={sessionId}>
                <CopyId id={sessionId} short={sessionId} />
              </span>
            </span>
            {tip != null ? <span data-tip={tip}>tip {tip}</span> : null}
          </span>
        ) : undefined
      }
    >
      {!lanes ? (
        empty ? <Empty>{empty}</Empty> : null
      ) : (
        <DataTable
          id="lanes"
          rows={lanes}
          columns={columns}
          rowKey={(lane) => lane.lane}
          rowData={(lane) => ({ lane: lane.lane, lease: lease(lane.lease_until, now).state })}
          fold={(lane) =>
            lane.state != null || lane.agent_stack != null ? (
              <div className="grid gap-2">
                {lane.state != null ? <Document value={lane.state} /> : null}
                {lane.agent_stack != null ? <Document value={lane.agent_stack} /> : null}
              </div>
            ) : null
          }
          empty="No lanes yet. The agent's lane appears with the first message."
          now={now}
        />
      )}
    </Panel>
  );
}

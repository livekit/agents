"use client";

import { Field, Panel } from "../chrome/panel";
import type { Grant } from "@/lib/types";

/** What the page is talking to: the grant's subject and agent, the endpoint and whether streams are there. */
export function InfoPanel({ grant, streamsOk }: { grant: Grant | null; streamsOk: boolean }) {
  return (
    <Panel id="info" title="Info">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(var(--w-info),1fr))] gap-x-8 gap-y-6 px-4 pt-2 pb-4" data-info="true">
        <section className="grid content-start gap-1" data-info-grant="true">
          <h3 className="m-0 pb-1 text-ink">Connection</h3>
          <Field label="Subject">{grant?.sub ?? ""}</Field>
          <Field label="Agent">{grant?.agent ?? ""}</Field>
          <Field label="Endpoint">{grant?.endpoint ?? ""}</Field>
          <Field label="Streams" tone={streamsOk ? undefined : "attention"}>
            {streamsOk ? "available" : "not available"}
          </Field>
        </section>
      </div>
    </Panel>
  );
}

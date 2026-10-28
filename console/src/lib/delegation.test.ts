import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { referencedTasks } from "./delegation";
import { DelegationCard } from "@/components/panels/delegation";
import type { TaskRow } from "./types";
it("renders only explicitly referenced work with the agent's name", () => {
  const make = (id: string, lane: string): TaskRow => ({ task_id: id, lane, seq: 1, name: null, message_id: null, status: "completed", updated_at: 100, ask: null, data: {} });
  const parent = make("parent", "desk");
  parent.data.history = [{ messageId: "m", role: "ROLE_AGENT", parts: [{ text: "Checking" }], referenceTaskIds: ["expert"] }];
  const expert = make("expert", "coverage");
  const unrelated = make("other", "delegate");
  const tasks = referencedTasks([parent, expert, unrelated], "desk");
  expect(tasks).toEqual([expert]);
  const html = renderToStaticMarkup(createElement(DelegationCard, { entry: { id: "e", at: 1, name: expert.lane, task: "Check coverage", progress: [] }, task: expert, log: undefined, now: 200, onOpenTask() {} }));
  expect(html).toContain("coverage");
  expect(html).not.toContain("delegate");
});

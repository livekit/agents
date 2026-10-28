import { expect, it } from "vitest";
import { merge } from "./delegation";
import type { TaskRow } from "@/lib/types";
it("uses task times in milliseconds and does not promote private metadata to user instructions", () => {
  const task: TaskRow = { task_id: "t", lane: "coverage", seq: 1, name: null, message_id: null, status: "completed", updated_at: 200, ask: null, data: { metadata: { "private/directiveType": "secret instruction" } } };
  const view = merge({ name: "coverage", task: "check coverage", progress: [], at: 100 }, task, undefined, 300);
  expect(view.elapsed).toBe(100);
  expect(view).not.toHaveProperty("directive");
});

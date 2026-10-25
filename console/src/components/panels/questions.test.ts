import { describe, expect, it } from "vitest";
import { openQuestions } from "./questions";
import type { TaskRow } from "@/lib/types";

const row = (task_id: string, seq: number, status: string, ask: Record<string, unknown>, lane = "desk"): TaskRow => ({
  task_id,
  lane,
  seq,
  name: null,
  message_id: null,
  status,
  status_message: null,
  updated_at: 0,
  data: null,
  internal: { ask },
});

describe("openQuestions", () => {
  it("puts the oldest question that is not queued in front of the person", () => {
    const questions = openQuestions(
      [
        row("AST_3", 3, "input-required", { question: "Window or aisle?", queued: true }),
        row("AST_1", 1, "completed", { question: "done" }),
        row("AST_2", 2, "input-required", { question: "Which email?", call_id: "c1", answer_tool: "ask" }),
        row("AST_9", 9, "input-required", { question: "the expert's own" }, "backend"),
      ],
      "desk",
    );
    expect(questions.map((q) => [q.task.task_id, q.front, q.queued])).toEqual([
      ["AST_2", true, false],
      ["AST_3", false, true],
    ]);
    expect(questions[0].answerTool).toBe("ask");
  });
});

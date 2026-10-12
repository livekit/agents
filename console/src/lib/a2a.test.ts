import { describe, expect, it } from "vitest";
import { cutEvents, extensionMeta, interpret, key } from "./a2a";

describe("cutEvents", () => {
  it("returns only complete events and keeps the tail", () => {
    const { events, rest } = cutEvents('data: {"a":1}\n\nevent: x\ndata: {"b":\ndata: 2}\n\ndata: {"c"');
    expect(events).toEqual(['{"a":1}', '{"b":2}']);
    expect(rest).toBe('data: {"c"');
  });
});

describe("interpret", () => {
  it("turns a progress status and an artifact into stage events", () => {
    expect(
      interpret({
        result: {
          statusUpdate: {
            taskId: "AST_1",
            status: { state: "TASK_STATE_WORKING", message: { parts: [{ text: "Signing in" }] } },
          },
        },
      }),
    ).toEqual([{ type: "progress", text: "Signing in" }]);
    expect(
      interpret({ result: { artifactUpdate: { taskId: "AST_1", artifact: { parts: [{ text: "Hi" }] }, append: true } } }),
    ).toEqual([{ type: "artifact", text: "Hi", append: true }]);
  });

  it("surfaces the question of an input-required turn and the ending's metadata", () => {
    const events = interpret({
      result: {
        statusUpdate: {
          taskId: "AST_2",
          final: true,
          status: { state: "TASK_STATE_INPUT_REQUIRED", message: { parts: [{ text: "Which pharmacy?" }] } },
        },
      },
    });
    expect(events).toEqual([
      { type: "question", text: "Which pharmacy?" },
      { type: "ended", taskId: "AST_2", state: "TASK_STATE_INPUT_REQUIRED", meta: {} },
    ]);
    const canceled = interpret({
      result: {
        statusUpdate: {
          taskId: "AST_3",
          status: { state: "TASK_STATE_CANCELED" },
          metadata: { [key("interrupted")]: true, [key("interruptionReason")]: "new-message", other: 1 },
        },
      },
    });
    expect(canceled).toEqual([
      {
        type: "ended",
        taskId: "AST_3",
        state: "TASK_STATE_CANCELED",
        meta: { interrupted: true, interruptionReason: "new-message" },
      },
    ]);
  });

  it("carries a replayed terminal task's answer", () => {
    const [event] = interpret({
      result: {
        task: {
          id: "AST_4",
          contextId: "AT_1",
          status: { state: "TASK_STATE_COMPLETED" },
          artifacts: [{ parts: [{ text: "Done" }] }],
        },
      },
    });
    expect(event).toEqual({ type: "task", taskId: "AST_4", contextId: "AT_1", state: "TASK_STATE_COMPLETED", artifactText: "Done" });
  });

  it("keeps only the extension's keys, shortened", () => {
    expect(extensionMeta({ [key("mergedInto")]: "AST_9", x: 1 })).toEqual({ mergedInto: "AST_9" });
  });
});

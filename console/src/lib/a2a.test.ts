import { describe, expect, it } from "vitest";
import { cutEvents, extensionMeta, interpret, key } from "./a2a";

describe("cutEvents", () => {
  it("returns only complete events and keeps the tail", () => {
    const { events, rest } = cutEvents('id: 3\ndata: {"a":1}\n\nevent: x\ndata: {"b":\ndata: 2}\n\ndata: {"c"');
    expect(events).toEqual([{ data: '{"a":1}', id: "3" }, { data: '{"b":2}' }]);
    expect(rest).toBe('data: {"c"');
  });
});

describe("interpret", () => {
  it("tells progress, a line said as written, and the end of a reply beside work apart", () => {
    const working = (message?: object, metadata?: Record<string, unknown>) =>
      interpret({ result: { statusUpdate: { taskId: "AST_1", status: { state: "TASK_STATE_WORKING", message }, metadata } } });
    expect(working({ parts: [{ text: "Signing in" }], metadata: { [key("progress")]: true } })).toEqual([{ type: "progress", text: "Signing in" }]);
    expect(working({ parts: [{ text: "Charged $210" , metadata: { [key("verbatim")]: true } }] })).toEqual([{ type: "said", text: "Charged $210" }]);
    expect(working({ parts: [{ text: "Three hotels" }] }, { [key("background")]: true })).toEqual([{ type: "replied" }]);
    expect(working({ parts: [{ text: "unflagged" }] })).toEqual([]);
  });

  it("names the artifact a chunk belongs to", () => {
    expect(
      interpret({ result: { artifactUpdate: { taskId: "AST_1", artifact: { artifactId: "art_1", parts: [{ text: "Hi" }] }, append: true, lastChunk: true } } }),
    ).toEqual([{ type: "artifact", id: "art_1", text: "Hi", append: true, last: true }]);
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

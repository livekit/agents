import { describe, expect, it } from "vitest";
import { lease, taskFlags, text } from "./format";

describe("text", () => {
  it("reads every shape the API hands back", () => {
    expect(text("hi")).toBe("hi");
    expect(text(["a", { text: "b" }])).toBe("a\nb");
    expect(text({ parts: [{ kind: "text", text: "q" }] })).toBe("q");
    expect(text({ message: { parts: [{ text: "m" }] } })).toBe("m");
    expect(text({ content: ["c"] })).toBe("c");
    expect(text(null)).toBe("");
  });
});

describe("lease", () => {
  it("tells free, held and lapsed apart on the server's clock", () => {
    const now = 1_000_000_000;
    expect(lease(0, now).state).toBe("free");
    expect(lease(now / 1000 + 5, now).state).toBe("held");
    expect(lease(now / 1000 - 5, now)).toEqual({ state: "lapsed", label: "lapsed 5s ago" });
  });
});

describe("taskFlags", () => {
  it("surfaces the merge and the interrupt", () => {
    expect(taskFlags({ metadata: { "livekit/mergedInto": "AST_x" } })).toEqual(["merged into AST_x"]);
    expect(taskFlags({ metadata: { "livekit/answeredWith": "AST_y" } })).toEqual(["answered with AST_y"]);
    expect(
      taskFlags({ metadata: { "livekit/interrupted": true, "livekit/interruptionReason": "new-message" } }),
    ).toEqual(["interrupted: new-message"]);
    expect(taskFlags(null)).toEqual([]);
    expect(
      taskFlags({ metadata: { "livekit/background": true, "livekit/queued": true, "livekit/cancelRequested": true } }),
    ).toEqual(["in the background", "queued", "cancel requested"]);
    expect(taskFlags({ metadata: { "livekit/background": false, "livekit/queued": false } })).toEqual([]);
  });
});

it("text reads the parts of artifacts and messages inside a list", () => {
  expect(text([{ artifactId: "art_1", parts: [{ kind: "text", text: "Coverage is active." }] }])).toBe("Coverage is active.");
  expect(text([{ messageId: "m1", role: "user", parts: [{ kind: "text", text: "aetna member 1" }] }])).toBe("aetna member 1");
});

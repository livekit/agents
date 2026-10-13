import { describe, expect, it } from "vitest";
import { decodeData, summarize } from "./stream";

describe("the stream panel", () => {
  it("summarises A2A frames the way the relay ships them", () => {
    expect(summarize({ task: { id: "AST_1", status: { state: "TASK_STATE_SUBMITTED" } } })).toBe("task submitted");
    expect(summarize({ artifactUpdate: { artifact: { parts: [{ text: "Hello" }] }, append: true } })).toBe("artifact +Hello");
    expect(
      summarize({ statusUpdate: { status: { state: "TASK_STATE_WORKING", message: { parts: [{ text: "Signing in" }] } } } }),
    ).toBe("status working Signing in");
  });

  it("decodes a JSON array, and length-prefixed records otherwise", () => {
    expect(decodeData('[{"a":1}]')).toEqual([{ a: 1 }]);
    const record = new TextEncoder().encode('{"b":2}');
    const bytes = new Uint8Array(4 + record.length);
    bytes[3] = record.length;
    bytes.set(record, 4);
    expect(decodeData(btoa(String.fromCharCode(...bytes)))).toEqual([{ b: 2 }]);
  });
});

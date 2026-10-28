import { expect, it } from "vitest";
import { summarize } from "./stream";
it("renders replayed progress and replies from the validated A2A records", () => {
  expect(summarize({ result: { artifactUpdate: { taskId: "t", contextId: "c", artifact: { artifactId: "a", parts: [{ text: "Hello" }] }, append: true } } })).toBe("artifact: Hello");
});

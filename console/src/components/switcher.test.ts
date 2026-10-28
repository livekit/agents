import { expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ExamplePicker } from "./switcher";
import type { Example } from "@/lib/examples";
it("names the agent in ordinary copy and leaves endpoint details to the inspector", () => {
  const example: Example = { agent: "desk", endpoint: "chat", name: "Coverage assistant", description: "Helps with coverage", voice: false, start: "uv run main.py dev", dir: "examples/frontdesk", url: "https://internal.example/agents/desk/default/chat", live: true };
  const html = renderToStaticMarkup(createElement(ExamplePicker, { examples: [example], selected: example, fallback: "Coverage assistant", onSelect() {} }));
  expect(html).toContain("Coverage assistant");
  expect(html).not.toContain("desk / chat");
  expect(html).not.toContain("internal.example");
});

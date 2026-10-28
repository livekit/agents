import { getJSON } from "./api";
import { decode } from "./decode";
export type { Example, Examples } from "./generated/Examples";
import type { Example } from "./generated/Examples";

/** An agent and its endpoint: what a thread was started on. */
export type Selection = { agent: string; endpoint: string };

export const listExamples = (signal?: AbortSignal) => getJSON("/examples", decode.examples, { signal });

export const sameExample = (example: Example, selection: Selection | null) =>
  selection !== null && example.agent === selection.agent && example.endpoint === selection.endpoint;

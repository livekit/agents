import { getJSON } from "./api";
export type { Example, Examples } from "./types";
import type { Example, Examples } from "./types";

/** An agent and its endpoint: what a thread was started on. */
export type Selection = { agent: string; endpoint: string };

export const listExamples = (signal?: AbortSignal) => getJSON<Examples>("/examples", { signal });

export const sameExample = (example: Example, selection: Selection | null) =>
  selection !== null && example.agent === selection.agent && example.endpoint === selection.endpoint;

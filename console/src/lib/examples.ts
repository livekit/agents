import { API_URL, HttpError } from "./api";

/** One `examples/<name>/console.toml`, and whether its endpoint answers. */
export type Example = {
  name: string;
  agent: string;
  /** The endpoint's name on the agent, as threads are tagged with it. */
  endpoint: string;
  description: string;
  /** It also takes a call. */
  voice: boolean;
  /** The shell command that starts its workers, run from the example's folder. */
  start: string;
  /** Its folder from the repository root, where `start` runs. */
  dir: string;
  /** The endpoint's URL, for the A2A client. */
  url: string;
  live: boolean;
};

/** An agent and its endpoint: what a thread was started on. */
export type Selection = { agent: string; endpoint: string };

export type Examples = { examples: Example[] };

/** Every example, and whether each one runs. */
export async function listExamples() {
  const response = await fetch(`${API_URL}/api/examples`, { cache: "no-store" });
  const body = (await response.json().catch(() => ({ error: response.statusText }))) as
    | Examples
    | { error?: string };
  if (!response.ok)
    throw new HttpError(response.status, (body as { error?: string }).error ?? response.statusText);
  return body as Examples;
}

export const sameExample = (example: Example, selection: Selection | null) =>
  selection !== null && example.agent === selection.agent && example.endpoint === selection.endpoint;

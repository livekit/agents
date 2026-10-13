import type { Grant, SessionDump, SessionRow, ConsoleConfig } from "./types";

/**
 * The console API (`examples/console_api.py`): the demo grant, the subject's
 * threads, one thread's database as a dump, and the stream relay. The
 * console talks to it cross-origin; the API allows the dev origin.
 */
export const API_URL = (
  process.env.NEXT_PUBLIC_AGENTDB_DEBUG_URL ?? "http://localhost:7795"
).replace(/\/$/, "");

/** A response that was not ok, with its status: a 404 is a session that is gone, not a fault. */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  const body = (await response.json().catch(() => ({ error: response.statusText }))) as
    | T
    | { error?: string };
  if (!response.ok)
    throw new HttpError(response.status, (body as { error?: string }).error ?? response.statusText);
  return body as T;
}

export const getConfig = () => getJSON<ConsoleConfig>("/api/config");

/** The project credential for one agent's endpoint, under the subject. */
export const getGrant = (sub: string, agent: string, endpoint: string) =>
  getJSON<Grant>(`/api/token?${new URLSearchParams({ sub, agent, endpoint })}`);

export async function listSessions(agent: string, endpoint: string, sub: string) {
  const query = new URLSearchParams({ agent, endpoint, sub, limit: "50" });
  const out = await getJSON<{ sessions: SessionRow[] }>(`/api/sessions?${query}`);
  return out.sessions;
}

/** A new thread under the subject on the agent's endpoint, so the first message carries its id. */
export async function createSession(sub: string, agent: string, endpoint: string) {
  const response = await fetch(`${API_URL}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sub, agent, endpoint }),
  });
  const body = (await response.json().catch(() => ({ error: response.statusText }))) as
    | { session: SessionRow }
    | { error?: string };
  if (!response.ok) throw new Error((body as { error?: string }).error ?? response.statusText);
  return (body as { session: SessionRow }).session;
}

/** The agent's chat items from the threads service, oldest first. */
export const getHistory = (id: string, agent: string) =>
  getJSON<{ items: Record<string, unknown>[] }>(
    `/api/sessions/${encodeURIComponent(id)}/history?${new URLSearchParams({ agent })}`,
  );

export const getSession = (id: string) =>
  getJSON<SessionDump>(`/api/sessions/${encodeURIComponent(id)}`);

/**
 * The session dump over SSE, pushed every two seconds by the API, so the
 * page keeps no polling loop. Returns the function that stops watching.
 */
export function watchSession(
  id: string,
  onDump: (dump: SessionDump) => void,
  onError: (message: string) => void,
) {
  const source = new EventSource(`${API_URL}/api/sessions/${encodeURIComponent(id)}/watch`);
  source.addEventListener("dump", (event) => {
    try {
      onDump(JSON.parse((event as MessageEvent).data) as SessionDump);
    } catch (error) {
      onError(String(error));
    }
  });
  source.addEventListener("error", (event) => {
    const data = (event as MessageEvent).data;
    if (typeof data === "string") {
      try {
        onError((JSON.parse(data) as { error?: string }).error ?? data);
        return;
      } catch {
        /* a transport error carries no data */
      }
    }
    onError(source.readyState === EventSource.CLOSED ? "watch closed" : "watch reconnecting");
  });
  return () => source.close();
}

/**
 * Names a conversation after its first turn: the console's own backend
 * (`app/api/title`) asks a model and writes the title on the registry row.
 * Returns the title, the existing one when the session already had one.
 */
export async function nameSession(sessionId: string, firstMessage: string) {
  const response = await fetch("/api/title", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, firstMessage }),
  });
  const body = (await response.json().catch(() => ({}))) as { title?: string; error?: string };
  if (!response.ok) throw new Error(body.error ?? response.statusText);
  return body.title ?? "";
}

export const streamUrl = (id: string, offset = -1) =>
  `${API_URL}/api/streams/${encodeURIComponent(id)}?offset=${offset}&live=sse`;

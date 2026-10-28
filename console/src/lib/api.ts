import { decode } from "./decode";

export const API_URL = "/api/console";
export class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
export async function getJSON<T>(path: string, parse: (value: unknown) => T, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw new HttpError(response.status, `Request failed (HTTP ${response.status})`);
  return parse(await response.json());
}
export const getConfig = () => getJSON("/config", decode.config);
export const getGrant = (sub: string, agent: string, endpoint: string, signal?: AbortSignal) =>
  getJSON(`/token?${new URLSearchParams({ sub, agent, endpoint })}`, decode.grant, { signal });
export async function listSessions(agent: string, endpoint: string, sub: string) {
  return (await getJSON(`/sessions?${new URLSearchParams({ agent, endpoint, sub, limit: "50" })}`, decode.sessions)).sessions;
}
export const getHistory = (id: string, agent: string) =>
  getJSON(`/sessions/${encodeURIComponent(id)}/history?${new URLSearchParams({ agent })}`, decode.history);
export const getSession = (id: string, signal?: AbortSignal) =>
  getJSON(`/admin/sessions/${encodeURIComponent(id)}`, decode.dump, { signal });

/** A bounded inspector snapshot refresh. Disposal cancels the in-flight request. */
export function watchSession(id: string, onDump: (dump: ReturnType<typeof decode.dump>) => void, onError: (message: string) => void) {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const refresh = async () => {
    try {
      const dump = await getSession(id, controller.signal);
      if (!controller.signal.aborted) onDump(dump);
    } catch {
      if (!controller.signal.aborted) onError("The inspector could not refresh this conversation.");
    } finally {
      if (!controller.signal.aborted) timer = setTimeout(() => void refresh(), 2000);
    }
  };
  void refresh();
  return () => { controller.abort(); clearTimeout(timer); };
}
export async function nameSession(sessionId: string, firstMessage: string) {
  const response = await fetch("/api/title", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sessionId, firstMessage }) });
  if (!response.ok) throw new Error("Conversation title could not be saved");
  const value: unknown = await response.json();
  if (!value || typeof value !== "object" || !("title" in value) || typeof value.title !== "string") throw new Error("Invalid title response");
  return value.title;
}
export const querySnapshot = (id: string, sql: string, limit: number, signal: AbortSignal) =>
  getJSON(`/admin/sessions/${encodeURIComponent(id)}/query`, decode.query, {
    method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sql, limit }),
  });

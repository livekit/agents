import type { ConsoleConfig, Grant, Sessions, History, SessionDump, QueryResult } from "./types";

export const API_URL = "/api/console";
export class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw new HttpError(response.status, `Request failed (HTTP ${response.status})`);
  return response.json() as Promise<T>;
}
export const getConfig = () => getJSON<ConsoleConfig>("/config");
export const getGrant = (sub: string, agent: string, endpoint: string, signal?: AbortSignal) =>
  getJSON<Grant>(`/token?${new URLSearchParams({ sub, agent, endpoint })}`, { signal });
export async function listSessions(agent: string, endpoint: string, sub: string) {
  return (await getJSON<Sessions>(`/sessions?${new URLSearchParams({ agent, endpoint, sub, limit: "50" })}`)).sessions;
}
export const getHistory = (id: string, agent: string) =>
  getJSON<History>(`/sessions/${encodeURIComponent(id)}/history?${new URLSearchParams({ agent })}`);
export const getSession = (id: string, signal?: AbortSignal) =>
  getJSON<SessionDump>(`/admin/sessions/${encodeURIComponent(id)}`, { signal });

/** A bounded inspector snapshot refresh. Disposal cancels the in-flight request. */
export function watchSession(id: string, onDump: (dump: SessionDump) => void, onError: (message: string) => void) {
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
  getJSON<QueryResult>(`/admin/sessions/${encodeURIComponent(id)}/query`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sql, limit }),
  });

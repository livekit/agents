/** Local developer backend. This secret is never included in a browser response. */
export const backendURL = (process.env.CONSOLE_API_URL ?? "http://127.0.0.1:7795").replace(/\/$/, "");
export function backendHeaders() {
  const token = process.env.CONSOLE_API_TOKEN;
  if (!token) throw new Error("CONSOLE_API_TOKEN is not configured");
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

/** The console runs locally; reject cross-origin requests to its privileged routes. */
export function localDeveloper(request: Request) {
  const url = new URL(request.url);
  const allowed = process.env.CONSOLE_ORIGIN ?? "http://localhost:3000";
  const origin = request.headers.get("origin");
  return url.origin === allowed && (!origin || origin === allowed) && request.headers.get("sec-fetch-site") !== "cross-site";
}

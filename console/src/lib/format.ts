/** Small formatters shared by the panels. Pure, so they are unit-tested. */

export function short(id: string | null | undefined, keep = 5) {
  if (!id) return "";
  return id.length > 18 ? `${id.slice(0, 10)}..${id.slice(-keep)}` : id;
}

/** The end of an id, for a row that shows the whole id on hover. */
export function tail(id: string | null | undefined, keep = 6) {
  if (!id) return "";
  return id.length > keep ? id.slice(-keep) : id;
}

export function ago(sec: number | null | undefined, now = Date.now()) {
  if (!sec) return "";
  const d = Math.round(now / 1000 - sec);
  if (d < 5) return "now";
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.round(d / 60)}m ago`;
  if (d < 86400) return `${Math.round(d / 3600)}h ago`;
  return `${Math.round(d / 86400)}d ago`;
}

export function inSec(sec: number | null | undefined, now = Date.now()) {
  if (!sec) return "";
  const d = Math.round(sec - now / 1000);
  if (d <= 0) return `lapsed ${-d}s ago`;
  if (d < 90) return `in ${d}s`;
  if (d < 3600) return `in ${Math.round(d / 60)}m`;
  return `in ${Math.round(d / 3600)}h`;
}

export function clock(sec: number | null | undefined) {
  if (!sec) return "";
  return new Date(sec * 1000).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function clockMs(ms: number | null | undefined) {
  if (!ms) return "";
  const date = new Date(ms);
  return `${clock(ms / 1000)}.${String(date.getMilliseconds()).padStart(3, "0")}`;
}

export function pretty(value: unknown) {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

/**
 * The text of a chat item's content, a task's status message or an A2A
 * message: strings, arrays of strings or parts, objects with `text`,
 * `content` or `parts`. Anything else is its JSON.
 */
export function text(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value))
    return value
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object") {
          const p = part as Record<string, unknown>;
          if (typeof p.text === "string") return p.text;
          /* An artifact or a message inside a list: its parts are the text. */
          if (Array.isArray(p.parts)) return text(p.parts);
          if (p.content !== undefined) return text(p.content);
          if (p.data !== undefined) return pretty(p.data);
        }
        return pretty(part);
      })
      .filter(Boolean)
      .join("\n");
  if (typeof value === "object") {
    const v = value as Record<string, unknown>;
    if (Array.isArray(v.parts)) return text(v.parts);
    if (v.message !== undefined) return text(v.message);
    if (typeof v.text === "string") return v.text;
    if (v.content !== undefined) return text(v.content);
  }
  return pretty(value);
}

/** The lease of a lane, from the server's clock: free, held with time left, or lapsed. */
export function lease(
  until: number,
  now: number,
): { state: "free" | "held" | "lapsed"; label: string } {
  if (!until) return { state: "free", label: "free" };
  const left = until - now / 1000;
  if (left <= 0) return { state: "lapsed", label: `lapsed ${Math.round(-left)}s ago` };
  return { state: "held", label: `${left.toFixed(1)}s left` };
}

/** The `livekit/*` metadata of a task row's `task` document: mergedInto, interrupted. */
export function taskFlags(task: unknown): string[] {
  const out: string[] = [];
  if (!task || typeof task !== "object") return out;
  const metadata = (task as { metadata?: Record<string, unknown> }).metadata;
  if (!metadata) return out;
  /* The agents extension's keys, whatever URI they are under. */
  const meta = (name: string) => Object.entries(metadata).find(([key]) => key.endsWith(`/${name}`))?.[1];
  if (meta("mergedInto")) out.push(`merged into ${short(String(meta("mergedInto")))}`);
  if (meta("interrupted")) out.push(`interrupted${meta("interruptionReason") ? `: ${meta("interruptionReason")}` : ""}`);
  return out;
}

export const TERMINAL = new Set(["completed", "failed", "canceled", "rejected"]);
export const FAILED = new Set(["failed", "canceled", "rejected"]);

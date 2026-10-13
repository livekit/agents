import type { SessionRow } from "@/lib/types";

/** What a session is called: the title on the registry row, else its first message, else nothing yet. */
export function sessionTitle(session: SessionRow | undefined, firstMessage?: string | null) {
  const title = session?.attributes?.title?.trim();
  if (title) return title;
  const first = firstMessage?.trim();
  if (first) return first.length > 48 ? `${first.slice(0, 47).trimEnd()}...` : first;
  return "New conversation";
}

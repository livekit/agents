import { NextResponse } from "next/server";

/**
 * Names a conversation: the web app's concern, not the agent's. Once a
 * session has its first completed turn and no `attributes.title` yet, the
 * page posts the session id and the first user message here; a model is
 * asked for a three to six word title (OpenAI, with `OPENAI_API_KEY` from
 * the environment the console reads), or, with no key, the message cut to
 * 40 characters; the title goes onto the registry row through the debug
 * API's `POST /api/sessions/{id}/attributes`, which carries the project
 * credential the way an application's backend would. Idempotent: a session
 * that has a title keeps it.
 */

const API_URL = (process.env.AGENTDB_DEBUG_URL ?? process.env.NEXT_PUBLIC_AGENTDB_DEBUG_URL ?? "http://localhost:7795").replace(/\/$/, "");
const OPENAI_URL = (process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1").replace(/\/$/, "");
const MODEL = process.env.CONSOLE_TITLE_MODEL ?? "gpt-4o-mini";
const FALLBACK_CHARS = 40;

export const dynamic = "force-dynamic";

const squeeze = (text: string) => text.replace(/\s+/g, " ").trim();

/** The message itself, cut short: what a title is when no model is asked or it does not oblige. */
export function fallbackTitle(message: string) {
  const text = squeeze(message);
  if (!text) return "New conversation";
  return text.length <= FALLBACK_CHARS ? text : `${text.slice(0, FALLBACK_CHARS - 3).trimEnd()}...`;
}

async function askModel(message: string, key: string): Promise<string | null> {
  const response = await fetch(`${OPENAI_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 24,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content:
            "You label conversations for a clinic's inbox. Given the patient's first message, answer with a title of three to six words that says what it is about, as a person would write it: no quotes, no full stop, nothing but the title.",
        },
        { role: "user", content: message },
      ],
    }),
  });
  if (!response.ok) return null;
  const body = (await response.json()) as { choices?: { message?: { content?: string } }[] };
  const title = squeeze(body.choices?.[0]?.message?.content ?? "")
    .replace(/^["']|["']$/g, "")
    .replace(/\.$/, "");
  return title && title.length <= 60 ? title : null;
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { sessionId?: string; firstMessage?: string } | null;
  const sessionId = body?.sessionId?.trim();
  if (!sessionId) return NextResponse.json({ error: "sessionId: none given" }, { status: 400 });
  const firstMessage = body?.firstMessage ?? "";

  const current = await fetch(`${API_URL}/api/sessions/${encodeURIComponent(sessionId)}`, { cache: "no-store" });
  if (!current.ok) return NextResponse.json({ error: `session: ${current.status}` }, { status: current.status });
  const existing = ((await current.json()) as { session?: { attributes?: Record<string, string> } }).session?.attributes?.title;
  if (existing) return NextResponse.json({ title: existing, written: false });

  const key = process.env.OPENAI_API_KEY;
  let title: string | null = null;
  if (key) {
    try {
      title = await askModel(firstMessage, key);
    } catch {
      title = null;
    }
  }
  title ??= fallbackTitle(firstMessage);

  const written = await fetch(`${API_URL}/api/sessions/${encodeURIComponent(sessionId)}/attributes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ attributes: { title } }),
  });
  if (!written.ok) return NextResponse.json({ error: `attributes: ${written.status}` }, { status: 502 });
  return NextResponse.json({ title, written: true });
}

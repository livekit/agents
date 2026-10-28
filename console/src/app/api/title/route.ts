import { NextResponse } from "next/server";

import { backendHeaders, backendURL, localDeveloper } from "@/lib/backend";
import { decode } from "@/lib/decode";
import type { SessionRow } from "@/lib/types";

const FALLBACK_CHARS = 40;
export const dynamic = "force-dynamic";

const squeeze = (text: string) => text.replace(/\s+/g, " ").trim();

/** The message itself, cut short: what a title is when no model is asked or it does not oblige. */
function fallbackTitle(message: string) {
  const text = squeeze(message);
  if (!text) return "New conversation";
  return text.length <= FALLBACK_CHARS ? text : `${text.slice(0, FALLBACK_CHARS - 3).trimEnd()}...`;
}

export async function POST(request: Request) {
  if (!localDeveloper(request)) return NextResponse.json({ error: "Developer access required" }, { status: 403 });
  let body: ReturnType<typeof decode.titleRequest>;
  try { body = decode.titleRequest(await request.json()); }
  catch { return NextResponse.json({ error: "Choose a conversation and message" }, { status: 400 }); }
  const { sessionId, firstMessage } = body;
  const current = await fetch(`${backendURL}/api/sessions/${encodeURIComponent(sessionId)}/info`, { cache: "no-store", headers: backendHeaders() });
  if (!current.ok) return NextResponse.json({ error: "The conversation could not be loaded" }, { status: current.status });
  const existing = (await current.json() as SessionRow).attributes.title;
  if (existing) return NextResponse.json({ title: existing, written: false });
  const title = fallbackTitle(firstMessage);

  const written = await fetch(`${backendURL}/api/sessions/${encodeURIComponent(sessionId)}/attributes`, {
    method: "POST",
    headers: backendHeaders(),
    body: JSON.stringify({ attributes: { title } }),
  });
  if (!written.ok) return NextResponse.json({ error: "The conversation title could not be saved" }, { status: 502 });
  return NextResponse.json({ title, written: true });
}

import { NextResponse } from "next/server";
import { AccessToken, RoomConfiguration, type VideoGrant } from "livekit-server-sdk";

/**
 * The room token for a voice call into the conversation on screen, Kits'
 * route shape: the browser's TokenSource posts what it wants (the agent to
 * dispatch and its metadata), this mints a participant token for a fresh
 * room with the dev key (`LIVEKIT_URL`, `LIVEKIT_API_KEY`,
 * `LIVEKIT_API_SECRET`, the local livekit-server's) and the dispatch in its
 * room configuration, so joining the room dispatches `fast-brain` with the
 * session id in the dispatch metadata, which is where the voice door reads
 * it (`examples/clinic.py`). With no session
 * id yet, one is made through the debug API first, so the call starts the
 * conversation and the page adopts it: the id comes back beside the token.
 *
 * For the demo on a dev stack and nothing else, as the debug API's grant is.
 */

const API_URL = (process.env.AGENTDB_DEBUG_URL ?? process.env.NEXT_PUBLIC_AGENTDB_DEBUG_URL ?? "http://localhost:7795").replace(/\/$/, "");

export const dynamic = "force-dynamic";

type Body = {
  room_name?: string;
  participant_name?: string;
  participant_identity?: string;
  participant_attributes?: Record<string, string>;
  room_config?: { agents?: { agent_name?: string; metadata?: string }[] };
};

export async function POST(request: Request) {
  const url = process.env.LIVEKIT_URL;
  const key = process.env.LIVEKIT_API_KEY;
  const secret = process.env.LIVEKIT_API_SECRET;
  if (!url || !key || !secret)
    return NextResponse.json({ error: "LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET are not set" }, { status: 500 });

  const body = ((await request.json().catch(() => null)) ?? {}) as Body;
  const dispatch = body.room_config?.agents?.[0];
  const agentName = dispatch?.agent_name ?? process.env.CONSOLE_AGENT ?? "fast-brain";
  let metadata: { session_id?: string; sub?: string; endpoint?: string } = {};
  try {
    metadata = dispatch?.metadata ? (JSON.parse(dispatch.metadata) as typeof metadata) : {};
  } catch {
    metadata = {};
  }
  let sessionId = metadata.session_id;
  if (!sessionId) {
    const made = await fetch(`${API_URL}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      /* the session is listed under the agent and the endpoint on screen, as a text one is */
      body: JSON.stringify({ sub: metadata.sub ?? body.participant_identity ?? "", agent: agentName, endpoint: metadata.endpoint ?? "" }),
    });
    if (!made.ok) return NextResponse.json({ error: `session: ${made.status} ${await made.text()}` }, { status: 502 });
    sessionId = ((await made.json()) as { session: { sessionId: string } }).session.sessionId;
  }

  const identity = body.participant_identity || metadata.sub || "demo-user";
  const roomName = body.room_name || `console-call-${sessionId}-${Date.now().toString(36)}`;
  const token = new AccessToken(key, secret, { identity, name: body.participant_name || identity, ttl: "1h", attributes: body.participant_attributes });
  const grant: VideoGrant = { room: roomName, roomJoin: true, canPublish: true, canPublishData: true, canSubscribe: true, canUpdateOwnMetadata: true };
  token.addGrant(grant);
  token.roomConfig = RoomConfiguration.fromJson(
    {
      agents: [
        {
          agentName,
          deployment: process.env.LIVEKIT_AGENT_DEPLOYMENT ?? "",
          metadata: JSON.stringify({ thread_id: sessionId }),
        },
      ],
    },
    { ignoreUnknownFields: true },
  );

  return NextResponse.json(
    {
      server_url: url,
      participant_token: await token.toJwt(),
      room_name: roomName,
      participant_name: body.participant_name || identity,
      participant_identity: identity,
      session_id: sessionId,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}

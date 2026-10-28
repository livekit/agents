import { NextResponse } from "next/server";
import { AccessToken, RoomConfiguration, type VideoGrant } from "livekit-server-sdk";

import { backendHeaders, backendURL, localDeveloper } from "@/lib/backend";
import { decode } from "@/lib/decode";

/** Mint a room participant token and attach the selected conversation to its dispatch. */
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  if (!localDeveloper(request)) return NextResponse.json({ error: "Developer access required" }, { status: 403 });
  const url = process.env.LIVEKIT_URL;
  const key = process.env.LIVEKIT_API_KEY;
  const secret = process.env.LIVEKIT_API_SECRET;
  if (!url || !key || !secret)
    return NextResponse.json({ error: "LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET are not set" }, { status: 500 });

  let body: ReturnType<typeof decode.callRequest>;
  let metadata: ReturnType<typeof decode.callMetadata>;
  try {
    body = decode.callRequest(await request.json());
    const dispatch = body.room_config.agents[0];
    if (!dispatch) throw new Error("Select an agent");
    metadata = decode.callMetadata(JSON.parse(dispatch.metadata));
    if (!metadata.endpoint) throw new Error("Select an endpoint");
  } catch {
    return NextResponse.json({ error: "Select an agent and endpoint for this call" }, { status: 400 });
  }
  const agentName = body.room_config.agents[0].agent_name;
  let sessionId = metadata.session_id;
  if (!sessionId) {
    const made = await fetch(`${backendURL}/api/sessions`, {
      method: "POST",
      headers: backendHeaders(),
      body: JSON.stringify({ sub: metadata.sub ?? body.participant_identity ?? "demo-user", agent: agentName, endpoint: metadata.endpoint }),
    });
    if (!made.ok) return NextResponse.json({ error: `session: ${made.status} ${await made.text()}` }, { status: 502 });
    sessionId = decode.created(await made.json()).session.sessionId;
  }

  const identity = body.participant_identity || metadata.sub || "demo-user";
  const roomName = body.room_name || `console-call-${sessionId}-${Date.now().toString(36)}`;
  const token = new AccessToken(key, secret, { identity, name: body.participant_name || identity, ttl: "1h", attributes: body.participant_attributes ?? undefined });
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

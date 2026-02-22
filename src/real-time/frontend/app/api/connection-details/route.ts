import { SignJWT } from "jose";

const API_KEY = process.env.LIVEKIT_API_KEY || "devkey";
const API_SECRET = process.env.LIVEKIT_API_SECRET || "secret";
const LIVEKIT_URL = process.env.LIVEKIT_URL || "ws://localhost:7880";

const secretKey = new TextEncoder().encode(API_SECRET);

async function createAccessToken(
  apiKey: string,
  apiSecret: string,
  room: string,
  identity: string
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + 24 * 60 * 60; // 24 hours

  return new SignJWT({
    iss: apiKey,
    sub: identity,
    aud: room,
    iat: now,
    exp: exp,
    grants: {
      room: room,
      roomJoin: true,
      canPublish: true,
      canPublishData: true,
      canSubscribe: true,
    },
  })
    .setProtectedHeader({ alg: "HS256", typ: "JWT" })
    .sign(secretKey);
}

export async function POST(request: Request) {
  try {
    const { room, identity } = await request.json();

    if (!room || !identity) {
      return Response.json(
        { error: "Missing room or identity" },
        { status: 400 }
      );
    }

    const token = await createAccessToken(
      API_KEY,
      API_SECRET,
      room,
      identity
    );

    // Convert WebSocket URL to HTTP URL for token endpoint
    const httpUrl = LIVEKIT_URL.replace(/^wss?:\/\//, "http://").replace(
      /^ws:\/\//,
      "http://"
    );

    return Response.json({
      url: LIVEKIT_URL,
      token: token,
    });
  } catch (error) {
    console.error("Failed to generate token:", error);
    return Response.json(
      { error: "Failed to generate token" },
      { status: 500 }
    );
  }
}

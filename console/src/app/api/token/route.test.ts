import { afterEach, expect, it, vi } from "vitest";
import { POST } from "./route";
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });
it("creates the first call's thread on the selected agent and endpoint", async () => {
  vi.stubEnv("LIVEKIT_URL", "ws://localhost:7880");
  vi.stubEnv("LIVEKIT_API_KEY", "devkey");
  vi.stubEnv("LIVEKIT_API_SECRET", "a-secret-at-least-thirty-two-characters");
  vi.stubEnv("CONSOLE_API_TOKEN", "local-console");
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(Response.json({ session: { sessionId: "AT_1", agentName: "desk", sub: "demo-user", attributes: {}, databaseId: "DB_1", createdAt: 1, lastActive: 1, expiresAt: 0, idleTtlSeconds: 86400 } }));
  vi.stubGlobal("fetch", fetcher);
  const request = new Request("http://localhost:3000/api/token", { method: "POST", body: JSON.stringify({ participant_identity: "demo-user", room_config: { agents: [{ agent_name: "desk", metadata: JSON.stringify({ sub: "demo-user", endpoint: "chat" }) }] } }) });
  const response = await POST(request);
  expect(response.status).toBe(200);
  expect(JSON.parse(fetcher.mock.calls[0][1]?.body as string)).toMatchObject({ agent: "desk", endpoint: "chat", sub: "demo-user" });
  expect(fetcher.mock.calls[0][1]?.headers).toMatchObject({ Authorization: "Bearer local-console" });
});

it("refuses malformed call input before creating a conversation or minting a token", async () => {
  vi.stubEnv("LIVEKIT_URL", "ws://localhost:7880");
  vi.stubEnv("LIVEKIT_API_KEY", "devkey");
  vi.stubEnv("LIVEKIT_API_SECRET", "a-secret-at-least-thirty-two-characters");
  const fetcher = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetcher);
  for (const body of [
    null,
    { room_config: { agents: [] } },
    { room_config: { agents: [{ agent_name: "desk", metadata: "not JSON" }] } },
    { room_config: { agents: [{ agent_name: "desk", metadata: JSON.stringify({ endpoint: 7 }) }] } },
    { participant_attributes: { role: 7 }, room_config: { agents: [{ agent_name: "desk", metadata: JSON.stringify({ endpoint: "chat" }) }] } },
  ]) {
    const response = await POST(new Request("http://localhost:3000/api/token", {
      method: "POST", body: JSON.stringify(body),
    }));
    expect(response.status).toBe(400);
  }
  expect(fetcher).not.toHaveBeenCalled();
});

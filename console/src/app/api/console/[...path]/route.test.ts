import { afterEach, expect, it, vi } from "vitest";
import { GET } from "./route";
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });
it("rejects cross-origin developer reads before forwarding backend credentials", async () => {
  const fetcher = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetcher);
  const response = await GET(new Request("http://localhost:3000/api/console/config", { headers: { Origin: "https://other.test" } }), { params: Promise.resolve({ path: ["config"] }) });
  expect(response.status).toBe(403);
  expect(fetcher).not.toHaveBeenCalled();
});

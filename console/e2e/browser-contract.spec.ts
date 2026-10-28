import { expect, test, type Page, type Route } from "@playwright/test";
import type { AgentCard } from "../src/lib/generated/AgentCard";

const endpoint = "http://127.0.0.1:3107/agents/desk/default/chat";
const card: AgentCard = { name: "Coverage assistant", description: "Helps with coverage", version: "1", capabilities: { streaming: true }, supportedInterfaces: [{ url: `${endpoint}/v1`, protocolBinding: "JSONRPC", protocolVersion: "1.0" }], defaultInputModes: ["text"], defaultOutputModes: ["text"], skills: [] };
async function prepare(page: Page) {
  const adminReads: string[] = [];
  await page.route("**/api/console/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.includes("/admin/")) adminReads.push(path);
    if (path.endsWith("/config")) return route.fulfill({ json: { grant: { subs: ["demo-user"] }, streams: { available: true, url: "" } } });
    if (path.endsWith("/examples")) return route.fulfill({ json: { examples: [{ name: card.name, description: card.description, agent: "desk", endpoint: "chat", voice: false, start: "uv run main.py dev", dir: "examples/frontdesk", url: endpoint, live: true }] } });
    if (path.endsWith("/token")) return route.fulfill({ json: { token: "scoped-browser-token", agent: "desk", sub: "demo-user", endpoint, endpointName: "chat", expiresAt: 9999999999 } });
    if (path.endsWith("/history")) return route.fulfill({ json: { items: [] } });
    if (path.includes("/admin/")) return route.fulfill({ json: { fetchedAt: Date.now(), session: { sessionId: "AT_1", agentName: "desk", sub: "demo-user", attributes: {}, databaseId: "DB_1", createdAt: 1, lastActive: 1, expiresAt: 0, idleTtlSeconds: 86400 }, database: { databaseId: "DB_1", tip: 1, tables: [], schemaVersion: 1, lanes: [], items: [], events: [], inbox: [], tasks: [] }, streams: { available: true, url: "" } } });
    return route.fulfill({ json: { sessions: [] } });
  });
  await page.route("**/.well-known/agent-card.json", async (route) => {
    expect(route.request().headers()["authorization"]).toBe("Bearer scoped-browser-token");
    expect(route.request().headers()["a2a-version"]).toBe("1.0");
    return route.fulfill({ json: card });
  });
  await page.route("**/api/title", (route) => route.fulfill({ json: { title: "Coverage request" } }));
  return adminReads;
}
async function reply(route: Route, question = false) {
  const request = route.request().postDataJSON();
  expect(route.request().method()).toBe("POST");
  expect(route.request().headers()["authorization"]).toBe("Bearer scoped-browser-token");
  expect(route.request().headers()["a2a-version"]).toBe("1.0");
  const frame = (result: unknown) => `data:${JSON.stringify({ jsonrpc: "2.0", id: request.id, result })}\r\n\r\n`;
  await route.fulfill({ contentType: "text/event-stream", body: [
    frame({ task: { id: "AST_1", contextId: "AT_1", status: { state: "TASK_STATE_SUBMITTED" } } }),
    ...(question ? [] : [frame({ artifactUpdate: { taskId: "AST_1", contextId: "AT_1", artifact: { artifactId: "answer", parts: [{ text: "Coverage is active 🙂" }] }, append: true } })]),
    frame({ statusUpdate: { taskId: "AST_1", contextId: "AT_1", status: { state: question ? "TASK_STATE_INPUT_REQUIRED" : "TASK_STATE_COMPLETED", ...(question ? { message: { messageId: "question", role: "ROLE_AGENT", parts: [{ text: "Which plan?" }] } } : {}) } } }),
  ].join("") });
}

test("a lost first acknowledgment retries the same message and keeps admin reads explicit", async ({ page }, testInfo) => {
  const adminReads = await prepare(page);
  const messages: string[] = [];
  await page.route("**/agents/desk/default/chat/v1", async (route) => {
    messages.push(route.request().postDataJSON().params.message.messageId);
    if (messages.length === 1) return route.abort("failed");
    return reply(route);
  });
  await page.goto("/");
  await expect(page.getByLabel("Message")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Coverage assistant", exact: true })).toBeVisible();
  await page.getByLabel("Message").fill("Check coverage");
  await page.getByLabel("Message").press("Enter");
  await expect(page.getByText(/The reply did not arrive/)).toBeVisible();
  await page.reload();
  await expect(page.getByText("Coverage is active 🙂", { exact: true })).toBeVisible();
  expect(messages).toHaveLength(2);
  expect(messages[1]).toBe(messages[0]);
  expect(adminReads).toEqual([]);
  await expect(page.getByRole("tab", { name: "Lanes", exact: true })).toHaveCount(0);
  await expect(page.getByText("Reply complete", { exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("chat.png") });
  await page.getByRole("button", { name: "Admin inspector", exact: true }).click();
  await expect(page.getByRole("tab", { name: "Lanes", exact: true })).toBeVisible();
  await expect.poll(() => adminReads.length).toBeGreaterThan(0);
  await page.getByRole("tab", { name: "Lanes", exact: true }).click();
  await page.screenshot({ path: testInfo.outputPath("admin.png") });
});

test("answers a question on the acknowledged task", async ({ page }) => {
  await prepare(page);
  let turns = 0;
  await page.route("**/agents/desk/default/chat/v1", async (route) => {
    if (++turns === 2) expect(route.request().postDataJSON().params.message).toMatchObject({ taskId: "AST_1", contextId: "AT_1" });
    return reply(route, turns === 1);
  });
  await page.goto("/");
  await expect(page.getByLabel("Message")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Coverage assistant", exact: true })).toBeVisible();
  await page.getByLabel("Message").fill("Check coverage");
  await page.getByLabel("Message").press("Enter");
  await expect(page.getByText("Which plan?", { exact: true })).toBeVisible();
  await page.getByLabel("Message").fill("My current plan");
  await page.getByLabel("Message").press("Enter");
  await expect(page.getByText("Coverage is active 🙂", { exact: true })).toBeVisible();
  expect(turns).toBe(2);
});

test("switching conversations aborts a reply without appending its late output", async ({ page }) => {
  await prepare(page);
  let release = () => {};
  const waiting = new Promise<void>((resolve) => { release = resolve; });
  let requested = false;
  await page.route("**/agents/desk/default/chat/v1", async (route) => {
    requested = true;
    await waiting;
    await reply(route).catch(() => undefined);
  });
  await page.goto("/");
  await expect(page.getByLabel("Message")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Coverage assistant", exact: true })).toBeVisible();
  await page.getByLabel("Message").fill("Check coverage");
  await page.getByLabel("Message").press("Enter");
  await expect.poll(() => requested).toBe(true);
  await page.getByRole("button", { name: "New conversation", exact: true }).click();
  release();
  await expect(page.getByLabel("Message")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Coverage assistant", exact: true })).toBeVisible();
  await expect(page.getByText("Coverage is active 🙂", { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem("console.pending"))).toBeNull();
});

test("reload resumes after the opaque cursor and reconciles a complete terminal answer", async ({ page }) => {
  await prepare(page);
  await page.addInitScript(() => {
    localStorage.setItem("console.session", "AT_1");
    localStorage.setItem("console.pending", JSON.stringify({ taskId: "AST_1", sessionId: "AT_1", cursor: "opaque/a:42", answer: "Partial reply", message: { messageId: "saved-message", contextId: "AT_1", role: "ROLE_USER", parts: [{ text: "Check coverage" }] } }));
  });
  await page.route("**/agents/desk/default/chat/v1", async (route) => {
    const request = route.request().postDataJSON();
    expect(request.method).toBe("tasks/resubscribe");
    expect(request.params).toEqual({ id: "AST_1", contextId: "AT_1", offset: "opaque/a:42" });
    return route.fulfill({ json: { jsonrpc: "2.0", id: request.id, result: { task: { id: "AST_1", contextId: "AT_1", status: { state: "TASK_STATE_COMPLETED" }, artifacts: [{ artifactId: "answer", parts: [{ text: "The complete reply" }] }] } } } });
  });
  await page.goto("/");
  await expect(page.getByText("The complete reply", { exact: true })).toBeVisible();
  await expect(page.getByText("Partial reply", { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem("console.pending"))).toBeNull();
});

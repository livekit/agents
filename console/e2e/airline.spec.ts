import { expect, test, type Page } from "@playwright/test";

/**
 * The airline (examples/airline-support) through the console, with the console API's
 * selection on it: `CONSOLE_AGENT=skyline-support` and its `triage` endpoint. Real models,
 * so the checks read states and the amount the graph computed, never the wording.
 */

const TURN = 120_000;
const SHOTS = process.env.E2E_SHOTS ?? "/tmp/lk-console";

async function say(page: Page, text: string) {
  const before = await page.locator("[data-entry='status'][data-state]").count();
  await page.getByLabel("Message").fill(text);
  await page.getByLabel("Message").press("Enter");
  await expect(page.locator("[data-entry='status'][data-state]")).toHaveCount(before + 1, { timeout: TURN });
  return page.locator("[data-entry='status'][data-state]").nth(before);
}

test("a claim without a booking reference: the graph asks, waits, and pays on the reply; the history survives a reload", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("Message")).toBeEnabled({ timeout: 60_000 });
  await page.locator("[data-action='new-conversation']").click();

  const asked = await say(
    page,
    "Hi, Tom Okafor, tom.okafor@example.com. You cancelled my flight SK517 to Lisbon two days before departure. What compensation do I get?",
  );
  await expect(asked).toHaveAttribute("data-state", "input-required");
  await expect(page.locator("[data-entry='question']").last()).toContainText(/booking reference/i);

  const answered = await say(page, "It is SKY5C1");
  await expect(answered).toHaveAttribute("data-state", "completed");
  await expect(page.locator("[data-entry='agent']").last()).toContainText(/400/);
  await page.screenshot({ path: `${SHOTS}/airline-two-turns.png` });

  await page.reload();
  await expect(page.locator("[data-entry='user']")).toHaveCount(2, { timeout: 30_000 });
  await expect(page.locator("[data-entry='user']").last()).toContainText("SKY5C1");
  await expect(page.locator("[data-entry='agent']").first()).toContainText(/booking reference/i);
  await expect(page.locator("[data-entry='agent']").last()).toContainText(/400/);
  await page.screenshot({ path: `${SHOTS}/airline-after-reload.png` });
});

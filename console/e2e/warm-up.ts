import { chromium, type FullConfig } from "@playwright/test";

/**
 * The suite runs against `next dev`, which compiles the route on its first
 * request. After a source edit that first compile can take longer than a
 * test's patience, so compile it once, outside the tests.
 */
export default async function warmUp(config: FullConfig) {
  const baseURL = config.projects[0]?.use?.baseURL ?? process.env.E2E_BASE_URL ?? "http://localhost:3000";
  const browser = await chromium.launch();
  const page = await browser.newPage();
  try {
    await page.goto(`${baseURL}/`, { timeout: 180_000 });
    await page.getByLabel("Message").waitFor({ state: "visible", timeout: 180_000 });
  } finally {
    await browser.close();
  }
}

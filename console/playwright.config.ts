import { defineConfig, devices } from "@playwright/test";

/** The caller's voice for the fake microphone: console/e2e/caller-wav.py writes it. */
const CALLER_WAV = process.env.E2E_CALLER_WAV ?? "/tmp/lk-console/caller.wav";

export default defineConfig({
  testDir: "./e2e",
  /* Compile the route before the first test, not inside it. */
  globalSetup: "./e2e/warm-up.ts",
  /* The suite drives one live stack with real models: one test at a time. */
  fullyParallel: false,
  workers: 1,
  timeout: 300_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  /* WebKit as well as Chromium: the app is used in a Safari-based browser,
     and a whole class of breakage only appears there. Chromium carries the
     fake microphone that plays the caller's WAV once, so the voice run has a
     voice; WebKit has no such device and tests the call's states without
     audio. */
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        permissions: ["microphone"],
        launchOptions: {
          args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            `--use-file-for-fake-audio-capture=${CALLER_WAV}%noloop`,
            "--autoplay-policy=no-user-gesture-required",
          ],
        },
      },
    },
    { name: "webkit", use: { ...devices["Desktop Safari"], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: "pnpm dev",
        url: "http://localhost:3000",
        reuseExistingServer: true,
        timeout: 120_000,
      },
});

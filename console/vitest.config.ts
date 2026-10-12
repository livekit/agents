import path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  /* e2e/ belongs to Playwright; vitest owns the unit tests under src/. */
  test: { include: ["src/**/*.test.ts"] },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});

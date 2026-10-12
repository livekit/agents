import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  /* Next writes an AGENTS.md and a CLAUDE.md into the app otherwise; the
     repository has its own. */
  agentRules: false,
  /* No floating dev badge over the room's status bar. */
  devIndicators: false,
};

export default nextConfig;

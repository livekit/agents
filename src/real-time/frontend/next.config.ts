import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Optimize for real-time media streaming
  experimental: {
    optimizePackageImports: ["@livekit/components-react", "lucide-react"],
  },
};

export default nextConfig;

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Optimize for real-time media streaming
  experimental: {
    optimizePackageImports: ["@livekit/components-react", "lucide-react"],
  },
};

module.exports = nextConfig;

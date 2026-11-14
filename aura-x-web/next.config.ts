import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Prevent turbopack root warnings when nested inside parent directories
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;

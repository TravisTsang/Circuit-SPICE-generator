import { fileURLToPath } from "node:url";
import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: rootDir,
  plugins: [react()],
  cacheDir: path.join(rootDir, "node_modules", ".vite"),
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: {
      strict: true,
      allow: [rootDir],
    },
  },
});

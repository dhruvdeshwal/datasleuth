/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: { global: "globalThis" }, // plotly.js's bundle expects Node's `global`
  server: {
    port: 5173,
    proxy: {
      "/sessions": "http://localhost:8000",
      "/traces": "http://localhost:8000",
    },
  },
  build: { chunkSizeWarningLimit: 1500 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});

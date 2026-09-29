import { defineConfig } from "@playwright/test";
import path from "node:path";

const PORT = 8123;
// PYTHON may be relative (e.g. ../.venv/Scripts/python); resolve so cmd/sh both find it.
const python = path.resolve(process.env.PYTHON ?? "python");
const server = path.resolve(process.cwd(), "../scripts/e2e_server.py");

// E2E is opt-in: `npm run build && npm run e2e` (uses a scripted fake LLM, no API key needed).
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "retain-on-failure" },
  webServer: {
    command: `"${python}" "${server}" ${PORT}`,
    url: `http://127.0.0.1:${PORT}/healthz`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});

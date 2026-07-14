/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Config Vite + Vitest.
// - `defineConfig` importato da "vitest/config" per tipizzare la chiave `test`
//   (con Vitest 4 + build `tsc -b` strict, importarlo da "vite" darebbe errore TS).
// - proxy `/api` verso il backend FastAPI su :8000 (niente CORS in dev).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  test: { environment: "jsdom", setupFiles: "./src/setupTests.ts", globals: true },
});

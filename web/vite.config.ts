import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API and the socket to a local server
// (python -m thirdeye.server.app), so the UI runs against it with no CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});

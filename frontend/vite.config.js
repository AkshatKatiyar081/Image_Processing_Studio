import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development, /api is proxied to the local FastAPI server (uvicorn on :8000).
// In production, set VITE_API_URL to the backend URL (e.g. https://ivp-studio-api.onrender.com).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
  preview: {
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
});

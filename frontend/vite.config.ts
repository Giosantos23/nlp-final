/**
 * Configuración de Vite (y de Vitest, que reutiliza este archivo).
 *
 * - `/api` se redirige a la API FastAPI local (SPEC §5.6) para evitar CORS en desarrollo.
 * - `VITE_MOCK=1` activa el modo demo offline (los fixtures de `src/mocks`).
 * - El visor PDF y los gráficos se separan en chunks propios para que la consola cargue rápido.
 */
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const API_TARGET = process.env.DOCUERP_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
    strictPort: true,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          motion: ["framer-motion"],
          charts: ["recharts"],
          pdf: ["react-pdf"],
        },
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});

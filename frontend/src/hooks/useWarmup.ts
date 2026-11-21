/**
 * Estado del precalentamiento de la API (`GET /api/health` → `warmup`). Mientras los modelos
 * cargan se consulta cada 2 s; al terminar deja de sondear. En demo offline devuelve `null`.
 */
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { WarmupInfo } from "../api/types";
import { useConnection } from "./useConnection";

const POLL_MS = 2000;

export function useWarmup(): WarmupInfo | null {
  const mode = useConnection();
  const [info, setInfo] = useState<WarmupInfo | null>(null);

  useEffect(() => {
    if (mode !== "live") {
      setInfo(null);
      return;
    }
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      const health = await api.health();
      if (!alive) return;
      const next = health?.warmup ?? null;
      setInfo(next);
      if (next && (next.state === "pending" || next.state === "running")) {
        timer = window.setTimeout(() => void tick(), POLL_MS);
      }
    };
    void tick();
    return () => {
      alive = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [mode]);

  return info;
}

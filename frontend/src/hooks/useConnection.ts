/** Suscribe un componente al modo de conexión del cliente (live / offline / mock). */
import { useSyncExternalStore } from "react";
import { connection, type ConnectionMode } from "../api/client";

export function useConnection(): ConnectionMode {
  return useSyncExternalStore(connection.subscribe, () => connection.mode);
}

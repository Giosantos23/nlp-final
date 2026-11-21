/**
 * Usuarios demo fijos (SPEC §3). Deben coincidir con los que siembra la BD de tickets.
 */
import type { DemoUser, UserRole } from "../api/types";

export const MOCK_USERS: DemoUser[] = [
  {
    user_id: "USR-0001",
    name: "Lucía Ordóñez",
    role: "portal_user",
    account: "Comercial Los Volcanes",
  },
  { user_id: "USR-0101", name: "Andrea Castillo", role: "n1", account: null },
  { user_id: "USR-0201", name: "Diego Herrera", role: "n2", account: null },
  { user_id: "USR-0301", name: "Sofía Ramírez", role: "admin", account: null },
];

/** Devuelve el usuario demo por defecto de un rol (el primero que lo tenga). */
export function defaultUserFor(role: UserRole, users: DemoUser[] = MOCK_USERS): DemoUser {
  const found = users.find((u) => u.role === role) ?? MOCK_USERS.find((u) => u.role === role);
  if (!found) throw new Error(`No hay usuario demo para el rol ${role}`);
  return found;
}

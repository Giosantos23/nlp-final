/**
 * Estado de la sesión de la consola: identidad simulada (rol + usuario), versión del ERP,
 * historial de consultas ("turnos") con su traza en vivo y confirmaciones HITL.
 *
 * Se usa un reducer (transiciones explícitas y fáciles de explicar) dentro de un contexto
 * de React. Las vistas leen el estado con `useSession()`.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import { ApiError, api, newSessionId } from "../api/client";
import type { DemoUser, ErpVersion, QueryRequest, QueryResponse, TraceEvent, UserRole } from "../api/types";
import { defaultUserFor, MOCK_USERS } from "../mocks/users";

/** Reenvío del mismo token ya usado: demuestra que la confirmación es de un solo uso. */
export type ReplayState = { status: "sending" } | { status: "rejected"; message: string } | { status: "accepted" };

export type ConfirmState =
  | { status: "idle" }
  | { status: "confirming" }
  | { status: "done"; response: QueryResponse; replay?: ReplayState }
  | { status: "cancelled" }
  | { status: "error"; message: string };

export interface Turn {
  id: string;
  request: QueryRequest;
  status: "streaming" | "done" | "error";
  events: TraceEvent[];
  response: QueryResponse | null;
  error: string | null;
  startedAt: number;
  /** Turno hermano con la misma consulta en la otra versión (comparación 17.0 vs 18.0). */
  compareOf?: string;
  confirm: ConfirmState;
}

interface State {
  users: DemoUser[];
  user: DemoUser;
  version: ErpVersion;
  sessionId: string;
  turns: Turn[];
  /** Turno que muestra el carril de pipeline y abre el inspector. */
  focusedTurnId: string | null;
}

type Action =
  | { type: "users"; users: DemoUser[] }
  | { type: "role"; role: UserRole }
  | { type: "user"; userId: string }
  | { type: "version"; version: ErpVersion }
  | { type: "start"; turn: Turn }
  | { type: "event"; id: string; event: TraceEvent }
  | { type: "done"; id: string; response: QueryResponse }
  | { type: "fail"; id: string; message: string }
  | { type: "focus"; id: string }
  | { type: "confirm"; id: string; confirm: ConfirmState }
  | { type: "clear" };

function patchTurn(state: State, id: string, fn: (t: Turn) => Turn): State {
  return { ...state, turns: state.turns.map((t) => (t.id === id ? fn(t) : t)) };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "users": {
      const users = action.users.length ? action.users : MOCK_USERS;
      const same = users.find((u) => u.user_id === state.user.user_id);
      return { ...state, users, user: same ?? defaultUserFor(state.user.role, users) };
    }
    case "role":
      return { ...state, user: defaultUserFor(action.role, state.users) };
    case "user": {
      const found = state.users.find((u) => u.user_id === action.userId);
      return found ? { ...state, user: found } : state;
    }
    case "version":
      return { ...state, version: action.version };
    case "start":
      return { ...state, turns: [...state.turns, action.turn], focusedTurnId: action.turn.id };
    case "event":
      return patchTurn(state, action.id, (t) => ({ ...t, events: [...t.events, action.event] }));
    case "done":
      return patchTurn(state, action.id, (t) => ({
        ...t,
        status: "done",
        response: action.response,
        // Si la API devolvió la traza completa, es la fuente de verdad frente a lo recibido en vivo.
        events: action.response.events.length ? action.response.events : t.events,
      }));
    case "fail":
      return patchTurn(state, action.id, (t) => ({ ...t, status: "error", error: action.message }));
    case "focus":
      return { ...state, focusedTurnId: action.id };
    case "confirm":
      return patchTurn(state, action.id, (t) => ({ ...t, confirm: action.confirm }));
    case "clear":
      return { ...state, turns: [], focusedTurnId: null };
  }
}

interface SessionApi extends State {
  setRole: (role: UserRole) => void;
  setUser: (userId: string) => void;
  setVersion: (version: ErpVersion) => void;
  /** Envía una consulta con streaming; devuelve el id del turno creado. */
  ask: (query: string, opts?: { version?: ErpVersion; compareOf?: string }) => string;
  cancel: () => void;
  focus: (id: string) => void;
  /** Confirma el borrador HITL; resuelve `true` si el servidor lo aceptó. */
  confirmAction: (turnId: string) => Promise<boolean>;
  /** Reenvía el token ya consumido; el servidor debe rechazarlo (409). */
  replayConfirm: (turnId: string) => Promise<void>;
  cancelAction: (turnId: string) => void;
  clear: () => void;
  focusedTurn: Turn | null;
  busy: boolean;
}

const SessionContext = createContext<SessionApi | null>(null);

let turnSeq = 0;

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => ({
    users: MOCK_USERS,
    user: defaultUserFor("n1"),
    version: "18.0" as ErpVersion,
    sessionId: newSessionId(),
    turns: [],
    focusedTurnId: null,
  }));
  const abortRef = useRef<AbortController | null>(null);
  // Copia "viva" del estado para callbacks estables sin recrearlos en cada render.
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    let alive = true;
    api
      .users()
      .then((users) => alive && dispatch({ type: "users", users }))
      .catch(() => {
        // Sin usuarios de la API se mantienen los usuarios demo fijos (SPEC §3).
      });
    return () => {
      alive = false;
    };
  }, []);

  const ask = useCallback((query: string, opts?: { version?: ErpVersion; compareOf?: string }) => {
    const s = stateRef.current;
    turnSeq += 1;
    const id = `turn-${Date.now().toString(36)}-${turnSeq}`;
    const request: QueryRequest = {
      query: query.trim(),
      role: s.user.role,
      user_id: s.user.user_id,
      erp_version: opts?.version ?? s.version,
      session_id: s.sessionId,
    };
    const turn: Turn = {
      id,
      request,
      status: "streaming",
      events: [],
      response: null,
      error: null,
      startedAt: Date.now(),
      confirm: { status: "idle" },
      ...(opts?.compareOf ? { compareOf: opts.compareOf } : {}),
    };
    dispatch({ type: "start", turn });
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    api
      .streamQuery(request, (event) => dispatch({ type: "event", id, event }), ctrl.signal)
      .then((response) => dispatch({ type: "done", id, response }))
      .catch((error: unknown) => {
        const message =
          error instanceof DOMException && error.name === "AbortError"
            ? "Consulta cancelada."
            : error instanceof ApiError
              ? error.message
              : "Ocurrió un error inesperado al procesar la consulta.";
        dispatch({ type: "fail", id, message });
      });
    return id;
  }, []);

  const confirmAction = useCallback(async (turnId: string) => {
    const s = stateRef.current;
    const turn = s.turns.find((t) => t.id === turnId);
    const pending = turn?.response?.pending_action;
    if (!turn || !pending) return false;
    dispatch({ type: "confirm", id: turnId, confirm: { status: "confirming" } });
    try {
      const response = await api.confirm({
        confirmation_token: pending.confirmation_token,
        role: turn.request.role,
        user_id: turn.request.user_id,
        session_id: s.sessionId,
      });
      dispatch({ type: "confirm", id: turnId, confirm: { status: "done", response } });
      return true;
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "No se pudo confirmar la acción.";
      dispatch({ type: "confirm", id: turnId, confirm: { status: "error", message } });
      return false;
    }
  }, []);

  const replayConfirm = useCallback(async (turnId: string) => {
    const s = stateRef.current;
    const turn = s.turns.find((t) => t.id === turnId);
    const pending = turn?.response?.pending_action;
    if (!turn || !pending || turn.confirm.status !== "done") return;
    const done = turn.confirm;
    const patch = (replay: ReplayState) => dispatch({ type: "confirm", id: turnId, confirm: { ...done, replay } });
    patch({ status: "sending" });
    try {
      const res = await api.confirm({
        confirmation_token: pending.confirmation_token,
        role: turn.request.role,
        user_id: turn.request.user_id,
        session_id: s.sessionId,
      });
      // La API real responde 409; la demo offline devuelve la traza con la herramienta bloqueada.
      const refused = res.blocked || res.events.some((e) => e.stage === "tool.create_ticket" && e.status !== "ok");
      patch(refused ? { status: "rejected", message: res.response.answer } : { status: "accepted" });
    } catch (error) {
      patch({ status: "rejected", message: error instanceof ApiError ? error.message : "El servidor rechazó el token." });
    }
  }, []);

  const value = useMemo<SessionApi>(() => {
    const focusedTurn =
      state.turns.find((t) => t.id === state.focusedTurnId) ?? state.turns[state.turns.length - 1] ?? null;
    return {
      ...state,
      focusedTurn,
      busy: state.turns.some((t) => t.status === "streaming"),
      setRole: (role) => dispatch({ type: "role", role }),
      setUser: (userId) => dispatch({ type: "user", userId }),
      setVersion: (version) => dispatch({ type: "version", version }),
      ask,
      cancel: () => abortRef.current?.abort(),
      focus: (id) => dispatch({ type: "focus", id }),
      confirmAction,
      replayConfirm,
      cancelAction: (id) => dispatch({ type: "confirm", id, confirm: { status: "cancelled" } }),
      clear: () => dispatch({ type: "clear" }),
    };
  }, [state, ask, confirmAction, replayConfirm]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components -- el hook acompaña a su proveedor.
export function useSession(): SessionApi {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession debe usarse dentro de <SessionProvider>");
  return ctx;
}

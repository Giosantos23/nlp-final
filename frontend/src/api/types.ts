/**
 * Tipos espejo de los contratos Pydantic de `src/docuerp/schemas.py` (SPEC §4).
 *
 * Regla: los nombres y la forma de los campos son idénticos a los de Python, para que la
 * respuesta JSON de la API se pueda usar sin transformaciones. Los tipos marcados como
 * "auxiliares" (manifiesto, evaluación, seguridad) describen lo que el frontend espera de
 * los endpoints de lectura de §5.6; el cliente los normaliza de forma tolerante.
 */

/* ------------------------------------------------------------------------------------ */
/* Vocabularios cerrados                                                                 */
/* ------------------------------------------------------------------------------------ */

export type UserRole = "portal_user" | "n1" | "n2" | "admin";
export type ErpVersion = "17.0" | "18.0";
export type ChunkErpVersion = ErpVersion | "any";
export type DocStatus = "vigente" | "obsoleto";
export type DocType = "odoo_doc" | "runbook" | "policy";
export type Intent =
  | "doc_query"
  | "ticket_status"
  | "ticket_create"
  | "ticket_classify"
  | "out_of_domain";
export type RiskLevel = "low" | "medium" | "high" | "critical";
export type TicketCategory =
  | "configuracion"
  | "error_funcional"
  | "acceso_permisos"
  | "rendimiento"
  | "migracion"
  | "capacitacion"
  | "integracion";
export type TicketPriority = "baja" | "media" | "alta" | "critica";
export type ErpModule =
  | "inventario"
  | "compras"
  | "ventas"
  | "facturacion"
  | "accesos"
  | "actualizacion"
  | "general";
export type RetrieverName = "bm25" | "dense" | "hybrid" | "rerank" | "mmr";
export type TraceStatus = "ok" | "blocked" | "error" | "skipped" | "retry";

/** Constantes compartidas con el backend (mismos valores que `schemas.py`). */
export const ABSTENTION_MESSAGE =
  "No encontré evidencia suficiente en la documentación autorizada para responder con " +
  "seguridad. Reformula la consulta o escálala a un analista N2.";
export const USER_ID_PATTERN = /^USR-\d{4}$/;
export const TICKET_ID_PATTERN = /TCK-\d{5}/;
export const CHUNK_ID_PATTERN = /^.+::p\d{4}::c\d{3}$/;
export const QUERY_MIN_CHARS = 3;
export const QUERY_MAX_CHARS = 2000;

/* ------------------------------------------------------------------------------------ */
/* Contratos principales (SPEC §4)                                                       */
/* ------------------------------------------------------------------------------------ */

export interface QueryRequest {
  query: string;
  role: UserRole;
  user_id: string;
  erp_version: ErpVersion;
  session_id?: string | null;
  confirmation_token?: string | null;
}

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  source: string;
  page: number;
  page_end: number;
  version: string;
  erp_version: ChunkErpVersion;
  status: DocStatus;
  section_path: string;
  module: ErpModule;
  doc_type: DocType;
  language: "en" | "es";
  allowed_roles: UserRole[];
  content_hash: string;
  ingested_at: string;
  token_count: number;
  flags: string[];
}

export interface ScoredChunk {
  chunk: Chunk;
  score: number;
  rank: number;
  retriever: RetrieverName;
  stage_scores: Record<string, number>;
}

export interface SourceRef {
  source: string;
  page: number;
  chunk_id: string;
}

/** CONTRATO DE SALIDA §11: exactamente estos cinco campos. */
export interface Answer {
  answer: string;
  sources: SourceRef[];
  abstained: boolean;
  tool_used: string | null;
  needs_human_review: boolean;
}

export interface TicketDraft {
  summary: string;
  description: string;
  category: TicketCategory;
  priority: TicketPriority;
  module: ErpModule;
  account_id?: string | null;
}

export interface PendingAction {
  action: "create_ticket";
  draft: TicketDraft;
  confirmation_token: string;
  expires_at: string;
}

export interface Usage {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
}

export interface TraceEvent {
  stage: string;
  status: TraceStatus;
  duration_ms: number;
  data: Record<string, unknown>;
}

export interface QueryResponse {
  trace_id: string;
  intent: Intent;
  response: Answer;
  pending_action: PendingAction | null;
  blocked: boolean;
  block_reason: string | null;
  latency_ms: number;
  usage: Usage;
  events: TraceEvent[];
}

/** Cuerpo de `POST /api/actions/confirm` (`docuerp.api.models.ConfirmRequest`). */
export interface ConfirmRequest {
  confirmation_token: string;
  role: UserRole;
  user_id: string;
  session_id?: string | null;
}

export interface WarmupStep {
  name: string;
  label: string;
  status: "pending" | "running" | "ok" | "error" | "skipped";
  duration_ms: number | null;
  error: string | null;
}

/** Precalentamiento de modelos e índices al arrancar la API (`docuerp.api.warmup`). */
export interface WarmupInfo {
  state: "disabled" | "pending" | "running" | "ready" | "partial";
  started_at: string | null;
  finished_at: string | null;
  total_ms: number | null;
  steps: WarmupStep[];
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_provider: string;
  index_ready: boolean;
  tickets_db_ready: boolean;
  mock?: boolean;
  models?: Record<string, string>;
  warmup?: WarmupInfo | null;
}

/* ------------------------------------------------------------------------------------ */
/* Tipos auxiliares de los endpoints de lectura                                          */
/* ------------------------------------------------------------------------------------ */

export interface DemoUser {
  user_id: string;
  name: string;
  role: UserRole;
  account?: string | null;
}

/** Entrada del manifiesto de fuentes (`data/manifest/*.yaml`, SPEC §5.1). */
export interface SourceEntry {
  id: string;
  title: string;
  source_file: string;
  origin: string;
  license: string;
  owner: string;
  format: string;
  version: string;
  erp_version: ChunkErpVersion;
  status: DocStatus;
  date: string;
  update_frequency: string;
  allowed_roles: UserRole[];
  sensitivity: string;
  module: ErpModule;
  doc_type: DocType;
  language: "en" | "es";
  sha256: string;
  pages?: number;
  chunks?: number;
  tokens?: number;
}

/** Métrica con meta del anteproyecto y, si existe, intervalo de confianza bootstrap 95 %. */
export interface MetricValue {
  key: string;
  label: string;
  value: number;
  baseline?: number | null;
  ci?: [number, number] | null;
  baseline_ci?: [number, number] | null;
  /** Meta del anteproyecto; `null` = métrica de referencia sin meta. */
  target: number | null;
  /** `gte`: cumple si value ≥ target; `lte`: cumple si value ≤ target. */
  direction: "gte" | "lte";
  unit: "ratio" | "ms" | "usd" | "tokens";
  note?: string;
}

export interface AblationRow {
  group: "chunker" | "embedder" | "retriever";
  config: string;
  recall_at_5: number;
  mrr: number;
  ndcg_at_5: number;
  latency_ms: number;
  selected?: boolean;
}

export interface CategoryRow {
  category: string;
  n: number;
  recall_at_5: number;
  faithfulness: number;
  citation_precision: number;
  passed: number;
}

export interface FailureRow {
  case_id: string;
  category: string;
  question: string;
  expected: string;
  observed: string;
  cause: string;
  severity: RiskLevel;
}

export interface EvalSummary {
  run_id: string;
  baseline_run_id: string;
  created_at: string;
  n_cases: number;
  runs: number;
  metrics: MetricValue[];
  ablations: AblationRow[];
  categories: CategoryRow[];
  failures: FailureRow[];
  paired_test?: { metric: string; p_value: number; delta: number }[];
}

export type RedteamVector =
  | "injection_direct"
  | "injection_indirect"
  | "permission_evasion"
  | "critical_action"
  | "forced_hallucination"
  | "pii_exfiltration"
  | "tool_abuse";

export type RedteamOutcome = "blocked" | "denied" | "abstained" | "needs_confirmation" | "masked";

export interface RedteamCase {
  id: string;
  vector: RedteamVector;
  attack: string;
  role: UserRole;
  threat: string;
  component: string;
  expected_behavior: string;
  expected_outcome: RedteamOutcome;
  observed: string;
  passed: boolean;
  mitigation: string;
  in_official_bench: boolean;
}

export interface SecuritySummary {
  run_id: string;
  created_at: string;
  cases: RedteamCase[];
}

export interface CorpusStats {
  documents: number;
  pages: number;
  chunks: number;
  tokens: number;
  flagged_chunks: number;
  duplicates: number;
  chunker: string;
  embedder: string;
  by_module: { module: ErpModule; chunks: number }[];
  token_histogram: { bucket: string; chunks: number }[];
}

export interface DocumentsResponse {
  sources: SourceEntry[];
  stats: CorpusStats;
}

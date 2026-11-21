# DocuERP · Frontend

Consola web de DocuERP (SPEC §5.8): consultas con citas verificables, traza del agente en
vivo, confirmación humana para acciones críticas, evaluación, seguridad y gobernanza del
corpus. Estética *dark tech* con los tokens de `design/tokens.json`.

**Stack:** Vite 6 · React 18 · TypeScript estricto · Tailwind 3 · Framer Motion · Recharts ·
lucide-react · react-pdf (pdf.js) · Vitest.

## Uso

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, /api → http://127.0.0.1:8000 (make serve)
npm run dev:mock     # demo offline: escenarios precargados, sin API
```

| Script | Qué hace |
|---|---|
| `npm run build` / `build:mock` | Typecheck (`tsc -b`) + build de producción (normal / demo) |
| `npm run lint` | ESLint (reglas con tipos) + `tsc --noEmit` |
| `npm test` | Pruebas unitarias (parser SSE, traza, citas, escenarios, métricas, normalizadores) |
| `npm run mock:pdfs` | Regenera los PDF de demo de `public/mock-docs/` desde `src/mocks/documents.json` |
| `npm run screenshots` | Capturas 1440×900 de cada vista (requiere `build:mock` + `preview`) |

La URL de la API se puede cambiar con `DOCUERP_API_URL=http://host:puerto npm run dev`.

## Vistas

- **Consola** (`#/consola`): barra de consulta (Enter envía, `/` enfoca), rol y usuario demo,
  versión 17.0/18.0, escenarios de ejemplo. La respuesta muestra citas `[fuente · pág.]`; al
  abrir una, el panel lateral trae el chunk y el PDF en esa página con el fragmento resaltado.
  Tiene estados para abstención, bloqueo, acceso restringido, revisión humana y herramientas
  (estado de ticket con SLA, clasificación con confianza), además del modal HITL con cuenta
  regresiva del token. El carril derecho marca cada etapa a medida que llega por SSE.
- **Inspector de traza** (`#/traza/<trace_id>`): waterfall de latencias, listas BM25 / denso /
  RRF / rerank con cambio de posición, gate τ, filtros RBAC, tokens/costo y JSON contractual.
- **Evaluación**, **Seguridad** y **Corpus y gobernanza**: KPIs contra las metas del
  anteproyecto con IC 95 %, ablaciones, matriz red-team y manifiesto de fuentes.

Atajos: `⌘K` paleta de comandos · `G` + `C/T/E/S/D` para navegar · `Esc` cierra paneles.

## Conexión con la API

`src/api/client.ts` es el cliente tipado (los tipos de `src/api/types.ts` son espejo de
`schemas.py`). `POST /api/query/stream` se lee con `fetch` + `SseParser`
(`event: trace` → `TraceEvent`, `event: final` → `QueryResponse`). Si `GET /api/health` no
responde, la UI cambia sola a **demo offline** y lo indica en la barra lateral.

Formas que la UI espera de los endpoints de lectura (los normalizadores aceptan variantes):
`GET /api/eval/summary` → `EvalSummary` (`metrics[]` con `value`, `target`, `direction`, `ci`,
`baseline`…), `GET /api/security/summary` → `{ run_id, cases: RedteamCase[] }`,
`GET /api/documents` → `SourceEntry[]` o `{ sources, stats }`. En `TraceEvent.data` de las
etapas de recuperación se leen `hits` (filas `{chunk_id, source, page, score, rank}` o
`ScoredChunk`), `filters` y `filtered_out`; en `evidence_gate`, `tau` y `top_score`.

## Modo demo

`src/mocks/` replica el contrato de la API con escenarios realistas: consulta nominal con 3
citas, abstención por τ, denegación por rol, estado de ticket, clasificación, creación con
borrador y confirmación (token de un solo uso), inyección bloqueada y 17.0 vs 18.0. Las
cifras de Evaluación, Seguridad y Corpus en este modo son **de demostración** y la interfaz
lo rotula así.

# DocuERP — tareas del proyecto. Ejecute `make help` para ver los objetivos.
SHELL := /bin/bash
.DEFAULT_GOAL := help

UV        ?= uv
RUN       := $(UV) run
DOCUERP   := $(RUN) docuerp
CHUNKER   ?= structure
EMBEDDER  ?= bge-m3
RUNS      ?= 3
CONFIGS   ?= bm25,dense,hybrid,hybrid_rerank,hybrid_rerank_mmr
SPLIT     ?= test
RUN_DIR   ?=
JUDGE     ?= claude-opus-5-5
# Corridas del reporte oficial: reports/runs.json fija cuál es cada una.
OFFICIAL  := eval/results/20261117-1112_test eval/results/20261117-1118_baseline eval/results/20261117-1136_final
FRONTEND  := frontend

.PHONY: help setup lint format typecheck test test-all corpus odoo-pdfs ingest ingest-chunks \
        ingest-ablation train warmup eval-validate eval-retrieval eval-e2e eval-judge eval-compare \
        eval-report redteam redteam-e2e serve serve-mock frontend frontend-mock demo report \
        report-preview screenshots-live agreement calibrate results clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Instala dependencias (uv), Chromium de Playwright y el frontend
	$(UV) sync
	$(RUN) playwright install chromium
	@if [ -f $(FRONTEND)/package.json ]; then cd $(FRONTEND) && npm install; fi

lint: ## Ruff (lint + formato en modo verificación)
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format: ## Aplica formato y correcciones automáticas de ruff
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

typecheck: ## mypy sobre el paquete
	$(RUN) mypy

test: ## Pruebas rápidas (excluye slow y llm)
	$(RUN) pytest

test-all: ## Todas las pruebas, incluidas slow y llm
	$(RUN) pytest -m "" --cov=docuerp --cov-report=term-missing

corpus: ## Corpus sintético: runbooks (PDF/DOCX), BD de tickets y datasets
	$(DOCUERP) data build-runbooks
	$(DOCUERP) data seed-tickets
	$(DOCUERP) data datasets

odoo-pdfs: ## Documentación RST de Odoo 17.0/18.0 → PDF por módulo
	$(RUN) python scripts/build_odoo_pdfs.py

ingest: ## Ingesta principal (CHUNKER=structure EMBEDDER=bge-m3) + reporte
	$(DOCUERP) ingest run --chunker $(CHUNKER) --embedder $(EMBEDDER) --sources all
	$(DOCUERP) ingest report

ingest-chunks: ## Chunks de los 3 chunkers sin embeddings + reporte
	for c in structure fixed256 fixed768; do $(DOCUERP) ingest run --chunker $$c --sources all --skip-embeddings; done
	$(DOCUERP) ingest report

ingest-ablation: ## Índices para ablación: fixed256, fixed768 y e5-base
	$(DOCUERP) ingest run --chunker fixed256 --embedder bge-m3 --sources all
	$(DOCUERP) ingest run --chunker fixed768 --embedder bge-m3 --sources all
	$(DOCUERP) ingest run --chunker structure --embedder e5-base --sources all
	$(DOCUERP) ingest report

train: ## Entrena el router de intención y el clasificador de tickets (TF-IDF y embeddings)
	$(DOCUERP) agent train-router
	$(DOCUERP) agent tools train-classifier --kind both

warmup: ## Precarga índices, embedder y reranker (antes de una demo)
	$(DOCUERP) retrieval warmup

eval-validate: ## Valida composición y evidencias del banco (60 casos: 42 de test y 18 de dev)
	$(DOCUERP) eval validate

eval-retrieval: ## Evaluación de recuperación por configuración (CONFIGS=... SPLIT=test)
	$(DOCUERP) eval retrieval --configs $(CONFIGS) --split $(SPLIT)

eval-e2e: ## Evaluación extremo a extremo baseline y final (RUNS=3 SPLIT=test)
	$(DOCUERP) eval e2e --config baseline --runs $(RUNS) --split $(SPLIT)
	$(DOCUERP) eval e2e --config final --runs $(RUNS) --split $(SPLIT)

eval-judge: ## LLM-as-judge sobre una corrida (RUN_DIR=eval/results/<id>, JUDGE ≠ generador)
	@test -n "$(RUN_DIR)" || (echo "Indica RUN_DIR=eval/results/<run_id>"; exit 2)
	$(DOCUERP) eval judge $(RUN_DIR) --model $(JUDGE)

eval-compare: ## Baseline vs final oficial (OFFICIAL) con IC 95 % y prueba pareada
	$(DOCUERP) eval compare eval/results/20261117-1118_baseline eval/results/20261117-1136_final

eval-report: ## Reporte oficial (corridas OFFICIAL), IC bootstrap y figuras
	$(DOCUERP) eval report $(OFFICIAL)

results: ## Hoja única de resultados (reports/RESULTS.md) y métricas del video
	$(RUN) python scripts/collect_results.py

redteam: ## Banco red-team de seguridad (solo guards, sin LLM)
	$(DOCUERP) security redteam --guards-only

redteam-e2e: ## Banco red-team de extremo a extremo (agente completo)
	$(DOCUERP) security redteam

serve: ## API FastAPI en :8000
	$(DOCUERP) serve --host 127.0.0.1 --port 8000

serve-mock: ## API en modo demo (respuestas simuladas, sin modelos ni LLM)
	$(DOCUERP) serve --host 127.0.0.1 --port 8000 --mock

frontend: ## Frontend Vite en :5173
	cd $(FRONTEND) && npm run dev

frontend-mock: ## Frontend en modo demo offline (sin API)
	cd $(FRONTEND) && npm run dev:mock

demo: ## API + frontend en paralelo (Ctrl+C detiene ambos)
	@trap 'kill 0' INT TERM EXIT; \
	$(DOCUERP) serve --host 127.0.0.1 --port 8000 & \
	(cd $(FRONTEND) && npm run dev) & \
	wait

report: ## Construye el informe final (docs/report/build/Informe_DocuERP.{docx,pdf})
	$(RUN) python scripts/build_report.py

report-preview: ## Mide la extensión del informe con valores ficticios
	$(RUN) python scripts/build_report.py --preview --pdf-only

agreement: ## Acuerdo entre anotadores del banco (κ de Cohen/Fleiss, Jaccard de evidencia)
	$(RUN) python scripts/annotation_agreement.py

calibrate: ## Calibra el umbral semántico del guard de entrada (bge-m3)
	$(DOCUERP) security calibrate

screenshots-live: ## Capturas del frontend contra la API real (requiere docuerp serve --port 8011)
	cd $(FRONTEND) && npm run build
	@trap 'kill 0' INT TERM EXIT; \
	(cd $(FRONTEND) && DOCUERP_API_URL=http://127.0.0.1:8011 npx vite preview --port 4173 --strictPort) & \
	sleep 3; cd $(FRONTEND) && $(RUN) python scripts/screenshots_live.py http://localhost:4173

clean: ## Borra artefactos generados reproducibles (no toca .env ni fuentes)
	rm -rf data/build data/processed data/index logs .cache
	rm -f data/tickets/*.db
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov

"""Configuración centralizada de DocuERP.

Fuentes, de mayor a menor precedencia:

1. Argumentos explícitos al construir :class:`Settings` (p. ej. en pruebas).
2. Alias planos de entorno: ``DOCUERP_LLM_PROVIDER``, ``DOCUERP_LLM_MODEL``,
   ``DOCUERP_JUDGE_MODEL`` y ``DOCUERP_HMAC_SECRET``.
3. Variables de entorno anidadas con prefijo ``DOCUERP_`` y separador ``__``
   (p. ej. ``DOCUERP_RETRIEVAL__TOP_K=8``).
4. ``configs/settings.yaml`` (o la ruta en ``DOCUERP_SETTINGS_FILE``).
5. Valores por defecto de los modelos.

El archivo ``.env`` se carga al entorno con ``python-dotenv`` **sin sobrescribir** variables
existentes y **sin imprimir** su contenido. Los secretos se guardan como ``SecretStr``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from docuerp import paths

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mmr"]
ChunkerName = Literal["structure", "fixed256", "fixed768"]
ProviderName = Literal["agent_sdk", "anthropic_api", "offline", "fake"]


class _Section(BaseModel):
    """Base de las secciones: prohíbe claves desconocidas para detectar errores de tipeo."""

    model_config = ConfigDict(extra="forbid")


class RetrievalWeights(_Section):
    """Pesos de la fusión RRF por recuperador."""

    bm25: float = 0.5
    dense: float = 0.5


class RetrievalSettings(_Section):
    """Perillas de recuperación (SPEC §5.2)."""

    mode: RetrievalMode = "hybrid_rerank"
    chunker: ChunkerName = "structure"
    embedder: str = "bge-m3"
    candidates_k: int = Field(default=20, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)
    tau: float = Field(default=0.30, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    include_obsolete: bool = False
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    batch_size: int = Field(default=16, ge=1)


class ChunkingSettings(_Section):
    """Parámetros de fragmentación (SPEC §5.1)."""

    target_tokens: int = Field(default=450, ge=32)
    overlap: float = Field(default=0.15, ge=0.0, lt=1.0)
    min_tokens: int = Field(default=40, ge=0)
    fixed_sizes: list[int] = Field(default_factory=lambda: [256, 768])


class ModelsSettings(_Section):
    """Identificadores de modelos de Hugging Face."""

    embedders: dict[str, str] = Field(
        default_factory=lambda: {
            "bge-m3": "BAAI/bge-m3",
            "e5-base": "intfloat/multilingual-e5-base",
        }
    )
    reranker: str = "BAAI/bge-reranker-v2-m3"
    tokenizer: str = "BAAI/bge-m3"
    max_seq_length: int = 1024
    use_fp16: bool = False


class ModelPrice(_Section):
    """Precio de lista en USD por millón de tokens."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cache_read: float | None = Field(default=None, ge=0.0)
    """Precio propio de lectura de caché; si falta, ``input × cache_read_multiplier``."""


class LLMSettings(_Section):
    """Proveedor, modelos y precios del LLM (SPEC §5.3)."""

    provider: ProviderName = "agent_sdk"
    model: str = "claude-haiku-5-5"
    judge_model: str = "claude-opus-5-5"
    max_tokens: int = Field(default=2048, ge=1)
    timeout_s: float = Field(default=45.0, gt=0.0)
    effort: str | None = "low"
    temperature: float = 0.0
    retries: int = Field(default=1, ge=0)
    fallback_provider: ProviderName | None = "offline"
    max_concurrency: int = Field(default=6, ge=1)
    use_client_pool: bool = False
    pricing: dict[str, ModelPrice] = Field(
        default_factory=lambda: {
            "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
            "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_read=0.10),
            "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.20),
        }
    )
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25

    def price_for(self, model: str) -> ModelPrice | None:
        """Devuelve el precio del modelo (o ``None`` si no está tarifado)."""
        return self.pricing.get(model)


class RouterSettings(_Section):
    """Umbrales del enrutador de intención."""

    ood_threshold: float = 0.35
    min_confidence: float = 0.55


class ClassifierSettings(_Section):
    """Umbral del clasificador de tickets."""

    min_confidence: float = 0.60


class AgentSettings(_Section):
    """Límites del agente y de las herramientas (SPEC §5.4)."""

    max_steps: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=3, ge=0)
    create_ticket_per_session: int = Field(default=3, ge=0)
    confirmation_ttl_min: int = Field(default=10, ge=1)
    tool_timeout_s: float = Field(default=10.0, gt=0.0)
    router: RouterSettings = Field(default_factory=RouterSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)


class SecuritySettings(_Section):
    """Perillas de seguridad (SPEC §5.5). ``hmac_secret`` proviene solo del entorno."""

    max_query_chars: int = Field(default=2000, ge=3)
    injection_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    semantic_attack_threshold: float = Field(default=0.74, ge=0.0, le=1.0)
    semantic_enabled: bool = True
    semantic_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    embedder: str | None = None  # None = usa retrieval.embedder; 'fake' en CI
    mask_pii_in_logs: bool = True
    block_on_canary: bool = True
    hmac_secret: SecretStr | None = None


class ApiSettings(_Section):
    """Servidor FastAPI (SPEC §5.6)."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )


class PathsSettings(_Section):
    """Rutas (relativas a la raíz del repo o absolutas). Use :meth:`resolve`."""

    data: str = "data"
    raw: str = "data/raw"
    build: str = "data/build"
    processed: str = "data/processed"
    index: str = "data/index"
    manifest: str = "data/manifest"
    tickets_db: str = "data/tickets/tickets.db"
    logs: str = "logs"
    audit_log: str = "logs/audit.jsonl"
    reports: str = "reports"
    eval: str = "eval"
    eval_results: str = "eval/results"

    def resolve(self, name: str) -> Path:
        """Devuelve la ruta absoluta de la clave ``name`` (p. ej. ``"tickets_db"``)."""
        return paths.resolve(getattr(self, name))


# ---------------------------------------------------------------------------
# Fuentes personalizadas
# ---------------------------------------------------------------------------

_FLAT_ALIASES: dict[str, tuple[str, str]] = {
    "DOCUERP_LLM_PROVIDER": ("llm", "provider"),
    "DOCUERP_LLM_MODEL": ("llm", "model"),
    "DOCUERP_JUDGE_MODEL": ("llm", "judge_model"),
    "DOCUERP_HMAC_SECRET": ("security", "hmac_secret"),
}


def _settings_file() -> Path:
    """Ruta del YAML de configuración (``DOCUERP_SETTINGS_FILE`` o ``configs/settings.yaml``)."""
    override = os.environ.get("DOCUERP_SETTINGS_FILE")
    return paths.resolve(override) if override else paths.SETTINGS_FILE


class _YamlSource(PydanticBaseSettingsSource):
    """Lee ``configs/settings.yaml``; si no existe, no aporta valores."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path = _settings_file()
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} debe contener un mapeo YAML en la raíz")
        return data


class _FlatAliasSource(PydanticBaseSettingsSource):
    """Traduce los alias planos ``DOCUERP_LLM_MODEL`` & cía. a la estructura anidada."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        out: dict[str, dict[str, Any]] = {}
        for env_name, (section, key) in _FLAT_ALIASES.items():
            value = os.environ.get(env_name)
            if value:  # se ignoran valores vacíos (p. ej. copiados de .env.example)
                out.setdefault(section, {})[key] = value
        return out


class Settings(BaseSettings):
    """Configuración completa de DocuERP. Obténgala con :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="DOCUERP_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    project: str = "DocuERP"
    seed: int = 42
    log_level: str = "INFO"
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Orden: init > alias planos > entorno anidado > YAML > defaults.

        ``dotenv_settings`` se omite a propósito: ``.env`` ya se volcó al entorno con
        python-dotenv en :func:`load_env`, así un único camino controla los secretos.
        """
        del dotenv_settings, file_secret_settings
        return (
            init_settings,
            _FlatAliasSource(settings_cls),
            env_settings,
            _YamlSource(settings_cls),
        )

    def embedder_model_id(self, key: str | None = None) -> str:
        """Devuelve el id de HF del embedder ``key`` (o del configurado en ``retrieval``)."""
        name = key or self.retrieval.embedder
        try:
            return self.models.embedders[name]
        except KeyError as exc:
            raise KeyError(f"Embedder desconocido: {name!r}") from exc

    def safe_dump(self) -> dict[str, Any]:
        """Volcado apto para logs/reportes: los ``SecretStr`` aparecen enmascarados."""
        return self.model_dump(mode="json")


_ENV_LOADED = False


def load_env(dotenv_path: Path | None = None) -> bool:
    """Carga ``.env`` al entorno del proceso una sola vez, sin sobrescribir ni imprimir.

    Returns:
        ``True`` si se encontró y cargó un archivo ``.env``.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return False
    _ENV_LOADED = True
    path = dotenv_path or paths.DOTENV
    if not path.is_file():
        return False
    return load_dotenv(path, override=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuración global (cacheada). Use ``get_settings.cache_clear()``
    en pruebas tras modificar el entorno."""
    load_env()
    return Settings()


__all__ = [
    "AgentSettings",
    "ApiSettings",
    "ChunkerName",
    "ChunkingSettings",
    "LLMSettings",
    "ModelPrice",
    "ModelsSettings",
    "PathsSettings",
    "ProviderName",
    "RetrievalMode",
    "RetrievalSettings",
    "SecuritySettings",
    "Settings",
    "get_settings",
    "load_env",
]

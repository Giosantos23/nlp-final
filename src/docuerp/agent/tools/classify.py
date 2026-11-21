"""Herramienta ``classify_ticket(text ≤ 2000)``: TF-IDF+LR (baseline) vs embeddings+LR.

Dos representaciones del texto del ticket, el mismo clasificador lineal encima:

* ``tfidf`` (baseline): bolsa de n-gramas de palabras (1–2) ponderada por TF-IDF. Captura
  coincidencias léxicas ("no puedo validar la recepción") pero no sinónimos.
* ``embeddings``: vector denso del modelo de embeddings compartido (bge-m3 por defecto),
  que acerca paráfrasis aunque no compartan palabras.

Encima de cada representación se entrena una **regresión logística** por objetivo
(``category``, ``priority``, ``module``) con ``class_weight="balanced"`` para que las
clases poco frecuentes no queden ignoradas. La probabilidad de la clase ganadora es la
confianza; si la de la categoría queda por debajo de ``agent.classifier.min_confidence``
(0.60), el resultado se marca ``needs_human_review``.

Los modelos se entrenan con ``data/classifier/tickets_train.jsonl``, se evalúan con
``tickets_test.jsonl`` (accuracy y F1 macro) y se guardan con joblib en
``data/index/classifier/{kind}.joblib``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp.agent.tools.base import ToolContext, ToolResult, ToolSpec, tool_roles
from docuerp.schemas import ErpModule, TicketCategory, TicketPriority
from docuerp.text import strip_accents

if TYPE_CHECKING:
    from docuerp.config import Settings

log = logging.getLogger(__name__)

ClassifierKind = Literal["tfidf", "embeddings"]
CLASSIFIER_KINDS: tuple[ClassifierKind, ...] = ("tfidf", "embeddings")
TARGETS: tuple[str, ...] = ("category", "priority", "module")

C_GRID: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0)
"""Rejilla de regularización de la regresión logística (se elige por CV en train)."""
"""Objetivos que predice el clasificador (uno por regresión logística)."""


class ClassifyTicketInput(BaseModel):
    """Argumentos de ``classify_ticket``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=3, max_length=2000)


class TicketExample(BaseModel):
    """Fila de ``data/classifier/tickets_{train,test}.jsonl`` (SPEC §10)."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    category: TicketCategory
    module: ErpModule
    priority: TicketPriority
    ticket_id: str = ""


class ClassifierNotReadyError(RuntimeError):
    """No hay modelo guardado ni datos para entrenarlo."""


def load_examples(path: Path) -> list[TicketExample]:
    """Lee y valida un JSONL de tickets etiquetados.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si una línea no es JSON válido o no cumple el formato (indica la línea).
    """
    examples: list[TicketExample] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(TicketExample.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path.name}:{lineno}: fila inválida ({exc})") from exc
    return examples


def _preprocess(text: str) -> str:
    """Preprocesado TF-IDF: minúsculas y sin tildes. Es una función de módulo (no una
    lambda) para que joblib/pickle pueda guardar el vectorizador."""
    return strip_accents(text.lower())


def _default_dirs() -> tuple[Path, Path]:
    """Directorios por defecto: datos de entrenamiento y modelos persistidos."""
    from docuerp import paths

    return paths.CLASSIFIER_DATA, paths.INDEX / "classifier"


class TicketClassifier:
    """Clasificador de tickets entrenable, persistible y de carga perezosa.

    Args:
        kind: ``tfidf`` (baseline) o ``embeddings``.
        embedder_name: embedder para ``kind="embeddings"`` (``fake`` en pruebas).
        model_dir: dónde guardar/cargar ``{kind}.joblib``.
        data_dir: dónde están ``tickets_train.jsonl`` y ``tickets_test.jsonl``.
        min_confidence: umbral bajo el cual se pide revisión humana.
        seed: semilla de la regresión logística.
    """

    def __init__(
        self,
        kind: ClassifierKind = "embeddings",
        *,
        embedder_name: str = "bge-m3",
        model_dir: Path | None = None,
        data_dir: Path | None = None,
        min_confidence: float = 0.60,
        seed: int = 42,
    ) -> None:
        if kind not in CLASSIFIER_KINDS:
            raise ValueError(f"Tipo de clasificador desconocido: {kind!r}")
        default_data, default_models = _default_dirs()
        self.kind: ClassifierKind = kind
        self.embedder_name = embedder_name
        self.model_dir = model_dir or default_models
        self.data_dir = data_dir or default_data
        self.min_confidence = min_confidence
        self.seed = seed
        self._vectorizer: Any = None
        self._models: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        """Archivo joblib del modelo de este tipo."""
        return self.model_dir / f"{self.kind}.joblib"

    @property
    def is_fitted(self) -> bool:
        """``True`` si ya hay modelos entrenados en memoria."""
        return bool(self._models)

    # ------------------------------------------------------------------ representación
    def _featurize(self, texts: list[str], *, fit: bool = False) -> Any:
        """Convierte textos en la matriz de características según ``kind``."""
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            if fit:
                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    preprocessor=_preprocess,
                )
                return self._vectorizer.fit_transform(texts)
            return self._vectorizer.transform(texts)
        from docuerp.embeddings import get_embedder

        embedder = get_embedder(self.embedder_name)  # type: ignore[arg-type]
        return embedder.embed_documents(texts)

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, examples: list[TicketExample]) -> TicketClassifier:
        """Entrena una regresión logística por objetivo."""
        from sklearn.linear_model import LogisticRegression

        if len(examples) < 2:
            raise ValueError("Se necesitan al menos 2 ejemplos para entrenar el clasificador")
        features = self._featurize([ex.text for ex in examples], fit=True)
        models: dict[str, Any] = {}
        for target in TARGETS:
            labels = [getattr(ex, target) for ex in examples]
            if len(set(labels)) < 2:
                raise ValueError(f"El objetivo {target!r} necesita al menos 2 clases distintas")
            c_value, cv_score = self._select_c(features, labels)
            model = LogisticRegression(
                C=c_value, max_iter=5000, class_weight="balanced", random_state=self.seed
            )
            model.fit(features, labels)
            models[target] = model
            self.metrics[f"{target}_lr_c"] = c_value
            if np.isfinite(cv_score):  # con datos mínimos (pruebas) la CV no es calculable
                self.metrics[f"{target}_cv_accuracy_train"] = cv_score
        self._models = models
        return self

    def _select_c(self, features: Any, labels: list[str]) -> tuple[float, float]:
        """Regularización ``C`` por validación cruzada estratificada **solo en train**.

        Con ``C=1`` fijo, la LR sobre embeddings normalizados quedaba subajustada (la
        prioridad bajaba a 0.48 de accuracy). Empates → el ``C`` más regularizado.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        _, counts = np.unique(labels, return_counts=True)
        folds = int(min(5, counts.min()))
        if folds < 2:
            return 1.0, float("nan")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        best_c, best_score = 1.0, -1.0
        for c_value in C_GRID:
            model = LogisticRegression(
                C=c_value, max_iter=5000, class_weight="balanced", random_state=self.seed
            )
            score = float(np.nanmean(cross_val_score(model, features, labels, cv=cv)))
            if np.isfinite(score) and score > best_score + 1e-9:
                best_c, best_score = c_value, score
        return (best_c, best_score) if best_score >= 0 else (1.0, float("nan"))

    def evaluate(self, examples: list[TicketExample]) -> dict[str, float]:
        """Accuracy y F1 macro por objetivo sobre ``examples``."""
        from sklearn.metrics import accuracy_score, f1_score

        if not self.is_fitted:
            raise ClassifierNotReadyError("El clasificador no está entrenado")
        features = self._featurize([ex.text for ex in examples])
        metrics: dict[str, float] = {"n_test": float(len(examples))}
        for target, model in self._models.items():
            gold = [getattr(ex, target) for ex in examples]
            pred = model.predict(features)
            metrics[f"{target}_accuracy"] = float(accuracy_score(gold, pred))
            metrics[f"{target}_f1_macro"] = float(
                f1_score(gold, pred, average="macro", zero_division=0)
            )
        return metrics

    # ------------------------------------------------------------------ persistencia
    def save(self, path: Path | None = None) -> Path:
        """Guarda vectorizador, modelos y métricas con joblib."""
        import joblib

        target = path or self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self.kind,
                "embedder_name": self.embedder_name,
                "vectorizer": self._vectorizer,
                "models": self._models,
                "metrics": self.metrics,
            },
            target,
        )
        return target

    def load(self, path: Path | None = None) -> TicketClassifier:
        """Carga un modelo guardado (valida que sea del mismo tipo)."""
        import joblib

        bundle = joblib.load(path or self.model_path)
        if bundle.get("kind") != self.kind:
            raise ValueError(f"El modelo guardado es {bundle.get('kind')!r}, no {self.kind!r}")
        self.embedder_name = bundle["embedder_name"]
        self._vectorizer = bundle["vectorizer"]
        self._models = bundle["models"]
        self.metrics = bundle.get("metrics", {})
        return self

    def ensure_ready(self) -> None:
        """Carga el modelo del disco o, si no existe, lo entrena con los datos disponibles.

        Raises:
            ClassifierNotReadyError: si no hay ni modelo guardado ni datos de entrenamiento.
        """
        with self._lock:
            if self.is_fitted:
                return
            if self.model_path.is_file():
                self.load()
                return
            train_path = self.data_dir / "tickets_train.jsonl"
            if not train_path.is_file():
                raise ClassifierNotReadyError(
                    f"No hay modelo en {self.model_path} ni datos en {train_path}"
                )
            self.train_and_save()

    def train_and_save(self) -> dict[str, float]:
        """Entrena con ``tickets_train.jsonl``, evalúa con ``tickets_test.jsonl`` si existe
        y guarda el modelo. Devuelve las métricas."""
        train = load_examples(self.data_dir / "tickets_train.jsonl")
        self.fit(train)
        metrics: dict[str, float] = {"n_train": float(len(train)), **self.metrics}
        test_path = self.data_dir / "tickets_test.jsonl"
        if test_path.is_file():
            metrics.update(self.evaluate(load_examples(test_path)))
        self.metrics = metrics
        self.save()
        (self.model_dir / f"{self.kind}.metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics

    # ------------------------------------------------------------------ predicción
    def predict(self, text: str) -> dict[str, Any]:
        """Predice categoría, prioridad y módulo con su confianza (probabilidad máxima)."""
        self.ensure_ready()
        features = self._featurize([text])
        out: dict[str, Any] = {"kind": self.kind}
        for target, model in self._models.items():
            probs = np.asarray(model.predict_proba(features)[0], dtype=float)
            order = np.argsort(probs)[::-1]
            out[target] = str(model.classes_[order[0]])
            out[f"{target}_confidence"] = round(float(probs[order[0]]), 4)
            if target == "category":
                out["category_alternatives"] = [
                    {"label": str(model.classes_[i]), "confidence": round(float(probs[i]), 4)}
                    for i in order[1:3]
                ]
        out["needs_human_review"] = out["category_confidence"] < self.min_confidence
        return out


class HybridTicketClassifier:
    """Combina TF-IDF y embeddings eligiendo, **por objetivo**, el que tuvo mejor accuracy
    de validación cruzada en train (nunca se mira el test para decidir).

    En los datos actuales la categoría y el módulo salen mejor con embeddings (semántica) y
    la prioridad con TF-IDF (pistas léxicas como «urgente» o «no podemos facturar»).
    """

    kind = "hybrid"

    def __init__(
        self,
        tfidf: TicketClassifier,
        embeddings: TicketClassifier,
        *,
        min_confidence: float = 0.60,
    ) -> None:
        self.tfidf = tfidf
        self.embeddings = embeddings
        self.min_confidence = min_confidence

    def ensure_ready(self) -> None:
        """Carga (o entrena) ambos clasificadores."""
        self.tfidf.ensure_ready()
        self.embeddings.ensure_ready()

    def choice(self) -> dict[str, str]:
        """Representación elegida por objetivo según la CV en train (empate → embeddings)."""
        picked: dict[str, str] = {}
        for target in TARGETS:
            key = f"{target}_cv_accuracy_train"
            tfidf_cv = self.tfidf.metrics.get(key)
            emb_cv = self.embeddings.metrics.get(key)
            better_tfidf = tfidf_cv is not None and (emb_cv is None or tfidf_cv > emb_cv)
            picked[target] = "tfidf" if better_tfidf else "embeddings"
        return picked

    def predict(self, text: str) -> dict[str, Any]:
        """Predicción combinada con la misma forma que :meth:`TicketClassifier.predict`."""
        self.ensure_ready()
        picked = self.choice()
        by_kind = {"tfidf": self.tfidf.predict(text), "embeddings": self.embeddings.predict(text)}
        out: dict[str, Any] = {"kind": self.kind, "per_target": picked}
        for target in TARGETS:
            source = by_kind[picked[target]]
            out[target] = source[target]
            out[f"{target}_confidence"] = source[f"{target}_confidence"]
        out["category_alternatives"] = by_kind[picked["category"]].get("category_alternatives", [])
        out["needs_human_review"] = out["category_confidence"] < self.min_confidence
        return out


_DEFAULT_LOCK = threading.Lock()
_DEFAULT: TicketClassifier | HybridTicketClassifier | None = None


def get_default_classifier() -> TicketClassifier | HybridTicketClassifier:
    """Clasificador por defecto: híbrido por objetivo (TF-IDF + embeddings del embedder de
    ``settings``), perezoso: construirlo no carga modelos; eso ocurre al predecir."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            from docuerp.config import get_settings

            settings = get_settings()
            common = {
                "min_confidence": settings.agent.classifier.min_confidence,
                "seed": settings.seed,
            }
            _DEFAULT = HybridTicketClassifier(
                TicketClassifier("tfidf", **common),  # type: ignore[arg-type]
                TicketClassifier(
                    "embeddings",
                    embedder_name=settings.retrieval.embedder,
                    **common,  # type: ignore[arg-type]
                ),
                min_confidence=settings.agent.classifier.min_confidence,
            )
        return _DEFAULT


async def classify_ticket(
    args: ClassifyTicketInput,
    *,
    classifier: TicketClassifier | HybridTicketClassifier | None = None,
) -> ToolResult:
    """Predice categoría, prioridad y módulo; baja confianza → ``needs_human_review``."""
    active = classifier or get_default_classifier()
    try:
        # Entrenar/cargar y calcular embeddings es CPU intensivo: se hace en un hilo.
        prediction = await asyncio.to_thread(active.predict, args.text)
    except (ClassifierNotReadyError, FileNotFoundError, ValueError) as exc:
        log.warning("Clasificador no disponible: %s", exc)
        return ToolResult(
            tool="classify_ticket",
            ok=False,
            error="El clasificador de tickets no está disponible (modelo no entrenado).",
            data={"unavailable": True, "error_type": type(exc).__name__},
        )
    return ToolResult(
        tool="classify_ticket",
        ok=True,
        data=prediction,
        needs_human_review=bool(prediction["needs_human_review"]),
    )


def train_classifier(
    kind: str = "embeddings",
    *,
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    embedder_name: str | None = None,
) -> dict[str, float]:
    """Entrena el clasificador (``tfidf`` | ``embeddings``) y devuelve métricas.

    Las métricas incluyen ``n_train`` y, si existe ``tickets_test.jsonl``, ``n_test`` y
    ``{category,priority,module}_{accuracy,f1_macro}``.
    """
    from docuerp.config import get_settings

    if kind not in CLASSIFIER_KINDS:
        raise ValueError(f"Tipo de clasificador desconocido: {kind!r} (tfidf | embeddings)")
    settings = get_settings()
    classifier = TicketClassifier(
        kind,  # type: ignore[arg-type]
        embedder_name=embedder_name or settings.retrieval.embedder,
        model_dir=model_dir,
        data_dir=data_dir,
        min_confidence=settings.agent.classifier.min_confidence,
        seed=settings.seed,
    )
    return classifier.train_and_save()


def build_classify_spec(
    classifier: TicketClassifier | HybridTicketClassifier, *, settings: Settings
) -> ToolSpec:
    """Especificación registrable de ``classify_ticket`` ligada a ``classifier``."""

    async def _handler(args: ClassifyTicketInput, ctx: ToolContext) -> ToolResult:
        del ctx  # clasificar texto no depende del usuario: no lee datos de nadie
        return await classify_ticket(args, classifier=classifier)

    return ToolSpec(
        name="classify_ticket",
        description="Sugiere categoría, prioridad y módulo de un texto de ticket (≤2000 "
        "caracteres); baja confianza → revisión humana.",
        input_model=ClassifyTicketInput,
        allowed_roles=tool_roles("classify_ticket"),
        timeout_s=settings.agent.tool_timeout_s,
        max_calls_per_query=1,
        handler=_handler,
    )

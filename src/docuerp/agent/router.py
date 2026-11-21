"""Enrutador híbrido determinista: reglas + clasificador sobre embeddings + fuera de dominio.

El router decide qué rama del grafo atiende la consulta. Combina tres capas, de la más
precisa a la más general:

1. **Reglas** (regex sobre el texto en minúsculas y sin tildes): verbos de creación junto a
   "ticket/caso/incidencia" → ``ticket_create``; "clasifica/categoriza" →
   ``ticket_classify``; un identificador ``TCK-\\d{5}`` o "estado del ticket" →
   ``ticket_status``. Las reglas ganan porque son explicables y no fallan con
   identificadores exactos. Una pregunta de procedimiento ("¿cómo creo un ticket en
   Helpdesk?") **no** dispara la regla de creación: es una consulta documental.
2. **Clasificador de intención**: regresión logística sobre embeddings (bge-m3) entrenada
   con ``data/router/intents.jsonl`` (split ``train``).
3. **Fuera de dominio**: si la similitud coseno máxima entre la consulta y los ejemplos de
   entrenamiento del dominio (todas las intenciones salvo ``out_of_domain``) es menor que
   ``agent.router.ood_threshold`` (0.35), la consulta se trata como fuera de dominio. Si la
   probabilidad del clasificador es menor que ``agent.router.min_confidence`` (0.55), va a
   la rama documental (``low_confidence``): nunca se dispara una acción con dudas.
4. **Guard de acciones** (``action_guard``): el clasificador solo puede proponer
   ``ticket_create`` si el texto pide crear algo de forma explícita; si no, rama documental.

Si no hay modelo ni datos para entrenarlo, el router degrada a ``doc_query`` (método
``fallback``): la rama documental es la más segura porque se abstiene
cuando no encuentra soporte.

El modelo entrenado se guarda con joblib en ``data/index/router/intent_lr.joblib``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp.schemas import Intent
from docuerp.text import strip_accents

log = logging.getLogger(__name__)

RouteMethod = Literal[
    "rules", "classifier", "ood_similarity", "low_confidence", "action_guard", "fallback"
]
"""Cómo se tomó la decisión (se muestra en la traza)."""

C_GRID: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0, 256.0)
"""Rejilla de regularización para la regresión logística del router."""

RULE_CONFIDENCE = 0.95
"""Confianza asignada a una coincidencia de regla (alta pero no absoluta)."""

FALLBACK_CONFIDENCE = 0.5
"""Confianza de la decisión por defecto cuando no hay clasificador."""

TICKET_ID_RE = re.compile(r"\bTCK-\d{5}\b", re.IGNORECASE)

# Las expresiones siguientes se aplican sobre texto en minúsculas y sin tildes.
_TICKET_NOUN = r"(?:ticket|tiquete|caso|incidencia|incidente|solicitud de soporte)s?"
_CREATE_RE = re.compile(
    r"\b(?:crea|crear|creame|creen|abre|abrir|abreme|abra|registra|registrar|registrame|"
    r"registre|levanta|levantar|levantame|levante|genera|generar|genere|reporta|reportar|"
    r"open|create|file)\b(?:\W+\w+){0,3}?\W+" + _TICKET_NOUN + r"\b"
    r"|\bnuev[oa]\s+" + _TICKET_NOUN + r"\b"
)
_HOWTO_RE = re.compile(
    r"\b(?:como|pasos para|procedimiento para|manera de|forma de|how to|how do i)\s+"
    r"(?:se\s+|puedo\s+|debo\s+)?"
    r"(?:crea|crear|creo|abre|abrir|abro|registra|registrar|registro|levanta|levantar|create|open)\b"
)
_CLASSIFY_RE = re.compile(
    r"\b(?:clasifica|clasificar|clasifique|clasificame|clasifiques|categoriza|categorizar|"
    r"categorice|categorizame|que categoria|que prioridad le|triage|triaje)\b"
)
_PROCEDURE_RE = re.compile(
    r"\b(?:procedimiento|pasos|como se|como puedo|como debo|politica|proceso)\b"
)
"""Pregunta por el procedimiento (documental), no por el estado de un ticket concreto."""
_ACTION_CUE_RE = re.compile(
    # Imperativos, infinitivos y enclíticos de alta: crea, creale, abre, ábreme, abrime,
    # registrar, levántale, reporta, escala, genera… (no «creo», que suele ser opinión).
    r"\b(?:abr|cre|registr|levant|report|escal|gener)(?:a|e|i|an|en|ar|ir|er)"
    r"(?:me|le|les|nos|lo|la)?\b"
    r"|\b(?:open|create|file|log)\b"
    r"|\bda\w* de alta\b"
    r"|\b(?:quiero|necesito|me gustaria|hay que|toca)\s+(?:abrir|crear|registrar|levantar|"
    r"reportar|poner|meter)\b"
    r"|\b" + _TICKET_NOUN + r"\b"
)
"""Señal explícita de que el usuario pide **crear** algo (verbo de alta o sustantivo de
ticket). Sin ella, el clasificador no puede disparar ``ticket_create``: una pregunta que solo
describe un problema («No me deja validar, ¿qué hago?») va a la rama documental."""
_STATUS_RE = re.compile(
    r"\b(?:estado|estatus|status|situacion|avance|seguimiento)\b(?:\W+\w+){0,4}?\W+"
    + _TICKET_NOUN
    + r"\b"
)


class RouteDecision(BaseModel):
    """Decisión del enrutador."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    method: RouteMethod = "rules"


class IntentExample(BaseModel):
    """Fila de ``data/router/intents.jsonl`` (SPEC §10)."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    intent: Intent
    split: Literal["train", "test"] = "train"
    hard: bool = False


def load_intents(path: Path) -> list[IntentExample]:
    """Lee y valida el JSONL de intenciones.

    Raises:
        FileNotFoundError: si no existe.
        ValueError: si una línea es inválida (indica el número de línea).
    """
    rows: list[IntentExample] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(IntentExample.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path.name}:{lineno}: fila inválida ({exc})") from exc
    return rows


def rule_intent(text: str) -> tuple[Intent, str] | None:
    """Aplica las reglas deterministas; devuelve ``(intent, motivo)`` o ``None``."""
    folded = strip_accents(text.lower())
    if _CREATE_RE.search(folded) and not _HOWTO_RE.search(folded):
        return "ticket_create", "regla: verbo de creación + ticket"
    if _CLASSIFY_RE.search(folded):
        return "ticket_classify", "regla: pedido de clasificación"
    if TICKET_ID_RE.search(text):
        return "ticket_status", "regla: identificador TCK-xxxxx"
    if _STATUS_RE.search(folded) and not _PROCEDURE_RE.search(folded):
        return "ticket_status", "regla: consulta de estado de ticket"
    return None


def _default_paths() -> tuple[Path, Path]:
    """Ruta por defecto de los datos de intención y del modelo persistido."""
    from docuerp import paths

    return paths.ROUTER_DATA / "intents.jsonl", paths.INDEX / "router" / "intent_lr.joblib"


class IntentRouter:
    """Router con modelo perezoso; ``route`` es determinista para una entrada dada.

    Args:
        embedder_name: embedder del clasificador (``fake`` en pruebas).
        model_path: archivo joblib del modelo.
        data_path: JSONL de intenciones para entrenar si el modelo no existe.
        ood_threshold: similitud mínima al dominio (por defecto de ``settings``).
        min_confidence: probabilidad mínima del clasificador (por defecto de ``settings``).
        seed: semilla de la regresión logística.
    """

    def __init__(
        self,
        *,
        embedder_name: str | None = None,
        model_path: Path | None = None,
        data_path: Path | None = None,
        ood_threshold: float | None = None,
        min_confidence: float | None = None,
        seed: int | None = None,
    ) -> None:
        from docuerp.config import get_settings

        settings = get_settings()
        default_data, default_model = _default_paths()
        self.embedder_name = embedder_name or settings.retrieval.embedder
        self.model_path = model_path or default_model
        self.data_path = data_path or default_data
        self.ood_threshold = (
            settings.agent.router.ood_threshold if ood_threshold is None else ood_threshold
        )
        self.min_confidence = (
            settings.agent.router.min_confidence if min_confidence is None else min_confidence
        )
        self.seed = settings.seed if seed is None else seed
        self._model: Any = None
        self._references: np.ndarray | None = None
        self.c_value: float = 1.0
        self.cv_accuracy: float = float("nan")
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ modelo
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embeddings L2-normalizados (producto punto = coseno)."""
        from docuerp.embeddings import get_embedder

        embedder = get_embedder(self.embedder_name)  # type: ignore[arg-type]
        return np.asarray(embedder.embed_documents(texts), dtype=np.float32)

    def fit(self, examples: list[IntentExample]) -> IntentRouter:
        """Entrena la regresión logística y guarda los vectores de referencia del dominio."""
        from sklearn.linear_model import LogisticRegression

        train = [ex for ex in examples if ex.split == "train"]
        labels = [ex.intent for ex in train]
        if len(set(labels)) < 2:
            raise ValueError("Se necesitan al menos 2 intenciones distintas en el split train")
        features = self._embed([ex.text for ex in train])
        c_value, cv_accuracy = self._select_c(features, labels)
        model = LogisticRegression(
            C=c_value, max_iter=5000, class_weight="balanced", random_state=self.seed
        )
        model.fit(features, labels)
        self.c_value, self.cv_accuracy = c_value, cv_accuracy
        in_domain = np.array([label != "out_of_domain" for label in labels])
        self._model = model
        self._references = features[in_domain] if in_domain.any() else features
        self._loaded = True
        return self

    def _select_c(self, features: np.ndarray, labels: list[str]) -> tuple[float, float]:
        """Elige la regularización ``C`` por validación cruzada estratificada **solo en train**.

        Con 1024 dimensiones y ~200 ejemplos, ``C=1`` deja probabilidades difusas (casi todo
        caía bajo ``min_confidence``); la búsqueda en rejilla evita fijarlo a ojo. Empates →
        el ``C`` más pequeño (más regularizado).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        counts = np.unique(labels, return_counts=True)[1]
        folds = int(min(5, counts.min()))
        if folds < 2:
            return 1.0, float("nan")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        best_c, best_score = 1.0, -1.0
        for c_value in C_GRID:
            model = LogisticRegression(
                C=c_value, max_iter=5000, class_weight="balanced", random_state=self.seed
            )
            score = float(cross_val_score(model, features, labels, cv=cv).mean())
            if score > best_score + 1e-9:
                best_c, best_score = c_value, score
        return best_c, best_score

    def evaluate(self, examples: list[IntentExample]) -> dict[str, float]:
        """Accuracy y F1 macro del router **completo** (reglas + modelo + OOD) en ``test``."""
        from sklearn.metrics import accuracy_score, f1_score

        test = [ex for ex in examples if ex.split == "test"]
        if not test:
            return {"n_test": 0.0}
        gold = [ex.intent for ex in test]
        pred = [self.decide(ex.text).intent for ex in test]
        metrics = {
            "n_test": float(len(test)),
            "accuracy": float(accuracy_score(gold, pred)),
            "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        }
        hard = [(g, p) for ex, g, p in zip(test, gold, pred, strict=True) if ex.hard]
        if hard:
            metrics["hard_accuracy"] = float(np.mean([g == p for g, p in hard]))
        return metrics

    def save(self, path: Path | None = None) -> Path:
        """Persiste modelo, referencias y configuración con joblib."""
        import joblib

        if self._model is None or self._references is None:
            raise RuntimeError("No hay modelo entrenado que guardar")
        target = path or self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embedder_name": self.embedder_name,
                "model": self._model,
                "references": self._references,
            },
            target,
        )
        return target

    def load(self, path: Path | None = None) -> IntentRouter:
        """Carga un modelo persistido (usa el embedder con el que se entrenó)."""
        import joblib

        bundle = joblib.load(path or self.model_path)
        self.embedder_name = bundle["embedder_name"]
        self._model = bundle["model"]
        self._references = np.asarray(bundle["references"], dtype=np.float32)
        self._loaded = True
        return self

    def _ensure_model(self) -> bool:
        """Carga o entrena el modelo una sola vez; ``False`` si no hay forma de tenerlo."""
        with self._lock:
            if self._loaded:
                return self._model is not None
            self._loaded = True
            try:
                if self.model_path.is_file():
                    self.load()
                elif self.data_path.is_file():
                    self.fit(load_intents(self.data_path))
                    self.save()
                else:
                    log.warning("Router sin modelo ni datos: se usará la decisión por defecto")
            except (OSError, ValueError, KeyError) as exc:
                log.warning("No se pudo preparar el clasificador de intención: %s", exc)
                self._model = None
            return self._model is not None

    # ------------------------------------------------------------------ decisión
    def decide(self, text: str) -> RouteDecision:
        """Decide la intención de ``text`` como :class:`RouteDecision`."""
        rule = rule_intent(text)
        if rule is not None:
            intent, why = rule
            return RouteDecision(
                intent=intent, confidence=RULE_CONFIDENCE, rationale=why, method="rules"
            )
        if not self._ensure_model() or self._references is None:
            return RouteDecision(
                intent="doc_query",
                confidence=FALLBACK_CONFIDENCE,
                rationale="sin clasificador entrenado: se usa la rama documental (con gate)",
                method="fallback",
            )
        vector = self._embed([text])[0]
        similarity = float(np.max(self._references @ vector))
        if similarity < self.ood_threshold:
            return RouteDecision(
                intent="out_of_domain",
                confidence=round(min(1.0, max(0.0, 1.0 - similarity)), 4),
                rationale=f"similitud máxima al dominio {similarity:.2f} < "
                f"umbral {self.ood_threshold:.2f}",
                method="ood_similarity",
            )
        probs = np.asarray(self._model.predict_proba(vector.reshape(1, -1))[0], dtype=float)
        best = int(np.argmax(probs))
        label: Intent = str(self._model.classes_[best])  # type: ignore[assignment]
        confidence = round(float(probs[best]), 4)
        if confidence < self.min_confidence:
            # Dentro del dominio pero sin una intención clara: la rama documental es la opción
            # segura, porque el gate de evidencia se abstiene si no hay soporte en el corpus.
            # Las acciones (consultar/crear/clasificar tickets) nunca se disparan con dudas.
            return RouteDecision(
                intent="doc_query",
                confidence=confidence,
                rationale=f"confianza del clasificador {confidence:.2f} < "
                f"{self.min_confidence:.2f} (mejor candidata: {label}); se usa la rama "
                "documental con gate de evidencia",
                method="low_confidence",
            )
        if label == "ticket_create" and not _ACTION_CUE_RE.search(strip_accents(text.lower())):
            # Una acción de escritura solo se propone si el usuario la pide de forma explícita.
            return RouteDecision(
                intent="doc_query",
                confidence=confidence,
                rationale=f"el clasificador sugirió ticket_create (p={confidence:.2f}) sin un "
                "pedido explícito de creación; se usa la rama documental",
                method="action_guard",
            )
        return RouteDecision(
            intent=label,
            confidence=confidence,
            rationale=f"clasificador: {label} (p={confidence:.2f}, similitud {similarity:.2f})",
            method="classifier",
        )

    def route(self, text: str) -> tuple[Intent, float, str]:
        """Devuelve ``(intent, confidence, rationale)``."""
        decision = self.decide(text)
        return decision.intent, decision.confidence, decision.rationale


def train_router(
    out_path: str | Path | None = None,
    *,
    data_path: Path | None = None,
    embedder_name: str | None = None,
) -> dict[str, float]:
    """Entrena el clasificador de intención, lo guarda y devuelve métricas en ``test``.

    Métricas: ``n_train``, ``n_test``, ``accuracy``, ``f1_macro`` y ``hard_accuracy`` (si
    hay casos ``hard``). La evaluación usa el router completo, igual que en producción.

    Raises:
        FileNotFoundError: si no existe el JSONL de intenciones.
    """
    router = IntentRouter(
        embedder_name=embedder_name,
        model_path=Path(out_path) if out_path is not None else None,
        data_path=data_path,
    )
    examples = load_intents(router.data_path)
    router.fit(examples)
    router.save()
    metrics = {"n_train": float(sum(ex.split == "train" for ex in examples))}
    metrics["lr_c"] = float(getattr(router, "c_value", 1.0))
    metrics["cv_accuracy_train"] = float(getattr(router, "cv_accuracy", float("nan")))
    metrics.update(router.evaluate(examples))
    return metrics

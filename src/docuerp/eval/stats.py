"""Estadística de la evaluación.

* Intervalos de confianza del 95 % por **bootstrap percentil** (10 000 remuestreos, semilla
  fija): remuestrear *con reemplazo* los casos y recalcular la media.
* Comparación **pareada** baseline vs variante: como ambos sistemas responden las mismas
  preguntas, se trabaja con la diferencia por caso. Se reporta un IC bootstrap de la
  diferencia, un valor-p por **prueba de permutación** (cambio aleatorio de signo de las
  diferencias), un valor-p por bootstrap centrado y, de apoyo, Wilcoxon.
* Acuerdo entre anotadores: **kappa de Cohen** (etiquetas categóricas) y **Jaccard**
  (conjuntos, p. ej. páginas de evidencia anotadas).

Todo es determinista dada la semilla.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Sequence

import numpy as np
from scipy import stats as sp_stats

DEFAULT_RESAMPLES = 10_000
"""Número de remuestreos bootstrap (exigido por el diseño de evaluación)."""
DEFAULT_SEED = 42
EXACT_PERMUTATION_MAX_N = 16
"""Hasta este tamaño la permutación es exacta (2^16 = 65 536 combinaciones de signos)."""


def _as_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convierte a ``ndarray`` float y valida que no esté vacío ni tenga NaN."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} debe ser una secuencia no vacía de números")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contiene NaN; filtre los casos sin valor antes de calcular")
    return arr


def _resample_means(arr: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Medias de ``n_resamples`` remuestreos con reemplazo (vectorizado por bloques)."""
    n = arr.size
    means = np.empty(n_resamples, dtype=float)
    block = max(1, 2_000_000 // n)  # limita la memoria a ~16 MB por bloque
    for start in range(0, n_resamples, block):
        stop = min(n_resamples, start + block)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = arr[idx].mean(axis=1)
    return means


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IC bootstrap percentil de la media.

    Returns:
        ``(media, límite_inferior, límite_superior)``. Con un único valor el intervalo
        degenera en ``(v, v, v)``.

    Raises:
        ValueError: ``values`` vacío, con NaN, o ``alpha`` fuera de (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe estar en (0, 1)")
    arr = _as_array(values, "values")
    mean = float(arr.mean())
    if arr.size == 1 or float(arr.min()) == float(arr.max()):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = _resample_means(arr, n_resamples, rng)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(low), float(high)


def _permutation_p_value(diffs: np.ndarray, n_resamples: int, rng: np.random.Generator) -> float:
    """Valor-p bilateral de H0 «la diferencia media es 0» por cambio de signo.

    Bajo H0 el signo de cada diferencia es intercambiable. Con ``n ≤ 16`` se enumeran
    todos los signos (prueba exacta); con más casos se muestrean ``n_resamples`` y se aplica
    la corrección ``(b + 1) / (B + 1)`` para no devolver un p-valor exactamente 0.
    """
    n = diffs.size
    observed = abs(float(diffs.mean()))
    tol = 1e-12
    if n <= EXACT_PERMUTATION_MAX_N:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        stats = np.abs((signs * diffs).mean(axis=1))
        return float((stats >= observed - tol).mean())
    exceed = 0
    block = max(1, 2_000_000 // n)
    done = 0
    while done < n_resamples:
        size = min(block, n_resamples - done)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        exceed += int((np.abs((signs * diffs).mean(axis=1)) >= observed - tol).sum())
        done += size
    return (exceed + 1) / (n_resamples + 1)


def paired_test(
    baseline: list[float],
    final: list[float],
    *,
    seed: int = 42,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Prueba pareada baseline vs variante sobre las mismas unidades (casos).

    Args:
        baseline: métrica por caso del sistema base.
        final: métrica por caso de la variante, en el mismo orden.
        seed: semilla del remuestreo.
        n_resamples: remuestreos bootstrap/permutación.
        alpha: nivel del IC de la diferencia.

    Returns:
        Diccionario con ``delta`` (media de ``final - baseline``), ``ci_low`` y
        ``ci_high`` (IC bootstrap pareado de la diferencia), ``p_value`` (permutación,
        el valor principal), ``p_value_bootstrap`` (bootstrap centrado), ``p_wilcoxon``
        (``nan`` si todas las diferencias son 0), ``cohens_dz`` (tamaño de efecto pareado),
        ``n`` y ``significant`` (1.0 si ``p_value < alpha``).

    Raises:
        ValueError: longitudes distintas, vacías o con NaN.
    """
    if len(baseline) != len(final):
        raise ValueError(
            f"Las series pareadas deben tener igual longitud ({len(baseline)} != {len(final)})"
        )
    base_arr = _as_array(baseline, "baseline")
    final_arr = _as_array(final, "final")
    diffs = final_arr - base_arr
    delta = float(diffs.mean())
    n = int(diffs.size)

    if np.allclose(diffs, 0.0):  # sin diferencias: nada que contrastar
        return {
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "p_value_bootstrap": 1.0,
            "p_wilcoxon": float("nan"),
            "cohens_dz": 0.0,
            "n": float(n),
            "significant": 0.0,
        }

    rng = np.random.default_rng(seed)
    _, ci_low, ci_high = bootstrap_ci(
        diffs.tolist(), n_resamples=n_resamples, alpha=alpha, seed=seed
    )
    p_perm = _permutation_p_value(diffs, n_resamples, rng)

    # Bootstrap bajo H0: se centran las diferencias (media 0) y se mide cuán extrema es la
    # media observada frente a las medias remuestreadas.
    centered = diffs - delta
    boot = _resample_means(centered, n_resamples, rng)
    p_boot = float((np.sum(np.abs(boot) >= abs(delta) - 1e-12) + 1) / (n_resamples + 1))

    try:
        p_wilcoxon = float(sp_stats.wilcoxon(diffs).pvalue)
    except ValueError:  # demasiadas diferencias nulas / muestra insuficiente
        p_wilcoxon = float("nan")

    std = float(diffs.std(ddof=1)) if n > 1 else 0.0
    dz = delta / std if std > 0 else math.copysign(math.inf, delta)
    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_perm,
        "p_value_bootstrap": p_boot,
        "p_wilcoxon": p_wilcoxon,
        "cohens_dz": dz,
        "n": float(n),
        "significant": 1.0 if p_perm < alpha else 0.0,
    }


# ---------------------------------------------------------------------------
# Acuerdo entre anotadores
# ---------------------------------------------------------------------------


def cohen_kappa(rater_a: Sequence[Hashable], rater_b: Sequence[Hashable]) -> float:
    """Kappa de Cohen entre dos anotadores con etiquetas categóricas.

    ``κ = (p_o - p_e) / (1 - p_e)``, donde ``p_o`` es el acuerdo observado y ``p_e`` el
    acuerdo esperado por azar según las frecuencias marginales de cada anotador.
    Si ambos usan una única etiqueta idéntica (``p_e = 1``) se devuelve 1.0.

    Raises:
        ValueError: listas vacías o de distinta longitud.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Los anotadores deben etiquetar la misma cantidad de ítems")
    if not rater_a:
        raise ValueError("No hay ítems para calcular el acuerdo")
    n = len(rater_a)
    observed = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / n
    labels = set(rater_a) | set(rater_b)
    expected = sum((rater_a.count(lab) / n) * (rater_b.count(lab) / n) for lab in labels)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def jaccard(
    a: set[Hashable] | frozenset[Hashable], b: set[Hashable] | frozenset[Hashable]
) -> float:
    """Índice de Jaccard ``|A ∩ B| / |A ∪ B|`` (dos conjuntos vacíos → 1.0)."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def mean_jaccard(sets_a: Sequence[set[Hashable]], sets_b: Sequence[set[Hashable]]) -> float:
    """Jaccard medio por ítem entre dos anotadores que marcan conjuntos (p. ej. páginas).

    Raises:
        ValueError: listas vacías o de distinta longitud.
    """
    if len(sets_a) != len(sets_b):
        raise ValueError("Los anotadores deben anotar la misma cantidad de ítems")
    if not sets_a:
        raise ValueError("No hay ítems para calcular el acuerdo")
    return sum(jaccard(a, b) for a, b in zip(sets_a, sets_b, strict=True)) / len(sets_a)

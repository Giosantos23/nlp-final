"""Estimación de costo en USD a precio de lista (``configs/settings.yaml: llm.pricing``).

Las tarifas son USD por millón de tokens. La lectura de caché usa el precio propio del
modelo (``cache_read``) si está definido —Sonnet 5.5 y Opus 5.5 la cobran a 0.05× la
entrada— o, si no, ``cache_read_multiplier`` veces el precio de entrada; la escritura de
caché cuesta ``cache_write_multiplier`` veces. En la API de Anthropic ``input_tokens``
**no** incluye los tokens de caché, por eso se suman por separado sin doble conteo.
"""

from __future__ import annotations

from docuerp.config import LLMSettings, get_settings

_TOKENS_PER_PRICE_UNIT = 1_000_000


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    *,
    settings: LLMSettings | None = None,
) -> float:
    """Costo estimado de una llamada; 0.0 si el modelo no está tarifado.

    Args:
        model: ID exacto del modelo (``claude-haiku-5-5``...).
        input_tokens: tokens de entrada normales (sin caché).
        output_tokens: tokens generados.
        cache_read_tokens: tokens leídos desde la caché de prompt.
        cache_creation_tokens: tokens escritos en la caché de prompt.
        settings: sección ``llm`` de la configuración (por defecto la global).

    Raises:
        ValueError: si algún conteo de tokens es negativo.
    """
    counts = (input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens)
    if any(count < 0 for count in counts):
        raise ValueError("los conteos de tokens no pueden ser negativos")
    llm = settings or get_settings().llm
    price = llm.price_for(model)
    if price is None:
        return 0.0
    cache_read_price = (
        price.cache_read
        if price.cache_read is not None
        else price.input * llm.cache_read_multiplier
    )
    input_cost = (
        input_tokens + cache_creation_tokens * llm.cache_write_multiplier
    ) * price.input + cache_read_tokens * cache_read_price
    output_cost = output_tokens * price.output
    return (input_cost + output_cost) / _TOKENS_PER_PRICE_UNIT

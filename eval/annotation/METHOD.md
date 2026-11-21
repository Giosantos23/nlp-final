# Protocolo de anotación del banco

El banco `eval/benchmark.jsonl` tiene 60 casos: 42 de test y 18 de dev. Se construyó con cuatro pasadas independientes asistidas por IA. La última es una adjudicación. Las cuatro quedan versionadas aquí para que el acuerdo se pueda volver a calcular.

| Archivo | Pasada | Qué contiene |
|---|---|---|
| `author.jsonl` | Autor | Redacta los 60 casos completos. La evidencia `(fuente, página)` está verificada contra el texto real de cada página. |
| `annotator_a.jsonl` | Anotador A | Recibe solo la pregunta, el rol, el usuario y la versión, sin las etiquetas del autor. Etiqueta comportamiento, respondibilidad, intención, herramienta y evidencia. |
| `annotator_b.jsonl` | Anotador B | Igual que A y sin ver a A ni al autor. |
| `adjudicated.jsonl` | Adjudicación | Parte del voto de mayoría (2 de 3) en cada campo. Lo corrige cuando el código o el corpus lo contradicen y deja el motivo en `notes` con el prefijo `[adj: …]`. |

El acuerdo se mide con `scripts/annotation_agreement.py`, que escribe `reports/benchmark/annotation_agreement.{md,json}`. Calcula el κ de Cohen por par y el κ de Fleiss en los campos categóricos, y el Jaccard medio de la evidencia a nivel de página y de fuente.

## Correcciones posteriores a la adjudicación

Se hicieron dos cambios sobre `adjudicated.jsonl` al escribir `eval/benchmark.jsonl`. Ambos ocurrieron antes de correr cualquier evaluación.

1. **SEC-07 vuelve a `deny`.** La adjudicación la había pasado a `tool_call` porque el agente respondía a un ticket fuera de alcance sin marcar abstención. En lugar de ajustar la etiqueta al sistema, se corrigió el sistema. Ahora `permission_check` corta con una denegación explícita que no confirma si el ticket existe y no expone datos. Así, la etiqueta semántica de la mayoría (autor + A) se puede alcanzar.
2. **TOL-10 y DEV-13 declaran `simulated_outages`.** Con ese campo, la evaluación simula la caída de esa herramienta solo en ese caso, con un `ContextVar` que no afecta a las consultas concurrentes. Antes había que exportar `DOCUERP_TOOLS_UNAVAILABLE` para toda la corrida.

## Limitación declarada

Los tres etiquetadores y el adjudicador son modelos de lenguaje. El acuerdo alto mide consistencia entre pasadas independientes, no validez frente a un experto humano. La validación humana del equipo se registra en `human_review.md` cuando se completa. Hasta entonces, el informe la declara pendiente.

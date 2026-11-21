"""Agente LangGraph y herramientas (SPEC §5.4). Propiedad de la rama ``feat/agent``.

Módulos:

* :mod:`~docuerp.agent.service` — fachada pública (``answer_query``,
  ``answer_query_stream``, ``confirm_action``).
* :mod:`~docuerp.agent.graph` — ``StateGraph`` con presupuesto de pasos/herramientas y
  degradación controlada.
* :mod:`~docuerp.agent.router` — enrutador híbrido (reglas + regresión logística +
  fuera de dominio).
* :mod:`~docuerp.agent.deps` — dependencias inyectables (reales o *fakes*).
* :mod:`~docuerp.agent.templates` — respuestas deterministas en español.
* :mod:`~docuerp.agent.tools` — registro y las cuatro herramientas.
"""

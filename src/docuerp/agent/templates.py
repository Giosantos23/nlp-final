"""Respuestas deterministas del agente (plantillas en español).

Las herramientas de tickets devuelven datos estructurados y exactos; redactarlos con una
plantilla (en lugar del LLM) garantiza que el estado, la prioridad o el token que ve el
usuario son **exactamente** los de la BD: no hay alucinación posible, la latencia es casi
nula y el costo es cero. Aquí también viven los mensajes seguros de bloqueo, rechazo fuera
de dominio y degradación controlada.

Todas las funciones devuelven un :class:`~docuerp.schemas.Answer` (contrato §11).
"""

from __future__ import annotations

from typing import Any

from docuerp.schemas import ABSTENTION_MESSAGE, Answer

STATUS_LABELS: dict[str, str] = {
    "nuevo": "nuevo",
    "en_progreso": "en progreso",
    "en_espera": "en espera",
    "resuelto": "resuelto",
    "cerrado": "cerrado",
}

OUT_OF_DOMAIN_MESSAGE = (
    "Solo puedo ayudarte con soporte funcional de Odoo 17.0/18.0 (procedimientos, "
    "configuración, errores) y con tickets de soporte (estado, clasificación o creación). "
    "Reformula tu consulta dentro de ese ámbito."
)

ASK_TICKET_ID_MESSAGE = (
    "Para consultar el estado necesito el identificador del ticket con el formato "
    "TCK-00000 (cinco dígitos)."
)

ASK_TICKET_TEXT_MESSAGE = (
    "Describe el problema con más detalle (al menos una oración: qué intentabas hacer, en "
    "qué módulo y qué error viste) para poder preparar o clasificar el ticket."
)


def _answer(
    text: str,
    *,
    tool: str | None,
    abstained: bool = False,
    review: bool = False,
) -> Answer:
    """Construye un ``Answer`` sin citas (las respuestas de tickets no citan documentos)."""
    return Answer(
        answer=text, sources=[], abstained=abstained, tool_used=tool, needs_human_review=review
    )


def abstention(tool: str | None = "search_documents", *, review: bool = False) -> Answer:
    """Abstención por evidencia insuficiente (mensaje fijo que exige el guard de salida)."""
    return _answer(ABSTENTION_MESSAGE, tool=tool, abstained=True, review=review)


def blocked(reason: str | None) -> Answer:
    """Consulta bloqueada por el guard de entrada (se escala a revisión humana)."""
    detail = f" Motivo: {reason.rstrip(' .')}." if reason else ""
    return _answer(
        "Tu consulta fue bloqueada por las políticas de seguridad de DocuERP y no se "
        f"procesó.{detail} Si crees que es un error, contacta a un analista N2.",
        tool=None,
        abstained=True,
        review=True,
    )


def out_of_domain() -> Answer:
    """Rechazo controlado de una consulta fuera del dominio."""
    return _answer(OUT_OF_DOMAIN_MESSAGE, tool=None, abstained=True)


def degraded(reason: str, *, tool: str | None = None) -> Answer:
    """Respuesta segura ante una falla (herramienta caída, timeout, límite alcanzado)."""
    return _answer(
        f"No pude completar tu solicitud de forma segura: {reason} La consulta quedó "
        "marcada para revisión de un analista humano.",
        tool=tool,
        abstained=True,
        review=True,
    )


def ask_ticket_id() -> Answer:
    """Pide el identificador del ticket cuando la consulta no lo trae."""
    return _answer(ASK_TICKET_ID_MESSAGE, tool="get_ticket_status")


def ask_ticket_text(tool: str) -> Answer:
    """Pide más detalle cuando el texto es demasiado corto para clasificar o crear."""
    return _answer(ASK_TICKET_TEXT_MESSAGE, tool=tool)


def denied(tool: str, reason: str | None) -> Answer:
    """Denegación por permisos (RBAC) de una herramienta."""
    detail = f" ({reason})" if reason else ""
    return _answer(
        f"No tienes permiso para realizar esta acción{detail}. Si la necesitas, solicítala "
        "a un analista con el rol adecuado.",
        tool=tool,
        abstained=True,
    )


def limit_reached(tool: str, reason: str | None) -> Answer:
    """Denegación por el límite de uso de la sesión (política anti-abuso, no una falla)."""
    detail = f" {reason}" if reason else ""
    return _answer(
        f"No puedo continuar con esta solicitud.{detail} Si necesitas más, pide a un "
        "analista que lo gestione.",
        tool=tool,
        abstained=True,
    )


def ticket_status(data: dict[str, Any]) -> Answer:
    """Redacta el estado de un ticket visible, o la respuesta genérica si no lo es."""
    ticket_id = data.get("ticket_id", "solicitado")
    if not data.get("visible"):
        # Denegación sin enumeración: el mismo texto si no existe o si es de otra cuenta.
        return _answer(
            f"No encontré el ticket {ticket_id} dentro de tu alcance de acceso. Verifica el "
            "identificador o pide a un analista con acceso que lo consulte.",
            tool="get_ticket_status",
            abstained=True,
        )
    status = STATUS_LABELS.get(str(data.get("status")), str(data.get("status")))
    parts = [
        f"El ticket {ticket_id} está en estado «{status}» (prioridad {data.get('priority')}, "
        f"categoría {data.get('category')}, módulo {data.get('module')}, "
        f"Odoo {data.get('erp_version')}).",
        f"Última actualización: {data.get('updated_at')}.",
    ]
    if data.get("sla_due_at"):
        if data.get("sla_breached"):
            sla = "vencido"
        elif data.get("status") in {"resuelto", "cerrado"}:
            sla = "no aplica (ticket finalizado)"
        else:
            sla = "en plazo"
        parts.append(f"SLA: vence {data.get('sla_due_at')} — {sla}.")
    if data.get("summary"):
        parts.append(f"Resumen: {data['summary']}")
    if data.get("needs_human_review"):
        parts.append("El ticket está marcado para revisión humana.")
    return _answer(
        " ".join(parts),
        tool="get_ticket_status",
        review=bool(data.get("needs_human_review")),
    )


def classification(data: dict[str, Any]) -> Answer:
    """Redacta la clasificación sugerida y advierte si la confianza es baja."""
    review = bool(data.get("needs_human_review"))
    text = (
        f"Clasificación sugerida: categoría «{data.get('category')}» "
        f"(confianza {float(data.get('category_confidence', 0.0)):.2f}), prioridad "
        f"«{data.get('priority')}» y módulo «{data.get('module')}»."
    )
    if review:
        text += " La confianza es baja: un analista debe revisar la clasificación."
    return _answer(text, tool="classify_ticket", review=review)


_ACCOUNT_ISSUES: dict[str, str] = {
    "account_not_found": "la cuenta indicada no existe",
    "account_out_of_scope": "la cuenta indicada no está dentro de tu alcance",
    "account_required": "indica la cuenta del cliente con el formato ACC-0000",
}


def ticket_proposal(data: dict[str, Any], *, review: bool) -> Answer:
    """Redacta el resultado de proponer un ticket (borrador, duplicado o falta de datos)."""
    if data.get("identity_verified") is False:
        return _answer(
            "No pude verificar tu identidad en el sistema de tickets; no se preparó ningún "
            "borrador.",
            tool="create_ticket",
            abstained=True,
            review=True,
        )
    if data.get("duplicate_of"):
        return _answer(
            f"Ese ticket ya fue registrado como {data['duplicate_of']}; no se creó un duplicado.",
            tool="create_ticket",
        )
    issue = data.get("account_issue")
    if issue:
        return _answer(
            f"No preparé el borrador porque {_ACCOUNT_ISSUES.get(str(issue), str(issue))}.",
            tool="create_ticket",
        )
    text = (
        f"Preparé un borrador de ticket (categoría «{data.get('category')}», prioridad "
        f"«{data.get('priority')}», módulo «{data.get('module')}», cuenta "
        f"{data.get('account_id')}). Todavía NO se ha registrado: confírmalo con el token de "
        f"confirmación antes de {data.get('expires_at')} (un solo uso)."
    )
    if review:
        text += " El borrador quedó marcado para revisión humana."
    return _answer(text, tool="create_ticket", review=review)


def ticket_committed(data: dict[str, Any], *, review: bool) -> Answer:
    """Redacta la confirmación de un ticket escrito (o ya existente por idempotencia)."""
    if data.get("duplicate"):
        text = (
            f"El ticket ya estaba registrado como {data.get('ticket_id')}; la confirmación "
            "no creó un duplicado."
        )
    else:
        text = (
            f"Ticket {data.get('ticket_id')} registrado con estado «nuevo» (prioridad "
            f"{data.get('priority')}, categoría {data.get('category')})."
        )
    if review:
        text += " Quedó marcado para revisión humana."
    return _answer(text, tool="create_ticket", review=review)


def confirmation_rejected(reason: str) -> Answer:
    """Rechazo de una confirmación HITL (token inválido, expirado, reutilizado o ajeno)."""
    return _answer(f"No se registró ningún ticket: {reason}", tool="create_ticket", abstained=True)

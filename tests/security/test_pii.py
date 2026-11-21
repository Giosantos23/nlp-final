"""Enmascaramiento de PII: validadores (Luhn, mod-97, DPI) y ausencia de falsos positivos."""

from __future__ import annotations

import pytest

from docuerp.security.pii import (
    PII_TOKENS,
    contains_pii,
    dpi_valid,
    find_pii,
    iban_valid,
    luhn_valid,
    mask_pii,
)

# --- validadores ------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    ["4111 1111 1111 1111", "4111-1111-1111-1111", "5500000000000004", "378282246310005"],
)
def test_luhn_accepts_valid_cards(number: str) -> None:
    assert luhn_valid(number)


@pytest.mark.parametrize(
    "number",
    ["4111 1111 1111 1112", "1234567812345678", "411111111111", "4111x11111111111", ""],
)
def test_luhn_rejects_invalid_cards(number: str) -> None:
    assert not luhn_valid(number)


@pytest.mark.parametrize(
    "iban",
    ["GT82TRAJ01020000001210029690", "DE89 3704 0044 0532 0130 00", "GB82WEST12345698765432"],
)
def test_iban_mod97_accepts_valid(iban: str) -> None:
    assert iban_valid(iban)


@pytest.mark.parametrize(
    "iban", ["GB82WEST12345698765433", "GT00TRAJ01020000001210029690", "XX12", "1234567890123456"]
)
def test_iban_mod97_rejects_invalid(iban: str) -> None:
    assert not iban_valid(iban)


def test_dpi_validates_department_and_municipality() -> None:
    assert dpi_valid("2456 78901 0101")  # depto 01, municipio 01
    assert dpi_valid("1234567892201")  # depto 22
    assert not dpi_valid("1234567892301")  # depto 23 no existe
    assert not dpi_valid("1234567890100")  # municipio 00
    assert not dpi_valid("123456789010")  # 12 dígitos


# --- detección y enmascaramiento ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Escríbele a ana.lopez@volcanes.com hoy", "email"),
        ("Mi correo es Juan.Perez+erp@gmail.co", "email"),
        ("Llámame al +502 2345-6789", "phone"),
        ("Tel. +502 23456789", "phone"),
        ("(502) 2345 6789 es la oficina", "phone"),
        ("Celular 5555-1234", "phone"),
        ("Soporte en EE. UU.: +1 415 555 2671", "phone"),
        ("Mi DPI es 2456 78901 0101", "dpi"),
        ("DPI 2456789010101", "dpi"),
        ("NIT 1234567-8", "nit"),
        ("El NIT: 12345678 está activo", "nit"),
        ("Facturar al 123456-K", "nit"),
        ("Pago con 4111 1111 1111 1111", "card"),
        ("IBAN GT82TRAJ01020000001210029690", "iban"),
        ("Cuenta DE89 3704 0044 0532 0130 00", "iban"),
        ("El servidor 192.168.1.10 no responde", "ip"),
        ("IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334", "ip"),
    ],
)
def test_detects_each_pii_type(text: str, kind: str) -> None:
    masked, kinds = mask_pii(text)
    assert kinds == [kind]
    assert PII_TOKENS[kind] in masked


@pytest.mark.parametrize(
    "text",
    [
        "Error ERR-INV-204 al validar el picking",
        "¿Cuál es el estado del ticket TCK-00042?",
        "Según el runbook RB-N2-003 v1.1",
        "Factura INV/2026/01879 por Q20,857.25 descuadrada",
        "El importe es Q 1,234,567.89 con IVA del 12 %",
        "Migración de 17.0 a 18.0 del módulo 17.0.1.0.0",
        "Programado para 2026-11-17 a las 10:30:00",
        "Modelo stock.picking, campo move_ids",
        "Hay 263 unidades y 4111 cajas en bodega",
        "Cuenta ACC-0001 del usuario USR-0101",
        "Orden de compra P00042 con 16 líneas",
        "Tarjeta inválida 4111 1111 1111 1112",
        "Número de serie 1234567899999 sin DPI",  # departamento 99: no es DPI
        "Versión 300.1.1.1 del conector",  # octeto fuera de rango: no es IP
    ],
)
def test_no_false_positives_on_domain_identifiers(text: str) -> None:
    assert mask_pii(text) == (text, [])
    assert not contains_pii(text)


def test_mixed_text_masks_everything_and_keeps_context() -> None:
    text = (
        "Cliente Ana (ana@volcanes.com, +502 2345-6789, NIT 1234567-8) reporta ERR-INV-204 "
        "en TCK-00042; tarjeta 4111 1111 1111 1111."
    )
    masked, kinds = mask_pii(text)
    assert kinds == ["email", "card", "nit", "phone"]
    assert "ERR-INV-204" in masked
    assert "TCK-00042" in masked
    assert "NIT [NIT]" in masked  # la palabra NIT se conserva, solo cambia el número
    for value in ("ana@volcanes.com", "2345-6789", "1234567-8", "4111"):
        assert value not in masked


def test_find_pii_returns_original_values_in_detection_order() -> None:
    found = find_pii("Correo a@b.co y DPI 2456 78901 0101")
    assert found == [("email", "a@b.co"), ("dpi", "2456 78901 0101")]


def test_masking_is_idempotent() -> None:
    once, _ = mask_pii("Llámame al +502 2345-6789 o a ana@volcanes.com")
    twice, kinds = mask_pii(once)
    assert twice == once
    assert kinds == []


def test_empty_text() -> None:
    assert mask_pii("") == ("", [])

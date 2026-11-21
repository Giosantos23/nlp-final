# Base de tickets sintéticos (`tickets.db`)

Base SQLite **ficticia y determinista** de Altamira Soluciones ERP. Se genera con:

```bash
uv run docuerp data seed-tickets          # 300 tickets, semilla 42
uv run docuerp data seed-tickets --n 500 --seed 7
```

`data/tickets/tickets.db` está en `.gitignore` (es reproducible); este README sí se versiona.
Todos los nombres, correos, teléfonos (+502), DPI (13 dígitos) y NIT son inventados; cerca del
14 % de las descripciones incluyen PII ficticia para probar el enmascarado.
El "ahora" de la simulación es fijo (`2026-11-16T17:00:00`, hora de Guatemala).

## Esquema

La fuente de verdad es `TICKETS_DDL` en `src/docuerp/data/tickets.py`. Las lecturas del agente
usan `sqlite3.connect("file:...?mode=ro", uri=True)` y SQL parametrizado; solo `create_ticket`
(confirmado por un humano) escribe, con otra conexión.

```sql
CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,          -- ACC-0001
    name         TEXT NOT NULL,             -- "Comercial Los Volcanes"
    erp_version  TEXT NOT NULL CHECK (erp_version IN ('17.0','18.0')),
    sla_tier     TEXT NOT NULL              -- basico | estandar | premium
);
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY CHECK (user_id GLOB 'USR-[0-9][0-9][0-9][0-9]'),
    display_name TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('portal_user','n1','n2','admin')),
    account_id   TEXT REFERENCES accounts(account_id)  -- solo portal_user
);
CREATE TABLE IF NOT EXISTS portfolios (      -- cartera de cuentas de cada N2
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
    PRIMARY KEY (user_id, account_id)
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    TEXT PRIMARY KEY CHECK (ticket_id GLOB 'TCK-[0-9][0-9][0-9][0-9][0-9]'),
    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
    reported_by  TEXT NOT NULL REFERENCES users(user_id),
    assigned_to  TEXT REFERENCES users(user_id),
    summary      TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL,
    priority     TEXT NOT NULL CHECK (priority IN ('baja','media','alta','critica')),
    module       TEXT NOT NULL,
    status       TEXT NOT NULL
                 CHECK (status IN ('nuevo','en_progreso','en_espera','resuelto','cerrado')),
    erp_version  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    sla_due_at   TEXT,
    needs_human_review INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT UNIQUE,
    -- Columnas agregadas por la rama de corpus (opcionales, siempre al final):
    resolved_at       TEXT,                                  -- NULL si sigue abierto
    first_response_at TEXT,                                  -- primera respuesta humana
    channel           TEXT NOT NULL DEFAULT 'portal'
                      CHECK (channel IN ('portal','correo','telefono','chat'))
);
CREATE INDEX IF NOT EXISTS idx_tickets_account ON tickets(account_id);
CREATE INDEX IF NOT EXISTS idx_tickets_assigned ON tickets(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tickets_reported ON tickets(reported_by);
-- Tablas agregadas por la rama de corpus:
CREATE TABLE IF NOT EXISTS sla_policies (    -- POL-SLA-001 §3
    priority            TEXT PRIMARY KEY CHECK (priority IN ('baja','media','alta','critica')),
    first_response_hours INTEGER NOT NULL,
    resolution_unit     TEXT NOT NULL CHECK (resolution_unit IN ('hours','business_days')),
    resolution_value    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_events (   -- historial de cada ticket
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES tickets(ticket_id),
    event_type  TEXT NOT NULL CHECK (event_type IN (
                    'creado','asignado','escalado','cambio_estado','cambio_prioridad',
                    'comentario','resuelto','cerrado')),
    actor_id    TEXT REFERENCES users(user_id),
    from_value  TEXT,
    to_value    TEXT,
    note        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ticket ON ticket_events(ticket_id);
```

Valores permitidos: estados `nuevo | en_progreso | en_espera | resuelto | cerrado`; prioridades
`baja | media | alta | critica`; categorías `configuracion | error_funcional | acceso_permisos |
rendimiento | migracion | capacitacion | integracion`; módulos `inventario | compras | ventas |
facturacion | accesos | actualizacion | general`.

## Cuentas cliente

| account_id | Nombre | Odoo | Nivel SLA |
|---|---|---|---|
| ACC-0001 | Comercial Los Volcanes | 18.0 | estandar |
| ACC-0002 | Distribuidora La Antigua | 17.0 | premium |
| ACC-0003 | Agroexportadora Pacaya | 18.0 | premium |
| ACC-0004 | Ferretería El Quetzal | 17.0 | basico |
| ACC-0005 | Textiles Atitlán | 18.0 | estandar |
| ACC-0006 | Café Tikal Exportaciones | 18.0 | premium |
| ACC-0007 | Transportes Petén | 17.0 | estandar |
| ACC-0008 | Industrias Chichicastenango | 17.0 | estandar |
| ACC-0009 | Constructora Los Cerros | 18.0 | basico |
| ACC-0010 | Muebles Xelajú | 17.0 | basico |
| ACC-0011 | Lácteos del Motagua | 18.0 | estandar |
| ACC-0012 | Farmacias Santa Lucía | 18.0 | premium |

## Usuarios

- **Portal** (`portal_user`): `USR-0001` a `USR-0012` (uno por cuenta, en el mismo orden) y
  `USR-0013` a `USR-0018` (segundo contacto de las cuentas 1 a 6).
- **N1**: `USR-0101` a `USR-0110`. **N2**: `USR-0201` a `USR-0205`.
  **Admin**: `USR-0301` y `USR-0302`.
- Usuarios demo fijos: `USR-0001` (portal, Comercial Los Volcanes), `USR-0101` (N1),
  `USR-0201` (N2) y `USR-0301` (admin).

Carteras de los N2 (tabla `portfolios`):

| N2 | Cuentas |
|---|---|
| USR-0201 | ACC-0001, ACC-0002, ACC-0003 |
| USR-0202 | ACC-0004, ACC-0005, ACC-0006 |
| USR-0203 | ACC-0007, ACC-0008 |
| USR-0204 | ACC-0009, ACC-0010 |
| USR-0205 | ACC-0011, ACC-0012 |

## Política de SLA (`sla_policies`, POL-SLA-001)

| Prioridad | Primera respuesta | Resolución |
|---|---|---|
| critica | 1 h | 8 horas |
| alta | 4 h | 24 horas |
| media | 8 h | 3 días hábiles |
| baja | 24 h | 5 días hábiles |

`sla_due_at` de cada ticket se calcula con esta tabla (horas corridas en crítica y alta; días
hábiles de lunes a viernes, sin feriados fijos, en media y baja).

## Tickets ancla

Números fijos con propietario y estado conocidos, útiles para demos y pruebas de RBAC:

| Ticket | Cuenta | Reportó | Asignado | Estado | Prioridad | Resumen |
|---|---|---|---|---|---|---|
| TCK-00007 | ACC-0001 | USR-0001 | USR-0101 | resuelto | media | Usuario del portal no puede restablecer su contraseña (ERR-ACC-140) |
| TCK-00042 | ACC-0001 | USR-0001 | USR-0101 | en_progreso | alta | Recepción de la orden P00418 llegó con 20 cajas menos (ERR-INV-204) |
| TCK-00100 | ACC-0005 | USR-0005 | USR-0102 | en_espera | media | Nueva serie y secuencia de facturas para sucursal Quetzaltenango |
| TCK-00175 | ACC-0002 | USR-0002 | USR-0103 | en_progreso | alta | Odoo tarda más de 40 segundos en abrir cualquier pantalla |
| TCK-00210 | ACC-0001 | USR-0001 | USR-0101 | nuevo | critica | Facturas FEL rechazadas por el certificador |
| TCK-00300 | ACC-0006 | USR-0006 | USR-0202 | en_progreso | alta | Reporte de valoración de inventario no termina de cargar (ERR-PERF-510) |

## Alcance por rol (SPEC §3)

```sql
-- portal_user: solo lo que reportó
SELECT * FROM tickets WHERE ticket_id = ? AND reported_by = ?;
-- n1: solo lo asignado a él
SELECT * FROM tickets WHERE ticket_id = ? AND assigned_to = ?;
-- n2: tickets de las cuentas de su cartera
SELECT t.* FROM tickets t JOIN portfolios p ON p.account_id = t.account_id
 WHERE t.ticket_id = ? AND p.user_id = ?;
-- admin: todos
SELECT * FROM tickets WHERE ticket_id = ?;
```

## Tablas agregadas por la rama de corpus

- `ticket_events`: historial (`creado`, `asignado`, `escalado`, `cambio_estado`,
  `cambio_prioridad`, `comentario`, `resuelto`, `cerrado`) con actor, valores previo/nuevo y nota.
- `sla_policies`: tiempos comprometidos por prioridad.
- Columnas nuevas en `tickets` (al final): `resolved_at`, `first_response_at`, `channel`
  (`portal | correo | telefono | chat`; las cuentas de nivel básico solo usan portal).

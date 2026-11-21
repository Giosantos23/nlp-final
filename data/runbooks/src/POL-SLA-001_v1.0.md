---
id: POL-SLA-001@v1.0
code: POL-SLA-001
title: Política de soporte y SLA
version: v1.0
status: vigente
access_level: N1+
erp_version: any
module: general
doc_type: policy
allowed_roles: [n1, n2, admin]
sensitivity: internal
date: 2026-04-14
update_frequency: anual
format: pdf
owner: Gerencia de Operaciones · Altamira Soluciones ERP
poisoned: false
---

# Política de soporte y SLA

## 1. Propósito y alcance

Esta política establece cómo **Altamira Soluciones ERP** presta soporte funcional de nivel N1 y N2 a los clientes que usan Odoo 17.0 y 18.0 en Guatemala, qué tiempos se comprometen según la prioridad de cada ticket, cómo se escala un caso y qué situaciones quedan fuera del soporte. Es de cumplimiento obligatorio para todo analista N1, N2 y administrador, y para cualquier herramienta automatizada que asista al personal.

## 2. Definiciones

- **Ticket:** registro de una solicitud con identificador `TCK-xxxxx`, creado por el cliente en el portal o por un analista a nombre del cliente.
- **Tiempo de primera respuesta:** tiempo entre la creación del ticket y la primera respuesta humana útil (no cuenta la confirmación automática).
- **Tiempo de resolución:** tiempo entre la creación del ticket y la solución confirmada o la entrega de una alternativa de trabajo aceptada por el cliente.
- **Día hábil:** lunes a viernes, excluidos los feriados de la sección 4.
- **Niveles de soporte:** N1 (primer contacto y runbooks N1), N2 (análisis funcional y configuración) y ADMIN (base de datos, seguridad y migraciones).

## 3. Prioridades y tiempos comprometidos

| Prioridad | Criterio | Primera respuesta | Resolución |
|---|---|---|---|
| Crítica | Sistema inaccesible o imposible facturar, validar despachos o cerrar el mes; afecta a toda la operación | 1 hora | 8 horas |
| Alta | Función principal degradada con alternativa parcial, o más de 5 usuarios afectados | 4 horas | 24 horas |
| Media | Error o consulta que afecta a pocos usuarios, con alternativa de trabajo | 8 horas | 3 días hábiles |
| Baja | Consulta, capacitación, mejora o configuración sin impacto en la operación | 24 horas | 5 días hábiles |

**Cómo corre el reloj.** Las horas de prioridad crítica y alta corren de forma continua mientras haya cobertura. Los días hábiles de prioridad media y baja se cuentan dentro del horario de soporte. El reloj se **pausa** cuando el ticket pasa a estado *en_espera* por falta de información del cliente, y se reanuda cuando el cliente responde.

**Criterio de prioridad.** La prioridad inicial la propone quien reporta, y el analista N1 la confirma o corrige con base en el impacto, no en la urgencia percibida. Cambiar la prioridad se registra como evento en el historial del ticket.

## 4. Horario de soporte y feriados

| Cobertura | Horario (hora de Guatemala, UTC−6) | Aplica a |
|---|---|---|
| Soporte estándar | Lunes a viernes de 8:00 a 17:00 | Todas las cuentas |
| Guardia extendida | Todos los días, 24 horas, solo para prioridad crítica | Cuentas con nivel *premium* |

Fuera de horario, los tickets de cuentas *básico* y *estándar* inician su reloj a las 8:00 del siguiente día hábil. Feriados sin cobertura estándar: 1 de enero, Jueves y Viernes Santo, 1 de mayo, 30 de junio, 15 de septiembre, 20 de octubre, 1 de noviembre, 24 de diciembre desde el mediodía y 25 de diciembre.

## 5. Niveles de servicio por cuenta

| Nivel | Canales | Guardia crítica | Contactos autorizados |
|---|---|---|---|
| Básico | Portal | No | Hasta 2 |
| Estándar | Portal y correo | No | Hasta 5 |
| Premium | Portal, correo y teléfono | Sí, 24 horas | Hasta 10 |

Los contactos autorizados son las personas del cliente que pueden solicitar cambios de acceso o ajustes de inventario. Todo cambio de contactos lo solicita un contacto autorizado por escrito.

## 6. Matriz de escalamiento

| Situación | Escala de → a | Plazo máximo |
|---|---|---|
| El N1 no resuelve en el 50 % del tiempo de resolución | N1 → N2 | Al cumplirse el 50 % |
| Requiere ajustes de inventario, conciliación o configuración | N1 → N2 | 30 minutos desde el diagnóstico |
| Requiere acceso a base de datos, grupos de seguridad o migración | N2 → ADMIN | 1 hora desde el diagnóstico |
| Prioridad crítica sin avance a la mitad del tiempo | N2 → ADMIN y Gerente de Operaciones | Al cumplirse el 50 % |
| Sospecha de acceso no autorizado | Cualquier nivel → ADMIN | Inmediato |
| Incumplimiento de SLA inminente (80 % del tiempo consumido) | Responsable actual → supervisor | Inmediato |

El escalamiento siempre incluye un resumen del caso, los pasos ya ejecutados y la evidencia. No se escala «en blanco».

## 7. Regla de confirmación humana para crear tickets

Ningún asistente, herramienta automatizada ni script crea un ticket por sí solo. La regla es:

1. La herramienta prepara un **borrador** del ticket (resumen, descripción, categoría, prioridad y módulo) y **no escribe** en la base de tickets.
2. Un analista **humano revisa y confirma** el borrador de forma explícita. La confirmación usa un código de un solo uso que **expira a los 10 minutos**.
3. Cada sesión puede crear **como máximo 3 tickets**. Una confirmación repetida del mismo borrador no genera duplicados.
4. Si quien solicita es un usuario de portal, el borrador queda marcado para **revisión humana** y lo gestiona un analista N1.
5. Las acciones de prioridad crítica requieren la confirmación de un analista N2 o superior.

## 8. Qué no cubre el soporte

- Desarrollos o personalizaciones a la medida (se cotizan como proyecto aparte).
- Fallos de infraestructura del cliente: red, servidores propios, impresoras y estaciones de trabajo.
- Capacitación de usuarios nuevos más allá de dos sesiones por trimestre y por cuenta.
- Módulos de terceros no instalados ni homologados por Altamira.
- Asesoría contable, fiscal o legal. Los impuestos y la facturación electrónica FEL se configuran, pero la interpretación fiscal es responsabilidad del contador del cliente.
- Solicitudes fuera del alcance contratado, o hechas por personas que no son contactos autorizados.
- Otros sistemas ERP o de nómina que no sean Odoo 17.0 o 18.0.

## 9. Medición y créditos por incumplimiento

El cumplimiento se mide mensualmente por prioridad. La meta es de 95 % de tickets dentro del tiempo de respuesta y 90 % dentro del tiempo de resolución. Si en un mes la cuenta *premium* recibe menos del 90 % en prioridad crítica, Altamira ofrece un crédito de **Q500** sobre la cuota del mes siguiente; para cuentas *estándar* el crédito es de **Q250**. Los créditos no se aplican por incumplimientos causados por falta de respuesta del cliente.

## 10. Control de versiones

| Versión | Fecha | Autor | Cambios |
|---|---|---|---|
| v1.0 | 2026-04-14 | Gerencia de Operaciones | Versión inicial: prioridades, tiempos, horario, escalamiento, confirmación humana y exclusiones |

Próxima revisión programada: abril de 2027.

---
id: RB-ADM-006@v2.0
code: RB-ADM-006
title: Migración de base de datos 17.0 a 18.0
version: v2.0
status: vigente
access_level: ADMIN
erp_version: any
module: actualizacion
doc_type: runbook
allowed_roles: [admin]
sensitivity: restricted
date: 2026-10-05
update_frequency: trimestral
format: pdf
owner: Administración de plataforma · Altamira Soluciones ERP
poisoned: false
---

# Migración de base de datos 17.0 a 18.0

## 1. Propósito y alcance

Este runbook, de **acceso restringido a administradores**, describe el procedimiento controlado para migrar la base de datos de un cliente de Altamira Soluciones ERP de **Odoo 17.0 a Odoo 18.0**. Incluye la evaluación previa, el checklist de preparación, las ventanas de mantenimiento, la ejecución, la validación funcional y el **plan de rollback**.

La versión v2.0 reemplaza el enfoque de v1.x (migración directa sin ensayo) por un esquema obligatorio de **dos ensayos en copia** antes de la ventana productiva.

> **Restricción:** este documento contiene procedimientos con acceso a base de datos y copias de respaldo. No se comparte con N1 ni N2 ni con el cliente. Los analistas N1 y N2 solo informan al cliente las fechas y el estado de la ventana.

## 2. Roles y responsabilidades

| Rol | Responsabilidad | Persona típica |
|---|---|---|
| Líder de migración | Coordina el plan, decide Go/No-Go | Administrador senior |
| Administrador de base de datos | Respaldos, copias, restauración, ejecución del proceso | Administrador |
| Analista funcional N2 | Pruebas de aceptación con el cliente | N2 de la cartera del cliente |
| Contacto del cliente | Aprueba la ventana y firma la aceptación | Gerente o jefe de TI del cliente |

## 3. Evaluación previa (4 a 6 semanas antes)

1. Inventario de módulos: instalados, de la comunidad, de terceros y personalizados. Todo módulo personalizado debe tener versión probada para 18.0; de lo contrario, ERR-MIG-701.
2. Medición del tamaño de la base (GB) y del filestore. Una base mayor a 50 GB requiere ventana extendida.
3. Revisión de datos: duplicados de contactos, productos sin categoría, asientos contables descuadrados y períodos abiertos antiguos.
4. Revisión de integraciones: facturación electrónica FEL con el certificador, correo, banca y comercio electrónico. Cada integración se prueba en el ensayo.
5. Presupuesto de tiempo: ver sección 5.

## 4. Checklist de preparación

| # | Verificación | Responsable | Criterio de aceptación |
|---|---|---|---|
| 1 | Respaldo completo de base y filestore | Administrador de BD | Archivo restaurado en servidor de pruebas sin errores |
| 2 | Copia de la base para ensayo | Administrador de BD | Copia accesible solo para el equipo de migración |
| 3 | Lista de módulos personalizados con versión 18.0 | Líder | 100 % compatibles o excluidos con aprobación del cliente |
| 4 | Usuarios de prueba por rol (ventas, compras, inventario, contabilidad) | N2 | Al menos 1 usuario por rol |
| 5 | Casos de prueba firmados por el cliente | N2 y cliente | Mínimo 25 casos críticos |
| 6 | Congelamiento de cambios de configuración | Líder | 72 horas antes del ensayo 2 |
| 7 | Comunicado de ventana enviado | N2 | 7 días antes de la ventana |

## 5. Ventanas de mantenimiento

| Tamaño de la base | Ensayo 1 | Ensayo 2 | Ventana productiva | Horario sugerido |
|---|---|---|---|---|
| Hasta 10 GB | 4 horas | 4 horas | 6 horas | Sábado de 20:00 a 02:00 |
| De 10 a 50 GB | 8 horas | 8 horas | 12 horas | Sábado de 18:00 a domingo 06:00 |
| Más de 50 GB | 16 horas | 16 horas | 24 horas | Fin de semana completo, con aprobación del gerente |

La ventana productiva se agenda **fuera de cierres contables y de campañas de ventas** del cliente. Cada ventana incluye una reserva de 25 % del tiempo para validación y rollback.

## 6. Procedimiento de migración

### 6.1 Ensayo en copia

1. Restaure la copia en el entorno de pruebas con Odoo 17.0 y confirme que abre.
2. Ejecute el proceso de actualización hacia 18.0, registrando hora de inicio y de fin de cada paso.
3. Registre cada error con su código: ERR-MIG-701 (módulo incompatible), ERR-MIG-730 (fallo de pre-migración de datos), ERR-MIG-740 (vista rota tras la actualización), ERR-MIG-750 (reporte o plantilla de correo con campos eliminados).
4. Corrija, repita el ensayo 1 hasta que no queden errores críticos y ejecute el **ensayo 2** completo, medido, con los casos de prueba del cliente.
5. El líder decide **Go/No-Go** con el resultado del ensayo 2: sin errores críticos, 100 % de casos críticos aprobados y tiempo total dentro de la ventana con la reserva de 25 %.

### 6.2 Ventana productiva

1. **T-60 min:** confirme con el contacto del cliente la hora de congelamiento. Notifique a usuarios que cierren sesión.
2. **T-30 min:** bloquee el acceso de usuarios, incluido el portal. Detenga acciones programadas y cualquier integración entrante.
3. **T-0:** respaldo final de base y filestore con suma de verificación (SHA-256) anotada en el acta.
4. Ejecute el proceso de migración con el mismo procedimiento validado en el ensayo 2. No improvise pasos.
5. Al terminar, ejecute la validación técnica: inicio de sesión, apertura de las aplicaciones Inventory, Purchase, Sales y Accounting, y reglas de acceso del portal.
6. Ejecute la validación funcional con el N2 y el contacto del cliente: los 25 casos críticos.
7. **Go-live o rollback** antes del límite de decisión (sección 8). Reabra el acceso y active las integraciones.

## 7. Validación posterior

| Plazo | Acción |
|---|---|
| Día 0 | Confirmación de acceso, casos críticos aprobados, acta firmada por el cliente |
| Día 1 | Monitoreo reforzado: tiempos de respuesta, errores en bitácora, facturación FEL |
| Día 3 | Revisión con el cliente; incidencias abiertas con prioridad alta |
| Día 7 | Cierre de la migración y actualización de la versión del cliente (de 17.0 a 18.0) en su ficha |

## 8. Plan de rollback

El rollback se ejecuta si ocurre cualquiera de los siguientes: error crítico sin solución en 60 minutos; imposibilidad de facturar o de validar entregas; fallo de integridad contable. **El límite de decisión es el 75 % del tiempo de la ventana** (por ejemplo, a las 4.5 horas en una ventana de 6 horas).

1. Mantenga bloqueado el acceso de usuarios.
2. Restaure el respaldo final de base y filestore tomado en T-0 y verifique la suma SHA-256.
3. Reinicie el servicio con Odoo 17.0 y valide inicio de sesión y una operación de cada aplicación.
4. Reactive acciones programadas e integraciones.
5. Notifique al cliente a través del N2 y reprograme una nueva ventana tras el análisis de causa raíz.
6. Documente el rollback como incidencia de prioridad alta (**ERR-MIG-799**).

## 9. Control de versiones

| Versión | Fecha | Autor | Cambios |
|---|---|---|---|
| v1.0 | 2025-11-20 | Administración | Versión inicial: migración directa con respaldo |
| v1.5 | 2026-03-02 | Administración | Incluye revisión de integraciones FEL |
| v2.0 | 2026-10-05 | Administración | Dos ensayos obligatorios en copia, ventanas por tamaño de base, límite de decisión de rollback al 75 % y códigos ERR-MIG |

Próxima revisión programada: enero de 2027.

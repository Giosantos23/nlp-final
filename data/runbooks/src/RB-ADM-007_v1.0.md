---
id: RB-ADM-007@v1.0
code: RB-ADM-007
title: Gestión de grupos de seguridad y reglas de registro
version: v1.0
status: vigente
access_level: ADMIN
erp_version: any
module: accesos
doc_type: runbook
allowed_roles: [admin]
sensitivity: restricted
date: 2026-09-29
update_frequency: semestral
format: docx
owner: Administración de plataforma · Altamira Soluciones ERP
poisoned: false
---

# Gestión de grupos de seguridad y reglas de registro

## 1. Propósito y alcance

Este runbook, de **acceso restringido a administradores**, define cómo se crean, modifican y auditan los **grupos de seguridad** y las **reglas de registro** en las bases Odoo 17.0 y 18.0 de los clientes de Altamira Soluciones ERP. Su objetivo es que cada usuario vea y edite únicamente lo que necesita para su trabajo.

Solo un administrador con el permiso **Administration: Access Rights** puede cambiar permisos. Los analistas N1 y N2 no modifican grupos: cuando un ticket requiere un cambio, lo escalan con el formato de la sección 6.

> **Peligro:** un error en los permisos puede dejar la base sin ningún administrador (*admin impotente*, ERR-SEC-820) o exponer datos de una compañía a otra. Todo cambio se prueba primero en copia.

## 2. Conceptos

| Concepto | Qué hace | Dónde se ve |
|---|---|---|
| Grupo | Conjunto de permisos de una aplicación asignado a varios usuarios | Settings app → Users & Companies → Groups |
| Derecho de acceso (ACL) | Permisos Read, Write, Create y Delete de un grupo sobre un modelo | Pestaña **Access Rights** del grupo |
| Regla de registro | Filtro por dominio que limita **qué registros** del modelo ve el grupo | Settings app → Technical → Security → Record Rules |
| Grupo heredado | Los usuarios del grupo reciben automáticamente los grupos listados | Pestaña **Inherited** del grupo |

Los menús **Groups** y **Technical** solo aparecen con el **modo desarrollador** activado (**Settings app → Developer Tools → Activate the developer mode**). Los derechos de acceso dicen *qué operaciones* se pueden hacer sobre un modelo; las reglas de registro dicen *sobre cuáles registros*. Se evalúan en conjunto: sin derecho de acceso no importa la regla.

## 3. Procedimiento: crear o modificar un grupo

1. Active el modo desarrollador y abra **Settings app → Users & Companies → Groups**.
2. Para crear, pulse **New**; para modificar, abra el grupo existente. Complete **Application** y **Name**.
3. Pestaña **Users:** agregue usuarios con **Add a line**. Los usuarios en negro tienen derechos administrativos y los de azul no.
4. Pestaña **Inherited:** liste los grupos que debe heredar. Un grupo de supervisor de inventario hereda el de usuario de inventario.
5. Pestaña **Menus** y **Views:** limite los menús y vistas visibles solo si es necesario.
6. Pestaña **Access Rights:** pulse **Add a line**, elija el **Model** y marque **Read**, **Write**, **Create** o **Delete** según lo mínimo requerido. Nombre el derecho con el patrón `acl_<grupo>_<modelo>`.
7. Guarde. Registre el cambio en el ticket y en la bitácora de cambios del cliente (sección 6).

Para un usuario individual, abra **Settings app → Users & Companies → Users**, pestaña **Access Rights**, y elija para cada aplicación el nivel: ninguno, **User: Own Documents Only**, **User: All Documents** o **Administrator**. Prefiera siempre el nivel más bajo.

## 4. Procedimiento: reglas de registro

1. Abra **Settings app → Technical → Security → Record Rules** y pulse **New**. Las reglas de un grupo también se ven en la pestaña **Record Rules** de su formulario.
2. Complete **Name**, **Object** (modelo) y los permisos que la regla afecta (**Apply for Read/Write/Create/Delete**).
3. En **Domain** escriba el filtro. Ejemplo de regla por compañía: `[('company_id', 'in', company_ids)]`. Ejemplo para vendedores que ven solo sus cotizaciones: `[('user_id', '=', user.id)]`.
4. En **Groups** asigne los grupos a los que aplica. Una regla **sin grupos es global** y se aplica a todos, incluido el administrador, lo que puede bloquear el acceso (ERR-SEC-830).
5. Guarde y pruebe con un usuario real con solo ese grupo (sección 5).

| Código | Síntoma | Causa típica |
|---|---|---|
| ERR-SEC-801 | Un usuario recibe «Acceso denegado» a registros que antes veía | Regla nueva con dominio demasiado estricto |
| ERR-SEC-810 | Usuario sin un menú esperado | Falta del grupo requerido o de su herencia |
| ERR-SEC-820 | Ningún usuario puede cambiar permisos | Se quitó **Access Rights** al último administrador |
| ERR-SEC-830 | Todos bloqueados en un modelo | Regla global con dominio incorrecto |

## 5. Pruebas y control de cambios

1. Todo cambio se prueba antes en una **copia** de la base. Si no existe copia, la ventana de cambio se agenda fuera del horario de operación del cliente (después de las 19:00).
2. Pruebe con tres perfiles: un usuario con el grupo, un usuario sin él y el portal. Verifique menús visibles, registros visibles y acciones permitidas.
3. Conserve al menos **dos** usuarios con **Administration: Access Rights** en la base para evitar ERR-SEC-820.
4. Documente en el ticket: grupo, modelo, dominio, usuarios afectados, resultado de las tres pruebas y fecha.
5. Reversión: si algo falla, quite la regla (marque **Active** en falso) o restaure la lista previa de usuarios del grupo. Nunca borre una regla de producción antes de archivarla durante 7 días.

## 6. Solicitudes desde N1 y N2, escalamiento y tiempos

Las solicitudes llegan por ticket de categoría `acceso_permisos`. El analista que escala incluye:

- Usuario afectado, cuenta y versión de Odoo (17.0 o 18.0).
- Permiso que necesita y justificación del negocio.
- Aprobación escrita del contacto autorizado del cliente.
- Código ERR-SEC-xxx, si aplica.

El administrador responde en 4 horas en prioridad alta y resuelve en 24 horas; en prioridad media, 8 horas y 3 días hábiles. Los cambios que otorguen permisos de **Administrator** requieren la aprobación del Gerente de Operaciones de Altamira, además de la del cliente.

## 7. Control de versiones

| Versión | Fecha | Autor | Cambios |
|---|---|---|---|
| v1.0 | 2026-09-29 | Administración | Versión inicial: grupos, derechos de acceso, reglas de registro, pruebas y códigos ERR-SEC |

Estado: vigente. Próxima revisión programada: marzo de 2027.

"""Plantillas de texto para los tickets sintéticos (resúmenes y descripciones en español).

Cada :class:`Template` describe un tipo de incidencia (categoría + módulo) con varias
redacciones de resumen y de descripción. Los marcadores ``{producto}``, ``{po}``,
``{monto}``… se rellenan en ``tickets.py`` con datos ficticios verosímiles de Guatemala
(montos en quetzales, DPI, NIT, teléfonos +502). Los códigos ``ERR-*`` son los mismos que
documentan los runbooks, de modo que un ticket y su runbook comparten identificadores.

Marcadores disponibles: ``{producto}``, ``{cantidad}``, ``{faltante}``, ``{po}``,
``{recepcion}``, ``{entrega}``, ``{so}``, ``{factura}``, ``{proveedor}``, ``{monto}``,
``{usuario}``, ``{almacen}``, ``{ubicacion}``, ``{lote}``, ``{modulo_custom}``,
``{minutos}``, ``{afectados}``, ``{segundos}``, ``{fecha}``, ``{erp}``, ``{err}``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    """Tipo de incidencia con sus redacciones alternativas."""

    category: str
    module: str
    summaries: tuple[str, ...]
    descriptions: tuple[str, ...]
    errors: tuple[str, ...] = ()
    erp_only: str | None = None  # "17.0" o "18.0" si la incidencia solo aplica a esa versión
    ambiguous: bool = False  # el texto mezcla categorías: el ticket queda para revisión humana
    weight: float = 1.0
    max_priority: str | None = (
        None  # tope de prioridad (p. ej. un pedido de catálogo no es crítico)
    )


TEMPLATES: tuple[Template, ...] = (
    # ------------------------------------------------------------------ configuración
    Template(
        "configuracion",
        "inventario",
        (
            "Configurar rutas de recepción en dos pasos para {almacen}",
            "Activar recepción en 2 pasos en el almacén {almacen}",
            "Solicitud de configuración de ubicaciones en {almacen}",
        ),
        (
            "Queremos que la mercadería pase primero por una zona de control de calidad antes de "
            "entrar al stock de {almacen}. Entiendo que se activa en la configuración de rutas "
            "de Inventario pero no estoy seguro de los pasos ni del impacto en las recepciones "
            "abiertas.",
            "Necesitamos separar la zona de recepción del stock disponible en {almacen}. Hoy "
            "todo entra directo y el bodeguero no alcanza a revisar. ¿Pueden apoyarnos con la "
            "configuración?",
        ),
    ),
    Template(
        "configuracion",
        "inventario",
        (
            "Cambiar unidad de medida de {producto}",
            "Unidades de medida incorrectas en {producto}",
        ),
        (
            "El producto {producto} se compra por caja pero se vende por unidad. Necesitamos "
            "configurar la conversión de unidades de medida para que el inventario cuadre. Hoy "
            "hay {cantidad} registradas y no sabemos en qué unidad.",
            "Al crear el producto {producto} quedó solo con la unidad 'Unidades'. Requerimos "
            "que la compra sea en cajas de 12 y la venta en unidades.",
        ),
    ),
    Template(
        "configuracion",
        "inventario",
        (
            "Activar seguimiento por lotes para {producto}",
            "Configurar números de lote y fechas de vencimiento en {producto}",
        ),
        (
            "Por requerimiento sanitario necesitamos controlar lote y fecha de vencimiento de "
            "{producto}. Ya tenemos existencias; queremos saber cómo activar el seguimiento sin "
            "perder el stock actual.",
            "Auditoría nos pidió trazabilidad por lote. Necesitamos activar el seguimiento en "
            "{producto} y asignar lotes a lo que ya está en {almacen}.",
        ),
    ),
    Template(
        "configuracion",
        "inventario",
        (
            "Regla de reabastecimiento mínimo/máximo para {producto}",
            "Crear reglas de reorden para productos de alta rotación",
        ),
        (
            "Queremos que el sistema genere solicitudes de compra automáticas cuando {producto} "
            "baje de {faltante} unidades. ¿Cómo se configura la regla de reabastecimiento?",
            "Se nos acaba el inventario antes de que compras se dé cuenta. Pedimos apoyo para "
            "configurar reglas de reorden con mínimo y máximo en los productos principales.",
        ),
    ),
    Template(
        "configuracion",
        "compras",
        (
            "Configurar aprobación de órdenes de compra mayores a {monto}",
            "Flujo de aprobación para compras de alto monto",
        ),
        (
            "Gerencia pidió que toda orden de compra mayor a {monto} requiera su aprobación "
            "antes de confirmarse. Necesitamos saber si Odoo lo permite y cómo dejarlo "
            "configurado.",
            "Hoy cualquier comprador confirma órdenes sin límite. Queremos un umbral de "
            "aprobación a partir de {monto}.",
        ),
    ),
    Template(
        "configuracion",
        "compras",
        (
            "Activar conciliación a tres vías en facturas de proveedor",
            "Configurar control de facturas por cantidades recibidas",
        ),
        (
            "Contabilidad quiere pagar solo lo que se recibió. Necesitamos activar el control "
            "de facturas por cantidades recibidas y la conciliación a tres vías en compras.",
            "Pagamos una factura de {proveedor} por mercadería que nunca llegó. Queremos "
            "configurar el control para que no se pueda pagar sin recepción.",
        ),
    ),
    Template(
        "configuracion",
        "compras",
        (
            "Plantilla de orden de compra con condiciones de pago",
            "Agregar términos de pago a las órdenes de compra de {proveedor}",
        ),
        (
            "Necesitamos que las órdenes a {proveedor} salgan con condiciones de pago a 30 días "
            "y el texto de entrega. Pedimos configurar una plantilla de orden.",
        ),
    ),
    Template(
        "configuracion",
        "ventas",
        (
            "Activar listas de precios para clientes mayoristas",
            "Configurar lista de precios en quetzales y dólares",
        ),
        (
            "Vendemos a mayoristas con precios distintos al público. Queremos activar las listas "
            "de precios y asignarlas por cliente. También necesitamos una en dólares para "
            "exportación.",
            "Los vendedores cambian el precio a mano en cada cotización. Solicitamos configurar "
            "listas de precios con descuentos por volumen.",
        ),
    ),
    Template(
        "configuracion",
        "ventas",
        (
            "Configurar descuentos por línea en cotizaciones",
            "Limitar descuentos máximos de los vendedores",
        ),
        (
            "Queremos habilitar el descuento por línea en las cotizaciones pero que nadie pueda "
            "dar más de 15 %. ¿Se puede configurar un tope?",
            "Los vendedores piden poder aplicar descuentos. Necesitamos activarlos y definir "
            "quién puede autorizar cada rango.",
        ),
    ),
    Template(
        "configuracion",
        "ventas",
        (
            "Cambiar política de facturación a lo entregado",
            "Facturar solo lo entregado en pedidos de venta",
        ),
        (
            "Hoy se factura al confirmar el pedido, pero contabilidad quiere facturar solo lo "
            "que se entregó. Necesitamos cambiar la política de facturación y aplicarla también "
            "a los productos existentes.",
        ),
    ),
    Template(
        "configuracion",
        "facturacion",
        (
            "Configurar IVA 12 % en productos nuevos",
            "Impuestos mal configurados en productos de {proveedor}",
        ),
        (
            "Los productos nuevos salen sin IVA en las facturas de cliente. Necesitamos dejar el "
            "IVA de 12 % como impuesto predeterminado de venta y revisar los productos ya "
            "creados.",
            "Se emitieron facturas con IVA en cero para {producto}. Por favor revisar la "
            "configuración de impuestos del producto y de la posición fiscal del cliente.",
        ),
    ),
    Template(
        "configuracion",
        "facturacion",
        (
            "Nueva serie y secuencia de facturas para sucursal Quetzaltenango",
            "Configurar secuencia de facturas por diario",
        ),
        (
            "Abrimos una sucursal y necesitamos una serie de facturación independiente. "
            "Solicito crear el diario y su secuencia de numeración.",
            "La numeración de facturas se mezcla entre sucursales. Necesitamos un diario por "
            "sucursal con su propia secuencia.",
        ),
    ),
    Template(
        "configuracion",
        "facturacion",
        (
            "Retenciones de ISR en facturas de proveedor",
            "Configurar impuesto de retención en compras a pequeños contribuyentes",
        ),
        (
            "Algunos proveedores son pequeños contribuyentes y debemos aplicar retención. "
            "Necesitamos configurar los impuestos y las posiciones fiscales para que se calculen "
            "solos en la factura de proveedor.",
        ),
    ),
    Template(
        "configuracion",
        "general",
        (
            "Configurar correo saliente de la base de datos",
            "Alta de nueva compañía en la misma base",
        ),
        (
            "Los correos de cotizaciones salen desde una dirección genérica y caen en spam. "
            "Necesitamos configurar el servidor de correo saliente con el dominio de la empresa.",
            "Constituimos una nueva sociedad y queremos manejarla en la misma base de datos. "
            "¿Cómo se configura una segunda compañía sin mezclar la contabilidad?",
        ),
    ),
    # ------------------------------------------------------------------ error funcional
    Template(
        "error_funcional",
        "inventario",
        (
            "Recepción {recepcion} llegó con diferencia de cantidad ({err})",
            "Orden {po}: se recibieron {faltante} unidades menos de {producto}",
            "Diferencia entre lo ordenado y lo recibido en {po}",
        ),
        (
            "Al validar la recepción {recepcion} de la orden {po} la bodega contó {faltante} "
            "unidades menos de {producto} de lo que dice la orden. Odoo muestra el error {err} "
            "y no sabemos si crear orden pendiente. El proveedor {proveedor} dice que enviará "
            "el resto.",
            "{usuario} recibió {producto} y llegaron {faltante} menos de las {cantidad} "
            "ordenadas ({po}). No sabemos cómo dejar la recepción para que no se pierda el "
            "saldo.",
        ),
        ("ERR-INV-204",),
    ),
    Template(
        "error_funcional",
        "inventario",
        (
            "Producto {producto} con cantidad negativa en {ubicacion}",
            "Inventario en negativo de {producto} ({err})",
        ),
        (
            "Desde ayer {producto} aparece con cantidad negativa en {ubicacion}. Hicimos "
            "entregas antes de registrar la recepción. Al intentar ajustar sale el error {err}.",
            "El reporte de existencias muestra -{faltante} unidades de {producto} en "
            "{almacen}. Necesitamos corregirlo antes del cierre y no entendemos de dónde viene.",
        ),
        ("ERR-INV-110",),
    ),
    Template(
        "error_funcional",
        "inventario",
        (
            "No se puede validar la entrega {entrega}: reserva bloqueada ({err})",
            "Entrega {entrega} atascada en 'Esperando disponibilidad'",
        ),
        (
            "La entrega {entrega} no se deja validar aunque físicamente hay existencias de "
            "{producto}. Aparece {err}. Hay {afectados} pedidos de clientes esperando.",
            "Hay stock en {almacen} pero la entrega {entrega} sigue en espera. Parece que la "
            "mercadería quedó reservada en otra operación.",
        ),
        ("ERR-INV-415", "ERR-INV-430"),
    ),
    Template(
        "error_funcional",
        "inventario",
        (
            "Valoración de inventario no coincide con contabilidad en {producto}",
            "Diferencia de {monto} entre inventario y balance",
        ),
        (
            "El reporte de valoración de {producto} da un valor distinto al de la cuenta de "
            "inventario del balance, con una diferencia de {monto}. Contador pide explicación "
            "antes del cierre mensual.",
            "Revisando el cierre vemos {monto} de diferencia entre el valor del stock y la "
            "contabilidad. Sospechamos de un costo mal capturado en {producto}.",
        ),
    ),
    Template(
        "error_funcional",
        "inventario",
        (
            "Ajuste de inventario de {producto} requiere aprobación",
            "Conteo físico: diferencia de {faltante} unidades en {producto}",
        ),
        (
            "Hicimos conteo físico en {almacen} y {producto} tiene una diferencia de {faltante} "
            "unidades equivalentes a {monto}. Necesitamos aplicar el ajuste y saber quién debe "
            "aprobarlo.",
            "Después del inventario cíclico quedaron diferencias en {ubicacion}. El valor total "
            "es de {monto}. ¿Cuál es el procedimiento para ajustarlo?",
        ),
        ("ERR-INV-301",),
    ),
    Template(
        "error_funcional",
        "compras",
        (
            "Factura de {proveedor} no coincide con la recepción ({err})",
            "No podemos registrar la factura {factura}: excede lo recibido",
        ),
        (
            "La factura {factura} de {proveedor} cobra {cantidad} unidades de {producto} pero "
            "solo se recibieron {faltante}. Odoo marca la factura como excepción ({err}) y "
            "contabilidad no sabe si pagar.",
            "Al crear la factura desde la orden {po} el sistema muestra una excepción de pago. "
            "La diferencia es de {monto}.",
        ),
        ("ERR-PUR-310", "ERR-PUR-322"),
    ),
    Template(
        "error_funcional",
        "compras",
        (
            "Factura duplicada de {proveedor}",
            "Se registró dos veces la factura {factura} ({err})",
        ),
        (
            "Detectamos que la factura {factura} de {proveedor} por {monto} está registrada dos "
            "veces. Una ya se pagó. Necesitamos ayuda para corregir sin descuadrar la "
            "contabilidad.",
        ),
        ("ERR-PUR-330",),
    ),
    Template(
        "error_funcional",
        "compras",
        (
            "No se puede confirmar la orden de compra {po}",
            "Error al crear orden de compra a {proveedor}",
        ),
        (
            "Al confirmar la orden {po} aparece un mensaje de error y la orden se queda en "
            "borrador. Ya intentamos con otro usuario con el mismo resultado.",
            "El botón Confirmar de la orden {po} no hace nada. Se necesita con urgencia porque "
            "el proveedor {proveedor} despacha mañana.",
        ),
    ),
    Template(
        "error_funcional",
        "compras",
        (
            "IVA incorrecto en factura de proveedor {factura}",
            "Diferencia de impuesto en factura de compra ({err})",
        ),
        (
            "La factura {factura} de {proveedor} tiene una diferencia de impuesto con el total "
            "del documento físico. Odoo calculó el IVA distinto por {monto}.",
        ),
        ("ERR-PUR-340",),
    ),
    Template(
        "error_funcional",
        "ventas",
        (
            "La cotización {so} no se puede confirmar",
            "Pedido de venta {so} no genera la entrega",
        ),
        (
            "Al confirmar la cotización {so} para {usuario} no se crea la entrega aunque los "
            "productos son almacenables. Necesitamos despachar hoy.",
            "Confirmamos el pedido {so} y no aparece el botón de entrega. Los productos "
            "tienen existencias en {almacen}.",
        ),
    ),
    Template(
        "error_funcional",
        "ventas",
        (
            "Precio incorrecto en cotización {so}",
            "Lista de precios no se aplica en {so}",
        ),
        (
            "La cotización {so} muestra el precio público de {producto} en lugar del precio "
            "mayorista del cliente. Verificamos que el cliente tiene lista de precios asignada.",
        ),
    ),
    Template(
        "error_funcional",
        "ventas",
        (
            "Descuento no se refleja en la factura del pedido {so}",
            "Total del pedido {so} diferente al de la factura",
        ),
        (
            "El pedido {so} tenía un descuento acordado, pero la factura sale por {monto} sin "
            "descuento. El cliente reclama.",
        ),
    ),
    Template(
        "error_funcional",
        "facturacion",
        (
            "Asiento contable descuadrado al publicar factura {factura}",
            "No se puede publicar la factura {factura}",
        ),
        (
            "Al intentar publicar la factura {factura} por {monto} Odoo informa que el asiento "
            "no está balanceado. Es la tercera factura con este problema esta semana.",
            "La factura {factura} se queda en borrador y no permite confirmar. Contabilidad "
            "necesita emitirla hoy para el cierre.",
        ),
    ),
    Template(
        "error_funcional",
        "facturacion",
        (
            "Nota de crédito no reduce el saldo del cliente {usuario}",
            "Error al aplicar nota de crédito sobre {factura}",
        ),
        (
            "Emitimos una nota de crédito por {monto} contra la factura {factura} pero el saldo "
            "pendiente del cliente no cambió. Necesitamos conciliarla.",
        ),
    ),
    Template(
        "error_funcional",
        "facturacion",
        (
            "Factura de cliente con IVA en cero",
            "Impuesto no se calcula en la factura {factura}",
        ),
        (
            "La factura {factura} salió sin IVA siendo un cliente gravado. Ya se la enviamos al "
            "cliente. ¿Cómo se corrige sin romper la numeración?",
        ),
    ),
    Template(
        "error_funcional",
        "facturacion",
        (
            "Pago registrado no concilia con la factura {factura}",
            "Pago duplicado de {monto} en cuenta de cliente",
        ),
        (
            "Registramos el pago de {monto} y la factura {factura} sigue como pendiente. El "
            "banco ya lo acreditó. Necesitamos conciliarlo antes del cierre.",
        ),
    ),
    # ------------------------------------------------------------------ acceso y permisos
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Usuario del portal no puede restablecer su contraseña ({err})",
            "Contacto {usuario} no puede ingresar al portal",
            "Olvidé mi contraseña del portal de clientes",
        ),
        (
            "{usuario} no logra entrar al portal. Pidió restablecer la contraseña pero el "
            "enlace del correo dice que ya no es válido ({err}). Necesita revisar sus facturas "
            "hoy.",
            "No puedo ingresar al portal desde ayer. Le doy a 'Restablecer contraseña' y no me "
            "llega ningún correo, ya revisé spam.",
        ),
        ("ERR-ACC-140", "ERR-ACC-160"),
        weight=2.0,
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Cuenta bloqueada tras varios intentos ({err})",
            "Usuario bloqueado, no puede iniciar sesión",
        ),
        (
            "{usuario} escribió mal la clave varias veces y ahora aparece 'cuenta bloqueada' "
            "({err}). Necesita volver a entrar para cerrar pedidos del día.",
            "Se bloqueó mi usuario después de unos intentos fallidos. ¿Pueden desbloquearlo "
            "o enviarme un enlace para cambiar la contraseña?",
        ),
        ("ERR-ACC-101",),
        weight=1.5,
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Dar acceso al portal a nuevo contacto {usuario}",
            "Solicitud de acceso al portal para {usuario}",
        ),
        (
            "Ingresó {usuario} a la empresa y necesita ver las facturas y pedidos en el portal. "
            "Pedimos otorgarle acceso con el correo corporativo.",
            "Por favor dar acceso al portal a tres personas del área de compras para que "
            "consulten órdenes y facturas.",
        ),
        ("ERR-ACC-170",),
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Cambio de correo de inicio de sesión de {usuario}",
            "El usuario no coincide con el nuevo correo corporativo",
        ),
        (
            "Cambiamos el dominio de correo de la empresa. {usuario} ahora tiene otro correo y "
            "con el anterior ya no puede entrar. Solicitamos actualizar su usuario.",
        ),
        ("ERR-ACC-150",),
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Usuario no ve el menú de Compras",
            "Faltan menús para {usuario} en la aplicación",
            "Acceso denegado al abrir órdenes de compra ({err})",
        ),
        (
            "{usuario} ya no ve la aplicación de Compras y antes sí. No cambiamos nada de "
            "nuestro lado. Sale 'acceso denegado' al abrir una orden ({err}).",
            "Un vendedor nuevo no ve la aplicación de Ventas completa. Necesitamos revisar sus "
            "permisos para que vea sus cotizaciones y las de su equipo.",
        ),
        ("ERR-ACC-205", "ERR-SEC-801", "ERR-SEC-810"),
        weight=1.5,
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Quitar acceso a usuario que salió de la empresa",
            "Revocar permisos de {usuario}",
        ),
        (
            "{usuario} dejó de trabajar con nosotros el {fecha}. Solicitamos archivar su "
            "usuario y revocar cualquier acceso al portal de inmediato.",
        ),
    ),
    Template(
        "acceso_permisos",
        "accesos",
        (
            "Activar verificación en dos pasos para administradores",
            "Configurar 2FA en los usuarios internos",
        ),
        (
            "Por política interna de seguridad necesitamos activar la autenticación de dos "
            "factores para los usuarios internos. ¿Cuál es el procedimiento?",
        ),
    ),
    Template(
        "acceso_permisos",
        "inventario",
        (
            "Bodeguero no puede validar recepciones: sin permiso",
            "Permisos de inventario insuficientes para {usuario}",
        ),
        (
            "{usuario} es el responsable de bodega pero no le aparece el botón de validar en las "
            "recepciones. Requiere permiso de usuario de inventario.",
        ),
        ("ERR-ACC-205",),
    ),
    Template(
        "acceso_permisos",
        "ventas",
        (
            "Vendedores ven cotizaciones de otros equipos",
            "Restringir que cada vendedor vea solo sus cotizaciones",
        ),
        (
            "Hoy todos los vendedores ven las cotizaciones de todos. Queremos que cada uno vea "
            "solo las suyas y que el jefe de ventas vea todas.",
        ),
        ("ERR-SEC-801",),
    ),
    # ------------------------------------------------------------------ rendimiento
    Template(
        "rendimiento",
        "general",
        (
            "El sistema está muy lento en horario de operación",
            "Odoo tarda más de {segundos} segundos en abrir cualquier pantalla",
            "Lentitud general desde esta mañana",
        ),
        (
            "Desde las 8:00 el sistema responde con mucha lentitud. Abrir una cotización tarda "
            "alrededor de {segundos} segundos y somos {afectados} usuarios afectados. Pasa en "
            "todos los equipos de la oficina.",
            "Todas las pantallas están lentas, incluso contactos. Esto ya nos tiene "
            "afectados {afectados} personas y estamos atrasados con pedidos.",
        ),
        ("ERR-PERF-501",),
        weight=1.5,
    ),
    Template(
        "rendimiento",
        "inventario",
        (
            "Reporte de valoración de inventario no termina de cargar ({err})",
            "El reporte de movimientos se queda cargando",
        ),
        (
            "El reporte de valoración del inventario se queda cargando más de {minutos} minutos "
            "y el navegador marca que la página no responde. Necesitamos el reporte para el "
            "cierre del mes.",
            "Al abrir el historial de movimientos de un año completo la página nunca termina. "
            "Con un mes sí funciona.",
        ),
        ("ERR-PERF-510",),
    ),
    Template(
        "rendimiento",
        "inventario",
        (
            "Error de concurrencia al validar transferencias ({err})",
            "Bloqueos al validar entregas al mismo tiempo",
        ),
        (
            "Cuando dos bodegueros validan entregas a la vez aparece un error de acceso "
            "concurrente ({err}) y una de las dos no se guarda. Pasa varias veces al día.",
            "Al validar entregas simultáneas en {almacen} el sistema se bloquea unos minutos y "
            "luego muestra error de serialización.",
        ),
        ("ERR-PERF-520",),
    ),
    Template(
        "rendimiento",
        "inventario",
        (
            "Reservas de entregas tardan horas en asignarse",
            "Las reservas de stock se retrasan",
        ),
        (
            "Las entregas confirmadas por la mañana no reservan el stock hasta la tarde. "
            "Bodega no puede preparar pedidos a tiempo. Notamos que empeoró desde la importación "
            "de productos.",
        ),
        ("ERR-PERF-501",),
    ),
    Template(
        "rendimiento",
        "facturacion",
        (
            "Publicar facturas masivas tarda demasiado",
            "Cierre contable muy lento al publicar asientos",
        ),
        (
            "Al publicar más de {cantidad} facturas en lote el sistema se congela varios "
            "minutos. En el cierre de mes esto nos detiene a todo el equipo contable.",
        ),
    ),
    # ------------------------------------------------------------------ migración
    Template(
        "migracion",
        "actualizacion",
        (
            "Planificar migración de Odoo 17.0 a 18.0",
            "Solicitud de cotización y fecha para migrar a la versión 18.0",
        ),
        (
            "Queremos migrar nuestra base de la versión 17.0 a la 18.0. Necesitamos conocer el "
            "proceso, la ventana de mantenimiento y qué hacer con nuestro módulo personalizado "
            "{modulo_custom}.",
            "Estamos en 17.0 y la gerencia decidió pasar a 18.0 antes de fin de año. ¿Cuánto "
            "tiempo estaría fuera de línea el sistema y qué pruebas debemos preparar?",
        ),
        ("ERR-MIG-701",),
        erp_only="17.0",
    ),
    Template(
        "migracion",
        "actualizacion",
        (
            "Módulo personalizado {modulo_custom} incompatible con 18.0",
            "Prueba de migración falló por módulo propio ({err})",
        ),
        (
            "En la prueba de migración a 18.0 el módulo {modulo_custom} no carga ({err}). "
            "Necesitamos saber si se adapta o se reemplaza por funcionalidad estándar.",
        ),
        ("ERR-MIG-701",),
        erp_only="17.0",
    ),
    Template(
        "migracion",
        "actualizacion",
        (
            "Vista rota después de la migración a 18.0 ({err})",
            "Pantalla en blanco al abrir {producto} desde la actualización",
        ),
        (
            "Desde que migramos a 18.0 el fin de semana, la vista de productos aparece rota y "
            "falta una columna que usábamos todos los días ({err}).",
            "Después de la actualización a 18.0 la plantilla de correo de cotizaciones muestra "
            "campos vacíos. Antes funcionaba bien.",
        ),
        ("ERR-MIG-740", "ERR-MIG-750"),
        erp_only="18.0",
    ),
    Template(
        "migracion",
        "actualizacion",
        (
            "Datos faltantes tras la migración a 18.0",
            "Saldos de clientes distintos después de migrar",
        ),
        (
            "Terminada la migración notamos que algunos saldos de clientes no coinciden con los "
            "de antes de la ventana: diferencia total de {monto}. Pedimos revisar los datos.",
            "Faltan {afectados} contactos que existían antes de la migración. Necesitamos "
            "recuperarlos.",
        ),
        ("ERR-MIG-730",),
        erp_only="18.0",
    ),
    Template(
        "migracion",
        "actualizacion",
        (
            "Integración FEL dejó de funcionar al actualizar",
            "Plantillas de reporte rotas luego de la actualización",
        ),
        (
            "Tras la actualización el reporte de factura ya no imprime el número de "
            "autorización. El certificador sigue funcionando pero la representación gráfica "
            "sale incompleta.",
        ),
        ("ERR-MIG-750",),
        erp_only="18.0",
    ),
    Template(
        "migracion",
        "general",
        (
            "¿Qué cambia entre 17.0 y 18.0 para usuarios de inventario?",
            "Diferencias de menús entre versiones",
        ),
        (
            "Antes de decidir la migración queremos saber qué cambia para los usuarios de "
            "bodega y compras entre 17.0 y 18.0, para planear la capacitación.",
        ),
        erp_only="17.0",
    ),
    # ------------------------------------------------------------------ capacitación
    Template(
        "capacitacion",
        "ventas",
        (
            "Capacitación para nuevo equipo de ventas",
            "Solicitud de sesión sobre cotizaciones y facturación",
        ),
        (
            "Ingresaron {afectados} vendedores nuevos. Queremos una sesión de capacitación sobre "
            "cómo crear cotizaciones, aplicar listas de precios y facturar pedidos.",
            "Pedimos una capacitación corta (1 hora) para que el equipo comercial aprenda a usar "
            "descuentos y plantillas de cotización.",
        ),
        weight=1.5,
    ),
    Template(
        "capacitacion",
        "inventario",
        (
            "Capacitación sobre conteos físicos y ajustes de inventario",
            "Cómo hacer inventario cíclico: solicitud de capacitación",
        ),
        (
            "Vamos a iniciar inventarios cíclicos en {almacen} y los bodegueros no conocen el "
            "proceso. Solicitamos capacitación sobre conteos físicos y cómo aplicar ajustes.",
            "Queremos que el personal de bodega aprenda a registrar recepciones parciales y "
            "órdenes pendientes. ¿Pueden darnos una sesión?",
        ),
    ),
    Template(
        "capacitacion",
        "compras",
        (
            "Capacitación sobre flujo de compras y conciliación de facturas",
            "Dudas de uso: de la solicitud de cotización a la factura",
        ),
        (
            "El nuevo comprador necesita aprender el flujo completo: solicitud de cotización, "
            "orden de compra, recepción y factura. Pedimos una sesión de capacitación.",
        ),
    ),
    Template(
        "capacitacion",
        "facturacion",
        (
            "Capacitación en facturación y notas de crédito",
            "Cómo conciliar pagos: sesión para contabilidad",
        ),
        (
            "Contrataron a una auxiliar contable y necesita aprender a emitir facturas, notas de "
            "crédito y conciliar pagos en Odoo.",
        ),
    ),
    Template(
        "capacitacion",
        "general",
        (
            "Manual de usuario del sistema para personal nuevo",
            "Guía rápida de uso del portal de clientes",
        ),
        (
            "¿Tienen una guía o manual para personal nuevo? Queremos que cada área tenga "
            "material de consulta sobre las funciones principales.",
            "Nuestros clientes preguntan cómo consultar facturas en el portal. ¿Existe una guía "
            "que podamos compartirles?",
        ),
        weight=1.5,
    ),
    # ------------------------------------------------------------------ integración
    Template(
        "integracion",
        "facturacion",
        (
            "Facturas FEL rechazadas por el certificador",
            "Error al certificar factura {factura} en FEL",
        ),
        (
            "Desde esta mañana el certificador rechaza todas las facturas que emitimos. El "
            "mensaje indica NIT inválido del receptor. Es el cierre del día y no podemos "
            "facturar a {afectados} clientes.",
            "Al certificar la factura {factura} el servicio FEL devuelve un error de conexión. "
            "Reintentamos varias veces sin éxito.",
        ),
        weight=2.0,
    ),
    Template(
        "integracion",
        "facturacion",
        (
            "Conciliación bancaria automática con el banco",
            "Importar estados de cuenta del banco a Odoo",
        ),
        (
            "Queremos importar los estados de cuenta de nuestro banco para conciliar de forma "
            "automática. Hoy lo hacemos manual y toma dos días al mes.",
        ),
        max_priority="media",
    ),
    Template(
        "integracion",
        "facturacion",
        (
            "Sincronizar facturas con el sistema del contador externo",
            "Exportar asientos al sistema contable de la firma de auditoría",
        ),
        (
            "Nuestro contador externo trabaja en otro sistema y necesita recibir los asientos "
            "mensualmente. Pedimos una exportación o integración.",
        ),
        max_priority="media",
    ),
    Template(
        "integracion",
        "ventas",
        (
            "Conectar tienda en línea con Odoo",
            "Pedidos de la tienda web no llegan a Odoo",
        ),
        (
            "Los pedidos de nuestra tienda en línea no aparecen en Odoo desde el {fecha}. "
            "Revisamos la conexión y no vemos errores de nuestro lado.",
            "Queremos que los pedidos de la tienda virtual se creen como pedidos de venta de "
            "forma automática y descuenten inventario.",
        ),
        max_priority="alta",
    ),
    Template(
        "integracion",
        "ventas",
        (
            "Acceso por API para nuestro sistema de reparto",
            "Consulta de existencias vía API para aplicación móvil",
        ),
        (
            "Nuestro equipo de reparto usa una aplicación propia y necesita leer pedidos y "
            "existencias de Odoo por API. Solicitamos credenciales y documentación.",
        ),
        max_priority="media",
    ),
    Template(
        "integracion",
        "general",
        (
            "Correos de Odoo no llegan al cliente",
            "Sincronizar correo entrante con el chatter",
        ),
        (
            "Las cotizaciones enviadas desde Odoo no le llegan a algunos clientes (Gmail y "
            "Outlook). Revisamos la bandeja de spam sin éxito.",
            "Queremos que las respuestas de los clientes por correo se registren automáticamente "
            "en la cotización o el ticket correspondiente.",
        ),
        weight=1.5,
        max_priority="alta",
    ),
    Template(
        "integracion",
        "compras",
        (
            "Recibir órdenes de compra de proveedores por EDI",
            "Integrar catálogo de {proveedor} con compras",
        ),
        (
            "{proveedor} nos ofrece enviarnos su catálogo y precios actualizados en un archivo "
            "automático. ¿Se puede integrar con las listas de precios de compra?",
        ),
        max_priority="media",
    ),
    # ------------------------------------------------------------------ ambiguos (revisión humana)
    Template(
        "acceso_permisos",
        "actualizacion",
        (
            "No veo las opciones de inventario desde la actualización",
            "Después del cambio de versión faltan permisos",
        ),
        (
            "Desde que nos actualizaron el sistema ya no veo varias opciones de inventario. No "
            "sé si es un problema de permisos o si cambió el menú con la nueva versión.",
        ),
        ("ERR-ACC-205",),
        ambiguous=True,
        weight=0.8,
    ),
    Template(
        "error_funcional",
        "general",
        (
            "El sistema da error al guardar",
            "Algo no funciona en el sistema desde hoy",
        ),
        (
            "Hoy al guardar un documento aparece un error y no sé explicar bien qué pasó. "
            "Creo que fue después de que {usuario} cambiara algo en la configuración, pero no "
            "estoy seguro.",
            "Nos aparece un mensaje de error en varias pantallas y también está más lento de lo "
            "normal. No sé si es de permisos, de configuración o de rendimiento.",
        ),
        ambiguous=True,
        weight=0.8,
    ),
    Template(
        "configuracion",
        "general",
        (
            "Consulta sobre cómo ajustar el sistema a nuestro proceso",
            "Necesitamos adaptar el flujo, no sabemos por dónde empezar",
        ),
        (
            "Queremos cambiar la forma en que trabajamos pedidos, compras y bodega pero no "
            "tenemos claro si es configuración, un desarrollo o capacitación. Nos gustaría que "
            "alguien nos oriente.",
        ),
        ambiguous=True,
        weight=0.8,
    ),
)

URGENCY_PHRASES: dict[str, tuple[str, ...]] = {
    "critica": (
        "Es URGENTE: no podemos operar y tenemos clientes esperando.",
        "Sistema detenido para toda la operación, necesitamos ayuda inmediata.",
        "Crítico: no podemos facturar ni despachar hoy.",
    ),
    "alta": (
        "Es importante resolverlo hoy porque afecta al equipo.",
        "Nos está afectando varias áreas, agradecemos prioridad.",
        "Tenemos cierre contable esta semana y esto nos detiene.",
    ),
    "media": (
        "Podemos trabajar con una alternativa mientras tanto.",
        "Si pueden revisarlo esta semana estaría bien.",
        "",
    ),
    "baja": (
        "No es urgente, cuando tengan oportunidad.",
        "Sin prisa, es para planificar.",
        "",
    ),
}

OPENINGS: tuple[str, ...] = (
    "Buen día. ",
    "Buenas tardes, equipo de soporte. ",
    "Estimados, buen día. ",
    "Hola. ",
    "",
    "",
    "Saludos. ",
)

CLOSINGS: tuple[str, ...] = (
    " Gracias por su ayuda.",
    " Quedo atento.",
    " Adjunto captura de pantalla.",
    " Agradezco su pronta respuesta.",
    " Esto ya nos había pasado antes.",
    "",
    "",
)

COMMENT_NOTES: tuple[str, ...] = (
    "Se solicita información adicional al cliente.",
    "Se reproduce el caso en la base de pruebas.",
    "Se revisa el procedimiento del runbook correspondiente.",
    "Cliente confirma los datos solicitados.",
    "Se comparten pasos de la solución con el cliente.",
    "Se pide evidencia (captura de pantalla o guía de remisión).",
    "Se coordina llamada con el contacto autorizado.",
    "Seguimiento programado con el cliente.",
)

CLIENT_NOTES: tuple[str, ...] = (
    "Cliente aporta capturas de pantalla adicionales.",
    "Cliente confirma que el problema persiste.",
    "Cliente informa que probó la sugerencia sin éxito.",
    "Cliente solicita una actualización del estado.",
    "Cliente comparte el documento solicitado.",
)

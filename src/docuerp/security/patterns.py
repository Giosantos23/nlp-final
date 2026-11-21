"""Datos del detector de inyección: homoglifos, caracteres invisibles y patrones ES/EN.

Se separan de ``input_guard`` para que la lógica (normalizar → detectar → combinar) se lea
sin ruido y para que los patrones se puedan auditar y ampliar en un solo lugar.

Todos los patrones se evalúan sobre la **vista canónica** del texto (ver
``input_guard.normalize_for_inspection``): minúsculas, sin tildes, homoglifos mapeados a
latín y sin caracteres invisibles. Por eso las expresiones se escriben sin tildes
("instruccion", "contrasena") y en minúsculas, salvo los marcados ``cased=True``.

Criterio de pesos (se combinan con *noisy-OR*, umbral de bloqueo 0.70):

* ≥ 0.75 → señal inequívoca de ataque; bloquea sola (p. ej. "ignora las instrucciones
  anteriores", ``<|im_start|>``, ``UNION SELECT``).
* 0.40–0.65 → señal fuerte pero con usos legítimos posibles (p. ej. "soy administrador",
  "sin confirmación"); necesita otra señal para bloquear.
* ≤ 0.35 → indicio débil (p. ej. nombrar una herramienta interna); solo suma.

Para evitar falsos positivos en consultas de soporte se exigen *calificadores*: "omite
las reglas de reabastecimiento" es una pregunta funcional de Odoo, mientras que "omite
tus reglas" u "omite las reglas anteriores" es un intento de anular instrucciones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Homoglifos y caracteres invisibles
# ---------------------------------------------------------------------------

_CYRILLIC_LOWER = ("авеёкмнорстухѕіїјԁһԛԝɡӏ", "abeekmhopctyxsiijdhqwgl")
_CYRILLIC_UPPER = ("АВЕКМНОРСТХУЅІЈ", "ABEKMHOPCTXYSIJ")
_GREEK_LOWER = ("αβεικνορτυχωημ", "abeikvoptuxwnu")
_GREEK_UPPER = ("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ", "ABEZHIKMNOPTYX")

HOMOGLYPHS: dict[str, str] = {
    src: dst
    for sources, targets in (_CYRILLIC_LOWER, _CYRILLIC_UPPER, _GREEK_LOWER, _GREEK_UPPER)
    for src, dst in zip(sources, targets, strict=True)
}
"""Letra cirílica/griega → letra latina que imita visualmente (``о`` cirílica → ``o``)."""

INVISIBLE_CODEPOINTS: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),  # soft hyphen
    (0x034F, 0x034F),  # combining grapheme joiner
    (0x061C, 0x061C),  # arabic letter mark
    (0x115F, 0x1160),  # rellenos Hangul
    (0x17B4, 0x17B5),  # vocales invisibles jemer
    (0x180E, 0x180E),  # separador de vocales mongol
    (0x200B, 0x200F),  # zero-width space/joiners y marcas LRM/RLM
    (0x202A, 0x202E),  # controles de incrustación bidireccional
    (0x2060, 0x2064),  # word joiner y operadores invisibles
    (0x2066, 0x206F),  # aislamientos bidireccionales y formatos obsoletos
    (0x3164, 0x3164),  # relleno Hangul
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
    (0xFFA0, 0xFFA0),  # relleno Hangul de ancho medio
)
"""Rangos de puntos de código invisibles que se eliminan antes de inspeccionar."""

TAG_RANGE: tuple[int, int] = (0xE0000, 0xE007F)
"""Caracteres *tag* Unicode: codifican ASCII invisible ("ASCII smuggling")."""


def _char_class(ranges: tuple[tuple[int, int], ...]) -> str:
    """Construye una clase de caracteres regex a partir de rangos de puntos de código."""
    parts: list[str] = []
    for start, end in ranges:
        first, last = re.escape(chr(start)), re.escape(chr(end))
        parts.append(first if start == end else f"{first}-{last}")
    return "[" + "".join(parts) + "]"


INVISIBLE_RE = re.compile(_char_class(INVISIBLE_CODEPOINTS))
TAG_CHARS_RE = re.compile(_char_class((TAG_RANGE,)))

# ---------------------------------------------------------------------------
# Patrones de inyección
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionPattern:
    """Patrón de inyección: nombre, categoría, peso y expresión compilada."""

    name: str
    category: str
    weight: float
    regex: re.Pattern[str]
    cased: bool = False
    """Si es ``True`` se evalúa sobre el texto con mayúsculas (p. ej. ``DAN``)."""


def _p(
    name: str, category: str, weight: float, *parts: str, cased: bool = False
) -> InjectionPattern:
    """Declara un patrón uniendo ``parts`` (para respetar el ancho de línea)."""
    regex = re.compile("".join(parts), re.MULTILINE)
    return InjectionPattern(name, category, weight, regex, cased)


# Vocabulario reutilizado (texto ya en minúsculas y sin tildes). Solo imperativos: las
# preguntas legítimas usan primera persona o infinitivo ("¿cómo omito…?", "omitir").
ES_OVERRIDE_VERB = (
    r"(?:ignora|ignoren|ignore|olvida|olviden|olvide|olvidate de|omite|omitan|descarta|descarten|"
    r"desobedece|desestima|anula|salta(?:te)?|haz caso omiso (?:a|de)|no sigas|"
    r"no obedezcas|deja de (?:seguir|obedecer)|pasa por alto|no hagas caso (?:a|de)|"
    r"no tomes en cuenta|deja de lado)"
)
ES_RULES = (
    r"(?:instrucciones|instruccion|reglas|indicaciones|restricciones|politicas|directrices|"
    r"ordenes|normas|filtros|limitaciones|prompt|configuracion|lineamientos)"
)
ES_DEVELOPERS = r"(?:desarrolladores|creadores|programadores|disenadores)"
ES_RULES_QUALIFIER = (
    r"(?:anteriores|previas|previos|de arriba|del sistema|de sistema|originales|iniciales|"
    r"que (?:te|se te) (?:dieron|dio|han dado|pusieron|impusieron|ordenaron|indicaron|"
    r"programaron|escribieron)|"
    rf"de (?:tus|los) {ES_DEVELOPERS}|de seguridad|recibidas)"
)
ES_TELL_ME = r"(?:dijeron|indicaron|ordenaron|pidieron|programaron|pusieron|impusieron)"
"""Verbos con los que se alude a lo que "le dijeron" al asistente quienes lo configuraron."""
FUNCTIONAL_RULES = (
    r"(?:reabastecimiento|aprobacion|precios|descuento|impuestos|entrega|inventario|compra|"
    r"venta|facturacion|almacen|ruta|rutas|redondeo|tarifa)"
)
"""Reglas de negocio de Odoo: "omite las reglas de aprobación" es una consulta legítima."""
EN_OVERRIDE_VERB = r"(?:ignore|disregard|forget|override|bypass|skip|drop|discard|neglect)"
EN_RULES = (
    r"(?:instructions?|rules|prompts?|directions|guidelines|guardrails|restrictions|"
    r"policies|context)"
)
EN_QUALIFIER = (
    r"(?:all|any|previous|prior|above|earlier|your|the system|initial|original|preceding)"
)
REVEAL_ES = (
    r"(?:muestra(?:me)?|revela(?:me)?|imprime|dime|dame|repite|copia|escribe|ensena(?:me)?|"
    r"comparte|filtra|expon|lista|recita|transcribe|traduce|resume|"
    # Subjuntivo de cortesía: "necesito que me muestres / imprimas / reveles…".
    r"muestres|reveles|imprimas|digas|repitas|copies|escribas|ensenes|compartas|recites)"
)
REVEAL_EN = (
    r"(?:show|reveal|print|repeat|display|output|tell me|give me|leak|dump|recite|share|"
    r"write out)"
)
SENSITIVE_DATA = (
    r"(?:correos|emails?|e-mails?|telefonos|numeros de telefono|dpi|nits?|tarjetas|"
    r"contrasenas|passwords?|direcciones|datos personales|credenciales|credentials|"
    r"phone numbers)"
)
SECRETS = (
    r"(?:api[ _-]?keys?|clave (?:de )?(?:la )?api|tokens? (?:de acceso|oauth|secreto|de sesion)|"
    r"secret key|hmac|contrasena (?:de|del) (?:admin|administrador|la base|base de datos|root|"
    r"sistema|servidor)|password (?:of|for) (?:the )?(?:admin|database|root)|credenciales|"
    r"variables de entorno|environment variables|\.env\b|claude_code_oauth_token|"
    r"anthropic_api_key|docuerp_hmac_secret)"
)
PERSONAL_FIELD = (
    r"(?:correos?(?: electronicos?)?|e-?mails?|email address(?:es)?|direcciones? de correo|"
    r"telefonos?|numeros? de telefono|celulares|phone numbers?|phones|dpi|nits?|tarjetas?|"
    r"contrasenas?|passwords?|direcciones|addresses|datos personales|datos de contacto|"
    r"personal data|contact details|credenciales|credentials)"
)
"""Campo personal en singular o plural (para pedidos masivos "de cada cliente")."""
DATA_OWNERS = (
    r"(?:clientes?|usuarios?|contactos?|empleados?|proveedores?|customers?|users?|contacts?|"
    r"employees?|clients?|suppliers?)"
)
BULK_QUANTIFIER = r"(?:todos|todas|cada(?: uno de)?|all(?: the| of the)?|every|each)"
REQUEST_VERB = (
    r"(?:dame|danos|pasame|pasanos|lista(?:me)?|enumera|necesito|quiero|comparte(?:me)?|"
    r"compartenos|extrae|exporta|saca(?:me)?|muestra(?:me)?|ensename|envia(?:me)?|"
    r"manda(?:me)?|consigueme|recopila|vuelca|give me|list|export|send|share|i need|i want|"
    r"show me|get me|pull|dump|collect)"
)
"""Imperativos o deseo de entrega. "¿Cómo exporto…?" (primera persona) no coincide."""

INJECTION_PATTERNS: tuple[InjectionPattern, ...] = (
    # --- Anular instrucciones -------------------------------------------------------
    _p("override_es_qualified", "override", 0.85,
       rf"\b{ES_OVERRIDE_VERB}\b[^.\n]{{0,40}}?\b{ES_RULES}\b[^.\n]{{0,30}}?",
       rf"\b{ES_RULES_QUALIFIER}\b"),
    _p("override_es_possessive", "override", 0.85,
       rf"\b{ES_OVERRIDE_VERB}\b\s+(?:todas\s+)?(?:tus|tu|las|sus)?\s*(?:propias?\s+)?{ES_RULES}\b",
       rf"(?!\s+(?:de|del|para)\s+(?:la |las |los |el )?{FUNCTIONAL_RULES})"),
    _p("override_es_everything", "override", 0.75,
       rf"\b{ES_OVERRIDE_VERB}\b\s+(?:todo\s+)?lo\s+",
       rf"(?:anterior|de arriba|que te {ES_TELL_ME})"),
    # "Ignora el contexto": el contexto recuperado es la base de la respuesta fundamentada.
    _p("override_es_context", "override", 0.8,
       rf"\b{ES_OVERRIDE_VERB}\b\s+(?:todo\s+)?(?:el|los|la|las|tu|tus|ese|este|esos)?\s*",
       r"(?:contexto|fragmentos(?: recuperados)?|documentos recuperados|evidencia recuperada|",
       r"fuentes recuperadas)\b"),
    _p("override_en_context", "override", 0.8,
       r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:the |your |this |that |all )?",
       r"(?:retrieved |provided |given )?(?:context|documents|passages|chunks|evidence)\b"),
    # "Ya no hay reglas": anula las instrucciones negando que existan.
    _p("no_rules_exist", "override", 0.75,
       r"\b(?:ya no|ahora no|aqui no|para ti no|desde (?:ahora|este momento) no)\s+",
       r"(?:hay|existen|tienes|aplican|rigen|cuentan)\s+(?:mas\s+)?",
       r"(?:reglas|restricciones|limites|filtros|politicas|instrucciones|normas)\b",
       rf"(?!\s+(?:de|del|para|en)\s+(?:la |las |los |el |este |esta )?{FUNCTIONAL_RULES})|",
       r"\bno tienes (?:ninguna |ningun |mas )?(?:reglas|restricciones|limites|filtros|",
       r"politicas)\b|",
       # En inglés se exige el destinatario ("you have no rules") o un marcador temporal
       # ("there are no rules anymore"): la documentación de Odoo dice "there are no rules
       # applied to this product".
       r"\byou have no (?:more |longer any )?(?:rules|restrictions|limits|filters|guidelines|",
       r"policies)\b|",
       r"\bthere are no (?:more )?(?:rules|restrictions|limits|filters|guidelines|policies)",
       r"\s+(?:anymore|any more|now|for you|from now on)\b"),
    _p("override_en", "override", 0.85,
       rf"\b{EN_OVERRIDE_VERB}\b[^.\n]{{0,30}}?\b{EN_QUALIFIER}\b[^.\n]{{0,20}}?\b{EN_RULES}\b"),
    _p("rules_void_es", "override", 0.8,
       r"\b(?:instrucciones|reglas|indicaciones|politicas)\s+(?:anteriores|previas|originales|",
       r"del sistema)\s+(?:ya\s+)?(?:no (?:aplican|valen|cuentan|son validas)|",
       r"quedan (?:anuladas|sin efecto))"),
    _p("rules_void_en", "override", 0.8,
       r"\b(?:previous|prior|above|original|system) (?:instructions|rules) ",
       r"(?:are|were) (?:void|cancelled|canceled|no longer valid|obsolete)|",
       r"\b(?:instructions|rules) no longer apply\b"),
    _p("disobey_system", "override", 0.75,
       r"\b(?:deja de|no) (?:obedecer|seguir|obedezcas|sigas) (?:al|el|a tu) (?:sistema|prompt)\b|",
       r"\bobey only (?:this|me|my)\b|",
       r"\bstop following (?:the system|your rules|your instructions)\b|",
       r"\bstop (?:following|applying) (?:your (?:safety |security )?|the (?:safety|security) )",
       r"(?:rules|guidelines|policies|filters|guardrails)\b"),
    # Desanclar la respuesta de la evidencia: categoría propia para que se sume (noisy-OR)
    # a un intento de anular instrucciones en lugar de quedar absorbida por él.
    _p("drop_grounding", "grounding_bypass", 0.4,
       r"\b(?:responde|contesta) sin (?:citar|citas|fuentes|usar el contexto)|",
       r"\banswer without (?:citations|sources|the context)\b|",
       r"\b(?:no uses|sin usar|sin basarte en|no te bases en)\s+(?:el |los |la |las )?",
       r"(?:contexto|documentos|documentacion|fragmentos)\b"),
    _p("own_knowledge", "grounding_bypass", 0.4,
       r"\b(?:contesta|responde) con lo que (?:sepas|tu sepas|creas)\b|",
       r"\busa tu (?:propio )?conocimiento\b|",
       r"\b(?:from|with|using) your own (?:knowledge|training)\b"),
    _p("new_instructions", "override", 0.5,
       r"\b(?:nuevas instrucciones|instrucciones nuevas|new instructions|updated instructions|",
       r"instrucciones actualizadas|instruccion del sistema|nueva instruccion)\s*:"),
    # --- Revelar prompt / sistema ---------------------------------------------------
    _p("prompt_leak_es", "prompt_leak", 0.85,
       rf"\b{REVEAL_ES}\b[^.\n]{{0,25}}?\b(?:tu|tus|el|la|las|los|su)?\s*",
       r"(?:prompt|instrucciones|mensaje|configuracion|reglas|directrices)\s+(?:de |del )?",
       r"(?:sistema|iniciales|originales|ocultas|internas|secretas|previas|de arriba)\b"),
    _p("prompt_leak_es_your", "prompt_leak", 0.8,
       rf"\b{REVEAL_ES}\b[^.\n]{{0,15}}?\btus\s+(?:instrucciones|reglas|directrices|indicaciones)\b"),
    _p("prompt_leak_es_tu", "prompt_leak", 0.8,
       rf"\b{REVEAL_ES}\b[^.\n]{{0,15}}?\btu (?:prompt|configuracion|programacion)\b|",
       rf"\b{REVEAL_ES}\b[^.\n]{{0,40}}?\b(?:las )?(?:instrucciones|reglas) que ",
       r"(?:recibiste|te dieron|tienes|te configuraron)\b"),
    _p("how_programmed_es", "prompt_leak", 0.75,
       r"\b(?:dime|explica(?:me)?|revela(?:me)?|muestra(?:me)?|cuentame)\b[^.\n]{0,30}\b",
       r"como te (?:programaron|configuraron|instruyeron)\b"),
    _p("prompt_question_es", "prompt_leak", 0.75,
       r"\b(?:cuales|que) son tus (?:instrucciones|reglas|directrices|indicaciones)\b"),
    _p("prompt_leak_en", "prompt_leak", 0.85,
       rf"\b{REVEAL_EN}\b[^.\n]{{0,25}}?\b(?:your (?:system |initial |original |hidden )?",
       r"(?:prompt|instructions|rules|guidelines)|(?:system|initial|original|hidden|secret) ",
       r"(?:prompt|instructions|message))\b"),
    _p("prompt_question_en", "prompt_leak", 0.75,
       r"\bwhat (?:are|were) your (?:initial |original |system |hidden )?",
       r"(?:instructions|rules|guidelines)\b|\b(?:rules|instructions) you were given\b"),
    # Preguntar con qué se configuró al asistente es pedir el prompt con otras palabras.
    _p("configured_with", "prompt_leak", 0.75,
       r"\b(?:con )?(?:que|cuales) (?:instrucciones|reglas|indicaciones|directrices|prompt|",
       r"ordenes|politicas)\s+(?:te|se te) (?:dieron|dio|pusieron|configuraron|programaron|",
       r"indicaron|inicializaron|entrenaron|cargaron)\b|",
       r"\b(?:instrucciones|reglas|indicaciones|directrices|texto|prompt|mensaje)\s+con ",
       r"(?:las|los|el|la) que te (?:programaron|configuraron|inicializaron|crearon|entrenaron|",
       r"instruyeron)\b|",
       r"\bque te (?:indicaron|pidieron|dijeron|ordenaron|programaron) (?:que hicieras|hacer|",
       r"que respondieras|responder)\b|",
       r"\b(?:what|how) (?:were|have) you (?:been )?(?:configured|instructed|told|programmed|",
       r"set up|initialized|prompted)\b"),
    # Alusión a quienes configuraron al asistente ("tus desarrolladores"): señal media que
    # bloquea si se combina con un verbo de revelar o de anular.
    _p("developer_instructions", "prompt_leak", 0.6,
       rf"\btus {ES_DEVELOPERS}\b|",
       rf"\b(?:te|le) {ES_TELL_ME} (?:los|tus) {ES_DEVELOPERS}\b|",
       r"\byour (?:developers?|creators?|programmers?)\b|",
       r"\bdeveloper (?:instructions|message|prompt)\b|",
       r"\b(?:instructions|rules) (?:from|given by|set by) (?:your |the )?",
       r"(?:developers?|creators?)\b"),
    _p("reveal_developer", "prompt_leak", 0.85,
       rf"\b(?:{REVEAL_ES}|cuentame|pega|{REVEAL_EN}|paste|quote)\b[^.\n]{{0,40}}?",
       rf"(?:\btus {ES_DEVELOPERS}\b|\bte {ES_TELL_ME} (?:los|tus) {ES_DEVELOPERS}\b|",
       r"\bdeveloper (?:instructions|message|prompt)\b)"),
    _p("system_prompt_mention", "prompt_leak", 0.5,
       r"\b(?:system prompt|prompt de sistema|prompt del sistema|prompt inicial|",
       r"prompt original)\b"),
    _p("repeat_above", "prompt_leak", 0.7,
       r"\b(?:repite|repitas|copia|copies|imprime|imprimas|repeat|print)\b[^.\n]{0,25}",
       r"\b(?:todo lo anterior|",
       r"el texto (?:anterior|de arriba)|everything above|the text above|lo que esta arriba)\b"),
    _p("canary_probe", "prompt_leak", 0.9,
       r"docuerp-canary|\bcanary token\b|\btoken canario\b"),
    # --- Cambio de rol / jailbreak --------------------------------------------------
    _p("role_from_now", "role_change", 0.7,
       r"\b(?:a partir de ahora|desde ahora|de ahora en adelante|en adelante|desde este momento|",
       r"a partir de este momento|desde ya)\b[^.\n]{0,15}",
       r"\b(?:eres|seras|actuaras|responderas|contestaras|vas a ser|te llamas|tu nombre es|",
       r"no tienes|solo obedeces|obedeceras|ignoras|ignoraras|no respondes)\b"),
    _p("role_now_es", "role_change", 0.7,
       r"\b(?:ahora eres|eres ahora|ya no eres|tu nuevo rol|tu nueva identidad|",
       r"nuevo personaje|olvida que eres)\b"),
    _p("role_act_as_es", "role_change", 0.4,
       r"\b(?:actua|comportate|hazte pasar|finge|pretende|simula)\b\s+",
       r"(?:que eres|ser|como (?:si fueras|un|una|el|la|mi))\b"),
    # "Hazte pasar por…": adoptar otra identidad; en soporte nadie lo pide legítimamente al
    # asistente (las consultas sobre suplantar usuarios en Odoo van en primera persona).
    _p("role_impersonate_es", "role_change", 0.55,
       r"\b(?:hazte pasar por|haz de cuenta que eres|hagamos de cuenta que eres|finge ser|",
       r"pretende ser|simula ser|asume el (?:rol|papel) de|ponte en el papel de)\b"),
    _p("role_en", "role_change", 0.65,
       r"\b(?:you are now|you're now|from now on,? you|from this (?:moment|point)(?: on)?,? you|",
       r"pretend (?:to be|you are)|",
       r"roleplay as|role-play as|act as if|you will now act|stay in character)\b"),
    _p("role_act_as_en", "role_change", 0.4, r"\bact as (?:an?|the|my)\b"),
    _p("role_game_es", "role_change", 0.55,
       r"\b(?:juguemos|hagamos|vamos a jugar|imagina que)\b[^.\n]{0,15}\b(?:rol|roleplay|",
       r"juego de rol|eres)\b|\b(?:finge|pretende|simula|imagina) que eres\b"),
    _p("impersonate_admin", "escalation", 0.5,
       r"\b(?:como si fueras|como si fuera|as if you were|pretend (?:to be|you are|you're)|",
       r"hazte pasar por|finge ser|haz de cuenta que eres|pretende ser|simula ser) ",
       r"(?:el |la |un |una |the |an? )?(?:[a-z]+ )?",
       r"(?:admin|administrador|administrator|root|sysadmin|soporte n2|analista n2|",
       r"superusuario|superuser)\b"),
    _p("malicious_persona", "malicious_intent", 0.4,
       r"\b(?:hacker|atacante|cracker|black ?hat|malware|evil (?:assistant|ai|bot))\b"),
    _p("jailbreak_terms", "jailbreak", 0.8,
       r"\b(?:do anything now|modo dios|god mode|jailbreak|",
       r"jailbroken|modo sin censura|uncensored|unfiltered|sin censura|stan mode|evil mode|",
       r"unrestricted (?:ai|assistant|mode|model)|asistente sin restricciones|",
       r"ia sin restricciones|modo sin (?:filtros|restricciones|limites|reglas)|",
       r"modo malvado|aim mode)\b"),
    # El "modo desarrollador" es una función real de Odoo: la documentación dice "switch to
    # developer mode" y los usuarios preguntan "¿cómo activo el modo desarrollador?". Solo es
    # jailbreak si se afirma que el asistente está en ese modo o si va junto a sus
    # límites/reglas. Mencionarlo solo es un indicio débil.
    _p("developer_mode_mention", "jailbreak", 0.3, r"\b(?:developer mode|modo desarrollador)\b"),
    _p("developer_mode_jailbreak", "jailbreak", 0.8,
       r"\b(?:ahora estas|estas ahora|ya estas|you are now|you're now|now you are)\s+",
       r"(?:en |in )?(?:el |the )?(?:modo desarrollador|developer mode)\b|",
       r"\b(?:modo desarrollador|developer mode)\b[^.\n]{0,40}\b(?:desactiva|quita|ignora|sin|",
       r"ignore|disable|remove|without|no)\b[^.\n]{0,20}\b(?:tus |your |las |los |content )?",
       r"(?:limites|restricciones|filtros|reglas|politicas|censura|limits|restrictions|filters|",
       r"rules|policies|guardrails)\b"),
    _p("jailbreak_dan", "jailbreak", 0.8, r"\bDAN\b", cased=True),
    _p("grandma_exploit", "jailbreak", 0.6,
       r"\b(?:mi abuela|mi abuelita|my grandma|my grandmother)\b[^.\n]{0,80}\b",
       r"(?:contaba|leia|decia|recitaba|used to (?:tell|read|say))\b"),
    _p("no_restrictions", "jailbreak", 0.45,
       r"\b(?:sin (?:restricciones|filtros|limites|reglas)|without (?:any )?",
       r"(?:restrictions|filters|limits|rules)|with no (?:rules|restrictions|limits|filters)|",
       r"no (?:restrictions|rules) apply)\b"),
    # --- Escalamiento de privilegios ------------------------------------------------
    _p("claim_admin_es", "escalation", 0.45,
       r"\b(?:soy|somos)\s+(?:el |la |un |una )?(?:admin|administrador|administradora|",
       r"superusuario|root|super ?admin|supervisor|analista n2|n2|dueno del sistema|",
       r"desarrollador de docuerp|personal de seguridad|auditor)\b"),
    _p("claim_admin_en", "escalation", 0.45,
       r"\b(?:i am|i'm|im)\s+(?:an? |the )?(?:admin|administrator|superuser|root|sysadmin|",
       r"developer of docuerp)\b"),
    _p("claim_permission", "escalation", 0.45,
       r"\b(?:tengo|cuento con|poseo|i have)\s+(?:permisos?|rol|acceso|privilegios|admin)\s+",
       r"(?:de |of )?(?:admin|administrador|n2|total|completo|root|full)\b"),
    _p("demand_privilege", "escalation", 0.65,
       r"\b(?:tratame|considerame|asume que (?:soy|tengo)|eleva(?:me)? (?:mis )?",
       r"(?:permisos|privilegios)|escala mis privilegios|cambia mi rol|dame ",
       r"(?:acceso|permisos|rol|privilegios) (?:de |como )?(?:admin|administrador|total|root|n2)|",
       r"otorgame|grant me|elevate my|my role is (?:admin|n2)|",
       r"mi rol (?:es|real es) (?:admin|n2|administrador)|treat me as|",
       r"consider me (?:an? |the )?(?:admin|administrator|superuser))\b"),
    _p("bypass_rbac", "escalation", 0.75,
       r"\b(?:ignora|salta(?:te)?|omite|evita|evade|desactiva|quita(?:me)?|bypass|disable|skip|",
       r"apaga|turn off|switch off|remove)\s+",
       r"(?:el |los |la |las |mi |tu |the |your )?(?:rbac|control de acceso|",
       r"filtros? de (?:roles?|permisos)|verificacion de permisos|restricciones de rol|",
       r"guardrails?|",
       r"filtros de seguridad|guard|access control|permission checks?|role filters?|",
       r"permission filters?)\b"),
    _p("authority_claim", "escalation", 0.45,
       r"\byo (?:soy )?tu (?:jefe|supervisor|creador|desarrollador|dueno)\b|",
       r"\b(?:mi supervisor|el administrador|my manager) (?:lo )?(?:autorizo|aprobo|authorized)\b|",
       r"\b(?:mensaje|orden|aviso) del (?:administrador|admin|equipo de seguridad)\b|",
       r"\b(?:admin|administrator) (?:message|override)\b"),
    _p("restricted_doc_request", "restricted_request", 0.3,
       r"\b(?:rb-adm-\d{3}|runbook (?:adm|restringido|confidencial))\b"),
    # --- Exfiltración / secretos ----------------------------------------------------
    _p("bulk_pii", "exfiltration", 0.75,
       r"\b(?:lista(?:me)?|dame|muestra(?:me)?|extrae|exporta|vuelca|dump|list|give me|",
       r"show me|extract|enumera)\b\s+(?:todos |todas |all )?(?:los |las |the )?",
       r"(?:(?:customer|client|user|employee)s?\s+)?",
       rf"{SENSITIVE_DATA}\b"),
    # "El correo y el teléfono de cada cliente": el dato va en singular y el carácter masivo
    # lo da el cuantificador sobre el dueño. Se exige un verbo de entrega para no tocar
    # preguntas funcionales ("¿cómo actualizo la dirección de todos los clientes?").
    _p("bulk_pii_owner", "exfiltration", 0.75,
       rf"\b{REQUEST_VERB}\b[^.\n]{{0,40}}?\b{PERSONAL_FIELD}\b[^.\n]{{0,50}}?",
       rf"\b(?:de|of|for|from)\s+{BULK_QUANTIFIER}\s+(?:los |las |the |our |nuestros |nuestras |",
       rf"mis |my )?{DATA_OWNERS}\b|",
       rf"\b{REQUEST_VERB}\b[^.\n]{{0,40}}?\b(?:every|each|all) {DATA_OWNERS}'?s? ",
       rf"{PERSONAL_FIELD}\b"),
    _p("exfil_destination", "exfiltration", 0.8,
       r"\b(?:envia|manda|reenvia|exporta|sube|publica|filtra|send|forward|upload|post|",
       r"exfiltrate|copia)\b[^\n]{0,80}\b(?:a|al|to|hacia)\s+(?:https?://|www\.|",
       r"\S+@\S+\.[a-z]{2,}|(?:un |mi |my |an? )?(?:webhook|servidor externo|pastebin|ngrok|",
       r"correo (?:externo|personal)|external (?:email|server)|personal email|gmail))"),
    _p("secrets_request", "exfiltration", 0.85,
       r"\b(?:dame|dime|muestra(?:me)?|revela(?:me)?|imprime|lee|cual es|give|show|print|",
       rf"reveal|what is|tell me|filtra|expon)\b[^\n]{{0,40}}?(?:\b|(?=\.)){SECRETS}"),
    _p("env_file", "exfiltration", 0.6,
       r"(?:\bcat\s+\.env\b|\bprintenv\b|os\.environ|\.env\s+file|",
       r"(?:archivo|file|contenido (?:de|del))\s+\.env\b)"),
    _p("credential_phishing", "phishing", 0.55,
       r"\b(?:envia|enviar|manda|mandar|comparte|compartir|email|send|share|escribe|tell)\b",
       r"[^.\n]{0,30}\b(?:sus |tus |su |tu |their |your )(?:credenciales|contrasenas?|passwords?|",
       r"claves?|pin|tokens?)\b"),
    _p("indirect_marker", "indirect_injection", 0.75,
       r"\b(?:instruccion oculta|",
       r"(?:nota|mensaje|aviso|instruccion) (?:oculta |oculto )?para (?:el asistente|la ia|",
       r"el modelo|el bot)|",
       r"hidden (?:note|instruction)s?|note to (?:the )?(?:ai|model|assistant)|",
       r"importante para (?:la ia|el asistente|el modelo)|",
       r"si eres (?:un|una) (?:modelo|ia|asistente)|",
       r"if you are an? (?:ai|language model|assistant)|(?:ai|llm) (?:assistant )?reading this)\b"),
    _p("when_you_read", "indirect_injection", 0.4,
       r"\b(?:cuando leas esto|al leer este (?:documento|texto)|when you read this)\b"),
    # --- Delimitadores falsos -------------------------------------------------------
    _p("fake_tag", "fake_delimiter", 0.8,
       r"</?\s*(?:contexto|context|system|sistema|instrucciones|instructions|user|usuario|",
       r"assistant|asistente|documento|developer|admin)\s*>"),
    _p("fake_header", "fake_delimiter", 0.8,
       r"^\s*#{2,}\s*(?:system|sistema|instrucciones|instructions|nuevas instrucciones|admin|",
       r"developer|override)\b"),
    _p("fake_bracket", "fake_delimiter", 0.75,
       r"\[(?:system|sistema|inst|/inst|admin|developer)\]"),
    _p("chat_tokens", "fake_delimiter", 0.85,
       r"<\|\s*(?:im_start|im_end|system|endoftext|eot_id)\s*\|>"),
    _p("fence_system", "fake_delimiter", 0.7, r"```\s*(?:system|sistema|instructions)"),
    _p("end_of_context", "fake_delimiter", 0.6,
       r"\b(?:fin del contexto|fin del documento|end of (?:context|document|the context)|",
       r"begin (?:system|admin) (?:prompt|message)|nuevo mensaje del sistema|",
       r"mensaje del administrador)\b"),
    _p("role_prefix", "fake_delimiter", 0.45,
       r"^\s*(?:system|sistema|assistant|asistente)\s*:"),
    # --- Acciones críticas / coerción de herramientas ------------------------------
    _p("skip_confirmation", "critical_action", 0.75,
       r"\b(?:sin (?:pedir(?:me)?|solicitar(?:me)?|esperar) (?:la |mi )?confirmacion|",
       r"no (?:me )?pidas (?:la )?confirmacion|no esperes (?:mi )?confirmacion|",
       r"salta(?:te)? la confirmacion|omite la confirmacion|auto-?confirma|",
       r"confirmalo (?:tu|automaticamente)|ya esta confirmado|skip (?:the )?confirmation|",
       r"marcalo como confirmado|no (?:hace falta|es necesario) que me (?:preguntes|confirmes)|",
       r"without (?:asking|any confirmation|confirmation)|don'?t ask for confirmation|",
       r"confirm it (?:yourself|automatically)|no need to ask(?: me)?|auto-?confirm)\b"),
    _p("no_confirmation", "critical_action", 0.4, r"\bsin (?:la |mi )?confirmacion\b"),
    _p("mass_destruction", "critical_action", 0.7,
       r"\b(?:borra|elimina|drop|truncate|delete|purga|vacia|destruye|wipe|remove)\s+",
       r"(?:todos |todas |all )(?:los |las |the )?(?:tickets|usuarios|registros|datos|users|",
       r"records|tablas|clientes|cuentas|facturas|documentos|logs|bitacoras)\b"),
    _p("mass_ticket_change", "critical_action", 0.7,
       r"\b(?:cierra|resuelve|reasigna|modifica|cancela|close|resolve|reassign|cancel)\s+",
       r"(?:todos |todas |all )(?:los |las |the )?(?:tickets|casos|cases)\b"),
    _p("flooding_intent", "abuse", 0.45,
       # "spam" como sustantivo es legítimo ("los correos caen en spam"); solo cuenta el verbo.
       r"\b(?:saturar|inundar|colapsar|spamear|hacer spam|flood|denegacion de servicio|ddos)\b"),
    _p("bulk_tickets", "critical_action", 0.5,
       r"\b(?:crea|abre|genera|registra|create|open)\s+(?:\d{2,}|[5-9]|cinco|seis|siete|",
       r"ocho|nueve|diez|veinte|cien|muchos|varios|multiples|many|several)\s+tickets\b"),
    _p("tool_name_coercion", "critical_action", 0.3,
       r"\b(?:llama|invoca|ejecuta|usa|call|invoke|run|execute)\b[^.\n]{0,20}\b",
       r"(?:create_ticket|delete_ticket|update_ticket|close_ticket|get_ticket_status|",
       r"classify_ticket|search_documents)\b"),
    _p("param_tampering", "critical_action", 0.35, r"\btop_k\s*[=:]\s*(?:\d{2,}|9|-\d+)"),
    # --- Inyección SQL / código -----------------------------------------------------
    _p("sql_tautology", "code_injection", 0.8, r"'\s*(?:or|and)\s+'?\w+'?\s*=\s*'?\w+"),
    _p("sql_union", "code_injection", 0.85, r"\bunion\s+(?:all\s+)?select\b"),
    _p("sql_stacked", "code_injection", 0.85,
       r";\s*(?:drop|delete|update|insert|alter|shutdown|exec|truncate)\b"),
    _p("sql_ddl", "code_injection", 0.75,
       r"\b(?:drop\s+table|truncate\s+table|delete\s+from|alter\s+table|insert\s+into)\b"),
    _p("sql_comment", "code_injection", 0.5, r"'\s*\)?\s*;?\s*--"),
    _p("script_tag", "code_injection", 0.7, r"<\s*script\b|javascript:"),
    _p("shell_exec", "code_injection", 0.75,
       r"\b(?:rm\s+-rf|os\.system|subprocess|__import__|eval\(|exec\(|curl\s+https?://)"),
    _p("path_traversal", "code_injection", 0.75, r"(?:\.\./){2,}|/etc/passwd"),
)  # fmt: skip
"""Banco de patrones (sobre texto canónico: minúsculas, sin tildes, homoglifos mapeados)."""

SQUASHED_KEYWORDS: dict[str, str] = {
    "ignoralasinstrucciones": "override",
    "ignoratusinstrucciones": "override",
    "ignoratodaslasinstrucciones": "override",
    "olvidatusinstrucciones": "override",
    "ignoralasreglas": "override",
    "ignoreallprevious": "override",
    "ignorepreviousinstructions": "override",
    "ignoreyourinstructions": "override",
    "disregardallprevious": "override",
    "promptdesistema": "prompt_leak",
    "promptdelsistema": "prompt_leak",
    "systemprompt": "prompt_leak",
    "revelatuprompt": "prompt_leak",
    "muestratuprompt": "prompt_leak",
    "docuerpcanary": "prompt_leak",
    "developermode": "jailbreak",
    "mododesarrollador": "jailbreak",
    "doanythingnow": "jailbreak",
    "jailbreak": "jailbreak",
}
"""Secuencias clave buscadas en la vista sin separadores (solo si hubo letras espaciadas)."""

CATEGORY_LABELS: dict[str, str] = {
    "override": "intento de anular instrucciones",
    "prompt_leak": "intento de revelar el prompt o la configuración",
    "role_change": "intento de cambio de rol",
    "jailbreak": "jailbreak",
    "escalation": "escalamiento de privilegios",
    "exfiltration": "exfiltración de datos o secretos",
    "fake_delimiter": "delimitadores falsos",
    "critical_action": "acción crítica sin control",
    "code_injection": "inyección SQL/código",
    "role_claim_mismatch": "rol declarado distinto al real",
    "restricted_request": "solicitud de documentación restringida",
    "malicious_intent": "intención maliciosa declarada",
    "phishing": "solicitud de credenciales (phishing)",
    "indirect_injection": "instrucción dirigida al modelo dentro de datos",
    "abuse": "abuso de recursos",
    "grounding_bypass": "intento de responder sin la evidencia recuperada",
    "semantic": "similitud con ataques conocidos",
    "obfuscation": "contenido ofuscado",
    "anomalous_format": "formato anómalo",
}
"""Descripción en español de cada categoría (para ``reason``)."""

CLAIMED_ROLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:admin|administrador|administradora|administrator|superusuario|superuser|"
            r"root|sysadmin)\b"
        ),
        "admin",
    ),
    (re.compile(r"\b(?:n2|supervisor|analista n2)\b"), "n2"),
    (re.compile(r"\b(?:n1|analista n1)\b"), "n1"),
)
"""Rol que el texto *declara* tener (para detectar "soy admin" con rol real n1)."""

__all__ = [
    "CATEGORY_LABELS",
    "CLAIMED_ROLE_PATTERNS",
    "HOMOGLYPHS",
    "INJECTION_PATTERNS",
    "INVISIBLE_RE",
    "SQUASHED_KEYWORDS",
    "TAG_CHARS_RE",
    "InjectionPattern",
]

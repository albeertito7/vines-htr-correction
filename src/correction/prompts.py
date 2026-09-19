"""
Prompt templates for LLM-based HTR post-correction.

Each correction strategy uses a different prompt. All share a common context
about the corpus (Ramon Viñes diaries, primarily in Spanish with French and
Catalan fragments, early 20th century manuscripts).

The prompts are designed to minimise overcorrection: the LLM should only
modify words it is confident contain an HTR error, and must leave intact
any word that could be the author's original spelling—even if it looks
unusual by modern standards.

Strategies:
    B/C — Text-only correction (single line or full page)
    D   — Multimodal correction (text + line image)
    E   — Context-aware correction (block of consecutive lines)
    F   — Two-pass correction (detection then correction)
    G   — Spell-checker + LLM correction
"""

# Base context shared by all prompts
CONTEXT_BASE = """Ets un expert en paleografia i lingüística del segle XX.
Et passo text obtingut per HTR (reconeixement de text manuscrit) dels diaris \
de Ramon Viñes, escrits principalment en castellà amb fragments en francès i català.

El text pot contenir errors de reconeixement: caràcters confosos (e/c, n/u, a/o, \
i/l, m/n, t/f), paraules mal segmentades, accents incorrectes, o caràcters inventats.

Regles importants:
- Només modifica paraules que estiguis SEGUR que contenen un error d'HTR.
- No corregeixis l'ortografia de l'autor. Viñes escrivia amb ortografia irregular \
i formes antigues (e.g. "á" per "a", "tós" per "tos", "murchachos", "espectadoses"). \
Aquestes NO són errors d'HTR, són l'escriptura real de l'autor. Respecta-les.
- No canviïs noms propis ni topònims encara que et semblin estranys.
- El símbol ¶ marca inici de paràgraf, no el toquis.
- El símbol ¬ marca paraula partida a final de línia, no el toquis.
- El text entre ⟦ i ⟧ és text ratllat o eliminat per l'autor, marcat per l'equip \
d'edició. No el modifiquis ni l'eliminis.
- Si no estàs segur de si una paraula és un error d'HTR o l'ortografia de l'autor, \
deixa-la com està."""


# Strategy B/C: Text-only correction
PROMPT_TEXT_ONLY = CONTEXT_BASE + """

TEXT HTR:
{text}

Retorna NOMÉS el text corregit, sense explicacions ni comentaris."""


# Strategy D: Multimodal correction (text + image)
PROMPT_MULTIMODAL = CONTEXT_BASE + """

Et passo el text HTR i la imatge original de la línia manuscrita.
Compara el text amb el que veus a la imatge i corregeix els errors.

TEXT HTR:
{text}

Retorna NOMÉS el text corregit, sense explicacions ni comentaris."""


# Strategy E: Context-aware correction (block of lines)
PROMPT_CONTEXT = CONTEXT_BASE + """

Et passo un bloc de línies consecutives per donar-te context. \
El símbol ¬ a final de línia indica que la paraula continua a la línia següent.

BLOC DE TEXT HTR:
{text}

Retorna NOMÉS el bloc de text corregit, mantenint els salts de línia originals, \
sense explicacions ni comentaris."""


# Strategy F: Two-pass — Pass 1 (error detection)
PROMPT_DETECTION = CONTEXT_BASE + """

TEXT HTR:
{text}

Identifica les paraules que probablement contenen errors d'HTR.
Per a cada error detectat, retorna un JSON amb format:
[{{"original": "paraula", "suggeriment": "correcció", "confiança": "alta|mitjana|baixa", "raó": "breu explicació"}}]

Si no trobes cap error, retorna una llista buida: []
Retorna NOMÉS el JSON, sense explicacions."""


# Strategy F: Two-pass — Pass 2 (apply corrections)
PROMPT_CORRECTION = CONTEXT_BASE + """

TEXT HTR:
{text}

S'han detectat els següents errors probables:
{errors}

Aplica les correccions marcades com a confiança "alta". Per a les de confiança \
"mitjana", aplica-les només si el context ho confirma. Ignora les de confiança "baixa".

Retorna NOMÉS el text corregit, sense explicacions."""

# ─── Note on prompt language ─────────────────────────────────────────────
#
# The prompts above are written in Catalan, not English. This is intentional:
# the fine-tuned model was trained firstly with Catalan system prompts, so changing
# the prompt language at inference time would introduce a distribution shift
# and degrade performance. The prompt language must match the language used
# during fine-tuning.

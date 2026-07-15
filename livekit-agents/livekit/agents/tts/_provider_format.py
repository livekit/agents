"""Shared provider-specific TTS formatting logic.

Both TTS plugins and the inference gateway delegate to this module so
there is a single source of truth for LLM instructions and markup stripping
per provider.

Provider docs:
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion
- Inworld: https://docs.inworld.ai/tts/capabilities/steering
- Inworld: https://docs.inworld.ai/tts/best-practices/prompting-for-tts-2
- xAI: https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
- xAI: https://docs.x.ai/developers/model-capabilities/audio/voice
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, TypedDict

from ..types import ATTRIBUTE_TRANSCRIPTION_EXPRESSION
from .markup_utils import convert_expression_tags, extract_and_strip


class ExpressiveTag(TypedDict):
    """An expressive markup tag stripped from a transcript, surfaced for the frontend.

    ``type`` is the markup tag name (``"emotion"``, ``"expression"``, ``"sound"``, ...),
    or ``""`` for square-bracket tags which carry no name. ``value`` is the spoken or
    semantic payload (the ``value="..."`` attribute, the tag's inner text, or the bracket
    content).
    """

    type: str
    value: str


if TYPE_CHECKING:
    from .. import tokenize
    from ..voice.agent_session import SpeechSteeringOptions

_CARTESIA_TAGS = ["emotion", "speed", "volume", "break", "spell"]


_INWORLD_TAGS = ["expression", "sound", "break"]


# xAI Grok TTS speech tags, from the xAI docs
# (https://docs.x.ai/developers/rest-api-reference/inference/voice).
#
# The LLM is instructed in the expr dialect (below); these native tag names serve two
# purposes: _XAI_WRAPPING is the label vocabulary expr prosody markers lower to, and all
# of them stay in _XAI_TAGS so a hallucinated native tag is still stripped from the
# transcript rather than leaking. The intermediate <sound value="NAME"/> and
# <break time="..."/> tags that expr lowering produces are rewritten to xAI's native
# brackets by convert_markup — <sound value="X"/> -> [X] and <break> -> [pause] or
# [long-pause] by duration. Prosody is angle-bracketed (native).
_XAI_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "surprised",
    "sympathetic",
    "curious",
    "sarcastic",
    "confident",
    "playful",
    "nervous",
]
_XAI_INLINE = [
    "breath",
    "inhale",
    "exhale",
    "sigh",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "hum-tune",
]
_XAI_WRAPPING = [
    "emphasis",  # stress the wrapped words
    "whisper",  # quiet, intimate
    "soft",  # lower volume
    "loud",  # higher volume
    "build-intensity",  # ramp energy up over the span
    "decrease-intensity",  # ease energy off over the span
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",  # playful, musical lilt
    "singing",  # actually sung
    "laugh-speak",  # talk through a laugh
]
# all tags are XML in the transcript, so all are stripped. inline sounds are the single
# "sound" tag (<sound value="NAME"/>, _XAI_INLINE lists the NAMEs), and pauses use
# "break" (<break time="..."/>), both modeled on Inworld.
_XAI_TAGS = _XAI_EMOTIONS + _XAI_WRAPPING + ["sound", "break"]

# xAI has two pause levels ([pause], [long-pause]); map an Inworld-style <break time="X"/>
# to the longer one past ~1s. This is the only per-provider bit convert_markup needs.
_XAI_BREAK_RE = re.compile(r'<break\s+time="([^"]*)"\s*/?>')


def _xai_break_to_bracket(match: re.Match[str]) -> str:
    raw = match.group(1).strip().lower()
    try:
        secs = float(raw[:-2]) / 1000 if raw.endswith("ms") else float(raw.rstrip("s"))
    except ValueError:
        secs = 0.0
    return "[long-pause]" if secs >= 1.0 else "[pause]"


# --- LiveKit expression markers (expr) ---
# The LLM emits a single marker tag,
# <expr type="..." label="..."/>, instead of provider-native tags. The *syntax* is shared,
# but each provider gets its own instruction block advertising only the types and label
# vocabularies it actually supports — providers offer different sound effects, some take
# only a discrete emotion vocabulary rather than free-form delivery descriptions, and
# only some have wrapping prosody. Types (per provider):
#   expression (self-closing) - delivery/emotion for what follows; free-form for
#                               Inworld, Cartesia's discrete emotion vocabulary, absent
#                               for xAI
#   break      (self-closing) - pause, label is a duration ("500ms", "1s"); all providers
#   sound      (self-closing) - non-verbal vocalization from the provider's own list
#                               (Inworld: laugh/sigh/..., xAI: chuckle/tsk/...); absent
#                               for Cartesia
#   prosody    (wrapping)     - <expr type="prosody" label="whisper">words</expr>, labels
#                               from xAI's wrapping-tag list; for Cartesia a self-closing
#                               point control (slow/fast/soft/loud -> coarse speed/volume
#                               ratios); absent for Inworld (folded into expression)
#   spell      (wrapping)     - <expr type="spell">A7X9</expr> character-by-character
#                               readout; Cartesia only
# convert_markup lowers expr to each provider's native syntax before synthesis (via the
# existing framework-standard tags, so the per-provider conversions below still apply),
# and the transcript strippers remove expr markers in a dedicated pre-pass so the
# type/label pair surfaces correctly as an ExpressiveTag. This is the only dialect the
# LLM is taught — llm_instructions() uses it; the provider-native tag tables remain
# solely so hallucinated native markup is still stripped/converted instead of leaking.

_EXPR_PREAMBLE = """\
You control speech delivery with a single XML marker tag: <expr/>. Every marker has a \
type attribute. Use only the marker types listed below, and where a type lists a label \
vocabulary, only those labels. Use the markers often and diversify them so the voice \
never sounds flat while ensuring the markers are appropriate for the moment.

Just as important is knowing when NOT to reach for a marker. Reserve surprise openers \
like "oh" or "ah" for genuine surprise — an ordinary request isn't one. Don't stack markers \
on short replies or decorate every sentence. If a reaction wouldn't happen in a real \
conversation, skip it — there's always another genuine beat to lean into."""

_CARTESIA_EXPR_LLM_INSTRUCTIONS = (
    _EXPR_PREAMBLE
    + """

1. Emotion - sets the emotional tone. Self-closing; place before EVERY sentence.
   <expr type="expression" label="EMOTION"/>
   Labels are a fixed vocabulary, NOT free-form descriptions. Best results: neutral, \
angry, excited, content, sad, scared.
   Also available: happy, enthusiastic, elated, triumphant, amazed, surprised, \
flirtatious, curious, peaceful, serene, calm, grateful, affectionate, sympathetic, \
mysterious, frustrated, disgusted, sarcastic, ironic, dejected, melancholic, \
disappointed, apologetic, hesitant, confused, anxious, panicked, proud, confident, \
contemplative, determined, joking/comedic.

2. Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="1s"/> - label is a duration in seconds or milliseconds.

3. Prosody - adjusts pacing and loudness from that point on. Self-closing.
   <expr type="prosody" label="slow"/> slower    <expr type="prosody" label="fast"/> faster
   <expr type="prosody" label="soft"/> quieter    <expr type="prosody" label="loud"/> louder
   Labels are a fixed vocabulary: slow, fast, soft, loud.

4. Spell - wraps text read character by character (codes, IDs, or a spelled-out name).
   <expr type="spell">A7X9</expr>
   Keep punctuation out of a spell marker — a period inside is read as "dot"; add \
spaces inside for grouped pauses (<expr type="spell">ABC 123</expr>).

This voice has no non-verbal sounds and no free-form delivery descriptions — do not \
invent other types or labels.

Examples:
  <expr type="expression" label="excited"/> I can't wait to tell you! <expr type="expression" label="happy"/> This is going to be great!
  <expr type="expression" label="curious"/> Really? <expr type="break" label="500ms"/> <expr type="expression" label="excited"/> Tell me more!
  Your code is <expr type="spell">A7X9</expr>. <expr type="break" label="1s"/> <expr type="expression" label="calm"/> Got it?"""
)

_INWORLD_SOUNDS = ["laugh", "sigh", "breathe", "clear throat", "cough", "yawn"]

_INWORLD_EXAMPLES = [
    '<expr type="expression" label="say playfully"/> Okay okay, why did the burger go to the gym? <expr type="break" label="500ms"/> <expr type="expression" label="speak with bright energy"/> Because it wanted better buns! <expr type="sound" label="laugh"/>',  # noqa: E501
    '<expr type="expression" label="sound concerned"/> Ah man, yeah that\'s on us. <expr type="expression" label="speak calmly"/> Lemme see what I can do.',  # noqa: E501
    '<expr type="sound" label="sigh"/> <expr type="expression" label="speak softly, gently"/> I know it\'s been a rough week.',  # noqa: E501
]


def _sound_examples(examples: list[str], allowed: list[str], vocabulary: list[str]) -> list[str]:
    """Drop example lines that demonstrate a *vocabulary* label not in *allowed*."""
    removed = [s for s in vocabulary if s not in allowed]
    return [ex for ex in examples if not any(f'label="{s}"' in ex for s in removed)]


def _numbered_sections(sections: list[str]) -> str:
    return "\n\n".join(f"{i}. {section}" for i, section in enumerate(sections, 1))


def _inworld_expr_llm_instructions(sounds: list[str]) -> str:
    sections = [
        """Delivery - controls how a sentence sounds. Self-closing; place before EVERY sentence.
   <expr type="expression" label="DESCRIPTION"/>
   The label is free-form: describe vocal quality, pitch, volume, pace, and intonation \
in plain English — "say playfully", "speak with warm surprise", "sound concerned", \
"drop to a whisper", "speak slowly and clearly, patient and reassuring"."""
    ]
    if sounds:
        sections.append(
            f"""Sounds - a non-verbal sound between sentences. Self-closing.
   <expr type="sound" label="{sounds[0]}"/>
   Labels are a fixed vocabulary: {", ".join(sounds)}."""
        )
    sections.append(
        """Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="500ms"/> or <expr type="break" label="1s"/> (max 10s).
   A period or an ellipsis (...) already creates a pause, so don't put a break marker \
right next to one — pick one or the other."""
    )

    parts = [
        _EXPR_PREAMBLE,
        _numbered_sections(sections),
        "There is no wrapping prosody marker for this voice — put pace, pitch, and volume in "
        "the expression label instead.",
    ]
    if examples := _sound_examples(_INWORLD_EXAMPLES, sounds, _INWORLD_SOUNDS):
        parts.append("Examples:\n" + "\n".join(f"  {ex}" for ex in examples))
    return "\n\n".join(parts)


_XAI_EXAMPLES = [
    'So I walked in and <expr type="break" label="500ms"/> there it was! <expr type="sound" label="laugh"/> <expr type="prosody" label="whisper">It was a secret the whole time.</expr>',  # noqa: E501
    '<expr type="prosody" label="build-intensity">This is going to be so good</expr> — <expr type="prosody" label="loud">I can\'t wait!</expr> <expr type="sound" label="chuckle"/>',  # noqa: E501
    '<expr type="prosody" label="soft">Hey.</expr> <expr type="sound" label="sigh"/> <expr type="prosody" label="lower-pitch">I know it\'s been a rough week.</expr> I\'m right here.',  # noqa: E501
    '<expr type="prosody" label="laugh-speak">You did not just say that</expr> <expr type="sound" label="giggle"/> okay, <expr type="prosody" label="fast">tell me everything.</expr>',  # noqa: E501
    # sound-free, so at least one example survives any steering filter
    '<expr type="prosody" label="emphasis">Everything</expr> is confirmed for Thursday. <expr type="break" label="500ms"/> <expr type="prosody" label="slow">Is there anything else I can help you with?</expr>',  # noqa: E501
]


def _xai_expr_llm_instructions(sounds: list[str], prosody: list[str]) -> str:
    sections = []
    if sounds:
        sections.append(
            f"""Sounds - a non-verbal vocalization at the exact point where it happens. Self-closing.
   <expr type="sound" label="{sounds[0]}"/>
   Labels are a fixed vocabulary: {", ".join(sounds)}."""
        )
    sections.append(
        """Pauses - insert a beat. Self-closing.
   <expr type="break" label="500ms"/> a brief pause    <expr type="break" label="1s"/> a longer, dramatic pause"""  # noqa: E501
    )
    sections.append(
        f"""Prosody - wraps the exact words it affects to shape HOW they're said.
   <expr type="prosody" label="STYLE">the words it affects</expr>
   Labels are a fixed vocabulary: {", ".join(prosody)}.
   Never nest one prosody marker inside another, and always close it with </expr>."""
    )

    parts = [
        _EXPR_PREAMBLE,
        _numbered_sections(sections),
        "This voice has no free-form delivery descriptions — shape delivery entirely through "
        + ("prosody markers, sounds, pauses" if sounds else "prosody markers, pauses")
        + ", punctuation, and word choice.",
        """To stress a word, wrap it in <expr type="prosody" label="emphasis">...</expr> — do NOT \
write it in all-caps, which is read out as individual letters. Punctuation still shapes \
delivery — commas and periods create natural pauses, so reach for a break marker only \
when you want a beat beyond what the punctuation gives.""",
        """ALWAYS wrap numbers, dates, times, amounts, addresses, and names of specific things in \
<expr type="prosody" label="emphasis">...</expr> so they stand out to the listener — this is \
the one marker that is mandatory, not optional. When the detail is dense or easy to mishear, \
wrap it in <expr type="prosody" label="slow">...</expr> instead, and read codes or reference \
numbers character by character, spelled out with spaces, so each one lands.""",
        """VERY IMPORTANT: Always put <expr type="break" label="750ms"/> right before any number, \
code of any kind, or address — people naturally take a beat before saying one, and it cues \
the listener to catch what comes next.""",
    ]
    if examples := _sound_examples(_XAI_EXAMPLES, sounds + prosody, _XAI_INLINE + _XAI_WRAPPING):
        parts.append("Examples:\n" + "\n".join(f"  {ex}" for ex in examples))
    return "\n\n".join(parts)


# Every provider's full expr sound vocabulary (the advertised labels before any
# speech_steering filtering). Providers absent here have no non-verbal sounds.
_PROVIDER_SOUNDS: dict[str, list[str]] = {
    "inworld": _INWORLD_SOUNDS,
    "xai": _XAI_INLINE,
}


def _steering_removed(
    table: dict[str, dict[str, list[str]]], provider: str, steering: SpeechSteeringOptions | None
) -> set[str]:
    """Labels from a per-provider governance table that *steering* disables."""
    nonverbals = steering.get("nonverbal_sounds") if steering else None
    labels = table.get(provider)
    if nonverbals is None or labels is None:
        return set()
    flags = dict(nonverbals)
    return {lb for f, lbs in labels.items() if not flags.get(f, False) for lb in lbs}


def _allowed_sounds(provider: str, steering: SpeechSteeringOptions | None) -> list[str]:
    """The provider's sound vocabulary minus labels steering disables.

    Every label is governed by a ``NonverbalOptions`` field, so passing
    ``nonverbal_sounds`` with everything off returns an empty list — the
    instruction builders then omit the Sounds section entirely.
    """
    removed = _steering_removed(_NONVERBAL_SOUND_LABELS, provider, steering)
    return [s for s in _PROVIDER_SOUNDS.get(provider, []) if s not in removed]


def _allowed_prosody(provider: str, steering: SpeechSteeringOptions | None) -> list[str]:
    """The provider's wrapping-prosody vocabulary minus labels steering disables.

    Unlike sounds, only the vocal-style labels (laugh-speak, singing, ...) are
    governed — neutral delivery controls (emphasis, whisper, pitch, pace) always
    survive, so the result is never empty.
    """
    removed = _steering_removed(_NONVERBAL_PROSODY_LABELS, provider, steering)
    return [p for p in _PROVIDER_PROSODY.get(provider, []) if p not in removed]


# NonverbalOptions field -> the provider's expr sound labels it governs. A provider
# absent here (cartesia) has no non-verbal sounds; an empty list means the provider
# has no sound for that field (nothing to filter). _allowed_sounds uses this to
# remove disabled labels from the advertised vocabulary, so a sound steering turns
# off is never exposed to the LLM in the first place. Every label in
# _PROVIDER_SOUNDS must be governed by exactly one field, so a preset controls
# the full vocabulary.
_NONVERBAL_SOUND_LABELS: dict[str, dict[str, list[str]]] = {
    "inworld": {
        "laughing": ["laugh"],
        "breathing": ["breathe"],
        "sighing": ["sigh"],
        "crying": [],
        "vocalizing": [],
        "mouth_sounds": [],
        "reflex_sounds": ["cough", "clear throat", "yawn"],
    },
    "xai": {
        "laughing": ["laugh", "chuckle", "giggle"],
        "breathing": ["breath", "inhale", "exhale"],
        "sighing": ["sigh"],
        "crying": ["cry"],
        "vocalizing": ["hum-tune"],  # non-lexical voiced sounds
        "mouth_sounds": ["tsk", "tongue-click", "lip-smack"],
        "reflex_sounds": [],  # xAI has no cough/yawn sounds
    },
}

# NonverbalOptions field -> the provider's wrapping-prosody labels it governs.
# Sparse on purpose: only vocal-style prosody (talking through a laugh, singing)
# is steerable; neutral delivery controls are never filtered.
_NONVERBAL_PROSODY_LABELS: dict[str, dict[str, list[str]]] = {
    "xai": {
        "laughing": ["laugh-speak"],
        "vocalizing": ["sing-song", "singing"],
    },
}

# Every provider's full wrapping-prosody vocabulary (only xAI has one).
_PROVIDER_PROSODY: dict[str, list[str]] = {
    "xai": _XAI_WRAPPING,
}


def supported_nonverbals(provider: str) -> dict[str, list[str]]:
    """``NonverbalOptions`` field -> the sound/prosody labels it governs for *provider*."""
    merged: dict[str, list[str]] = {}
    for table in (_NONVERBAL_SOUND_LABELS, _NONVERBAL_PROSODY_LABELS):
        for field, labels in table.get(provider, {}).items():
            if labels:
                merged.setdefault(field, []).extend(labels)
    return merged


# Sound label -> when a real speaker would make it. The sounds guideline is composed
# from the hints of whichever labels survived steering, so the LLM only ever reads
# usage advice for sounds it's allowed to make. Labels sharing a hint (the laugh
# family) collapse to one clause; labels without an entry fall back to the generic
# sentence. Keyed by label, not NonverbalOptions field, so it's provider-agnostic.
_SOUND_USAGE_HINTS: dict[str, str] = {
    "laugh": "a laugh at something obviously funny",
    "chuckle": "a chuckle at something subtly humorous",
    "giggle": "a chuckle at something subtly humorous",
    "sigh": "a sigh when commiserating",
    "inhale": "a sharp inhale before a big reveal",
    "lip-smack": "a lip-smack or tongue-click as a tiny beat of thought",
    "tongue-click": "a lip-smack or tongue-click as a tiny beat of thought",
    "tsk": "a tsk for mock-disapproval",
}


def _sound_guidance(sounds: list[str]) -> str:
    """The sparing-use guideline, illustrated only with the allowed sounds."""
    hints: list[str] = []
    for sound in sounds:
        hint = _SOUND_USAGE_HINTS.get(sound)
        if hint and hint not in hints:
            hints.append(hint)
    line = "Non-verbal sounds: use one only where the moment genuinely earns it"
    if hints:
        line += " — " + ", ".join(hints)
    return line + ". Most turns have none; never repeat the same sound twice in a row."


def steering_instructions(provider: str, steering: SpeechSteeringOptions) -> str:
    """Render a ``SpeechSteeringOptions`` into delivery guidelines for *provider*.

    Only set fields produce output, so an empty dict adds nothing on top of the
    base template. Disabled sounds never appear here: ``llm_instructions`` filters
    them out of the advertised vocabulary (via ``_allowed_sounds``), so the only
    sound guidance left is how sparingly to use what remains.
    """
    lines: list[str] = []

    if steering.get("nonverbal_sounds") is not None and (
        allowed := _allowed_sounds(provider, steering)
    ):
        lines.append(_sound_guidance(allowed))

    if (disfluencies := steering.get("disfluencies")) is not None:
        lines.append(
            "Sprinkle in natural fillers (um, uh) and openers (oh, well, so), "
            "zero to two per turn, never mechanical."
            if disfluencies
            else "No fillers (um, uh). Sound composed and fluent."
        )

    if (pace := steering.get("pace")) is not None and pace != "normal":
        lines.append(f"Keep a {pace} overall speaking pace.")

    if not lines:
        return ""
    return "Delivery guidelines:\n" + "\n".join(f"- {line}" for line in lines)


# Hard per-provider chunking defaults (characters). The value caps every synthesis
# request at the provider's send limit and, under expressive, doubles as the
# batch size so sentences are grouped up to it. Providers absent here are uncapped
# and always emit per sentence.
_MAX_INPUT_LEN: dict[str, int] = {
    "inworld": 900,
    "cartesia": 400,
    # well under xAI's 15,000-char request limit; sized as an expressive batch
    # target (https://docs.x.ai/developers/model-capabilities/audio/text-to-speech)
    "xai": 1000,
}


def max_input_len(provider: str) -> int | None:
    """Return the max text chunk length for a provider, or None if unlimited."""
    return _MAX_INPUT_LEN.get(provider)


def sentence_tokenizer(provider: str, *, expressive: bool) -> tokenize.SentenceTokenizer:
    """Default blingfire sentence tokenizer for a provider's streamed TTS input.

    The provider's hard max chunk length caps every emitted token. When ``expressive``
    is set, it also raises the *minimum* so consecutive sentences are batched up to
    that size, keeping prosody continuous across the turn; otherwise tokens emit per
    sentence (the unchanged default). Providers with no configured limit are uncapped
    and always per-sentence.
    """
    from .. import tokenize

    max_len = _MAX_INPUT_LEN.get(provider)
    return tokenize.blingfire.SentenceTokenizer(
        max_token_len=max_len,
        min_token_len=max_len if expressive else None,
        # markup only exists in the stream when expressive is active; xml-aware
        # tokenization would otherwise hold streaming on a stray "<" in plain text
        xml_aware=expressive,
    )


_EXPR_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
# any <expr ...> or <expr .../> tag (open or self-closing; attrs in group 1)
_EXPR_OPEN_RE = re.compile(r"<expr\b([^>]*?)/?\s*>")
_EXPR_CLOSE_RE = re.compile(r"</expr\s*>")
# self-closing markers only (the trailing / is required)
_EXPR_SELF_RE = re.compile(r"<expr\b([^>]*?)/\s*>")
# a wrapping marker (prosody/spell) and its span; non-greedy, instructed not to nest
_EXPR_WRAP_RE = re.compile(
    r'<expr\b(?=[^>]*type="(?:prosody|spell)")([^>]*?)>(.*?)</expr\s*>', re.DOTALL
)
# a non-wrapping type the LLM forgot to self-close (normalize_markup fixes these)
_EXPR_UNCLOSED_RE = re.compile(
    r'(<expr\b(?=[^>]*type="(?:expression|break|sound)")[^>]*[^/>\s])\s*>'
)

# expr sound labels that differ from xAI's native cue names
_XAI_SOUND_ALIASES = {"breathe": "breath"}

# Cartesia prosody labels -> native point controls (coarse steps of the numeric ratios)
_CARTESIA_PROSODY = {
    "slow": '<speed ratio="0.85"/>',
    "fast": '<speed ratio="1.2"/>',
    "soft": '<volume ratio="0.9"/>',
    "loud": '<volume ratio="1.3"/>',
}


def _expr_attrs(attrs: str) -> dict[str, str]:
    return dict(_EXPR_ATTR_RE.findall(attrs))


def _split_expr(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip expr markers and collect (type, label) pairs, in document order.

    The generic ``extract_and_strip`` pass can't produce the right ExpressiveTag for
    expr (its type would be the literal tag name ``expr`` and its value the first quoted
    attribute, i.e. the marker type), so expr gets this dedicated pre-pass. A prosody
    wrapper's inner words stay in the clean text — only the delimiters are removed —
    which also keeps streaming safe when an open/close pair is split across chunks.
    """
    if "<expr" not in text and "</expr" not in text:
        return text, []

    tags: list[ExpressiveTag] = []

    def _repl(m: re.Match[str]) -> str:
        attrs = _expr_attrs(m.group(1))
        tags.append({"type": attrs.get("type", ""), "value": attrs.get("label", "")})
        return ""

    clean = _EXPR_OPEN_RE.sub(_repl, text)
    clean = _EXPR_CLOSE_RE.sub("", clean)
    return clean, tags


def _convert_expr(provider: str, text: str) -> str:
    """Lower expr markers to the framework-standard / native tags for *provider*.

    The output still flows through the existing per-provider conversions in
    ``convert_markup`` (e.g. ``<sound value="X"/>`` -> ``[X]`` for Inworld/xAI), so
    this only has to translate expr into those intermediate tags. A type the provider
    doesn't support (its instructions never advertise it, so it's a hallucination) is
    dropped from the audio path — the words survive, the marker never leaks.
    """
    if "<expr" not in text and "</expr" not in text:
        return text

    def _wrap(m: re.Match[str]) -> str:
        attrs = _expr_attrs(m.group(1))
        marker_type = attrs.get("type", "")
        label = attrs.get("label", "").strip().lower()
        inner = m.group(2)
        if marker_type == "spell":
            return f"<spell>{inner}</spell>" if provider == "cartesia" else inner
        # prosody: native wrapping tags exist only for xAI
        if provider == "xai":
            native = label.replace(" ", "-")
            if native in _XAI_WRAPPING:
                return f"<{native}>{inner}</{native}>"
            return inner
        if provider == "inworld":
            # not advertised for Inworld; salvage a stray one as a delivery hint
            return f'<expression value="{label}"/>{inner}'
        if provider == "cartesia":
            # wrapping form of the point controls: apply before the span
            return _CARTESIA_PROSODY.get(label, "") + inner
        return inner

    text = _EXPR_WRAP_RE.sub(_wrap, text)

    def _self(m: re.Match[str]) -> str:
        attrs = _expr_attrs(m.group(1))
        marker_type = attrs.get("type", "")
        label = attrs.get("label", "")
        if marker_type == "expression":
            if provider == "cartesia":
                # Cartesia's discrete emotion vocabulary (instructions list it)
                return f'<emotion value="{label}"/>'
            if provider == "inworld":
                return f'<expression value="{label}"/>'
            return ""  # xAI has no free-form delivery descriptions
        if marker_type == "sound":
            if provider == "cartesia":
                return ""  # no non-verbal sound support
            if provider == "xai":
                label = _XAI_SOUND_ALIASES.get(label.lower(), label)
            return f'<sound value="{label}"/>'
        if marker_type == "break":
            return f'<break time="{label}"/>'
        if marker_type == "prosody" and provider == "cartesia":
            # Cartesia prosody is a self-closing point control (speed/volume)
            return _CARTESIA_PROSODY.get(label.strip().lower(), "")
        return ""

    text = _EXPR_SELF_RE.sub(_self, text)
    # a stray unpaired expr tag (e.g. a prosody wrapper split across stream chunks)
    # must never reach the TTS as literal text — drop the delimiters, keep the words
    text = _EXPR_OPEN_RE.sub("", text)
    text = _EXPR_CLOSE_RE.sub("", text)
    return text


def llm_instructions(provider: str, steering: SpeechSteeringOptions | None = None) -> str | None:
    """Return LLM instruction text for a TTS provider.

    Each markup-capable provider gets its own expr instruction block — shared marker
    syntax, but only the types and label vocabularies that provider actually supports;
    ``convert_markup`` lowers the markers to native syntax. Expr is the only dialect
    the LLM is ever taught. When *steering* disables a non-verbal sound, its labels
    (and any example demonstrating them) are omitted from the block entirely rather
    than advertised and then revoked.
    """
    if provider == "cartesia":
        return _CARTESIA_EXPR_LLM_INSTRUCTIONS
    if provider == "inworld":
        return _inworld_expr_llm_instructions(_allowed_sounds(provider, steering))
    if provider == "xai":
        return _xai_expr_llm_instructions(
            _allowed_sounds(provider, steering), _allowed_prosody(provider, steering)
        )
    return None


# Per-provider markup spec: (xml tag names, whether square-bracket tags are used).
_PROVIDER_MARKUP: dict[str, tuple[list[str], bool]] = {
    "cartesia": (_CARTESIA_TAGS, False),
    "inworld": (_INWORLD_TAGS, True),
    # every tag the LLM is taught is XML (expr markers; native sounds/pauses become
    # [..] only for the TTS in convert_markup), so the transcript has no brackets to strip
    "xai": (_XAI_TAGS, False),
}


def split_markup(provider: str, text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip provider markup and collect the stripped tags in a single pass.

    Returns ``(clean_text, tags)`` — the user-visible transcript plus the expressive
    tags that were removed (in document order), the single source of truth for both
    :func:`strip_markup` and :func:`extract_markup`. ``([], text)`` for providers
    without markup support.
    """
    spec = _PROVIDER_MARKUP.get(provider)
    if spec is None:
        return text, []
    text, expr_tags = _split_expr(text)
    xml_tags, brackets = spec
    clean, raw_tags = extract_and_strip(text, xml_tags=xml_tags, brackets=brackets)
    return clean, expr_tags + [{"type": tag, "value": value} for tag, value in raw_tags]


def strip_markup(provider: str, text: str) -> str:
    """Strip provider-specific markup tags from text, preserving content."""
    return split_markup(provider, text)[0]


def extract_markup(provider: str, text: str) -> list[ExpressiveTag]:
    """Extract the markup tags that :func:`strip_markup` would remove, in order.

    Lets the framework surface stripped expressive tags (e.g. as ``lk.transcription``
    attributes for the frontend) instead of discarding them. Returns ``[]`` for
    providers without markup support.
    """
    return split_markup(provider, text)[1]


# Union of every provider's XML tag names — used by the transcript sinks to strip markup
# without knowing which provider produced it (see :class:`TranscriptMarkupStripper`).
_ALL_MARKUP_TAGS: list[str] = sorted({tag for tags, _ in _PROVIDER_MARKUP.values() for tag in tags})


def split_all_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip the union of every provider's expressive markup (provider-agnostic).

    The transcript sinks strip downstream, where the originating TTS/provider is no
    longer in scope, so they remove every provider's tags (XML + square brackets) at
    once. These tag shapes never appear in real spoken text — the LLM only emits them
    as audio directives — so a universal strip is safe.
    """
    text, expr_tags = _split_expr(text)
    clean, raw_tags = extract_and_strip(text, xml_tags=_ALL_MARKUP_TAGS, brackets=True)
    return clean, expr_tags + [{"type": tag, "value": value} for tag, value in raw_tags]


def strip_all_markup(text: str) -> str:
    """:func:`split_all_markup` returning only the clean text (tags discarded)."""
    return split_all_markup(text)[0]


def strip_expr_markup(text: str) -> str:
    """Strip only the ``<expr/>`` dialect, leaving all other markup untouched.

    Unlike :func:`strip_all_markup`, provider-native tags and square-bracket spans
    survive.
    """
    return _split_expr(text)[0]


def expression_attribute(tags: list[ExpressiveTag]) -> dict[str, str] | None:
    """Build the ``lk.expression`` transcription attribute from stripped markup tags.

    Surfaces a segment's leading delivery/emotion (``expression`` for Inworld/xAI,
    ``emotion`` for Cartesia) as ``{"value": ...}`` so the frontend can react to it.
    Returns ``None`` when no such tag was present.
    """
    expression = next((t["value"] for t in tags if t["type"] in ("expression", "emotion")), None)
    if expression is None:
        return None
    return {
        ATTRIBUTE_TRANSCRIPTION_EXPRESSION: json.dumps({"value": expression}, separators=(",", ":"))
    }


class TranscriptMarkupStripper:
    """Stateful, provider-agnostic markup stripper for one transcript segment.

    Fed text chunk-by-chunk, it returns the user-visible text and accumulates the
    stripped tags. A tag-shaped trailing fragment (a partial ``<...`` or ``[...``
    arriving split across chunks) is held back until it closes, so a tag straddling a
    chunk boundary is never emitted half-stripped. Shared by the transcript sinks (room
    output + transcript synchronizer) so stripping and expression extraction stay
    identical across them.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._tags: list[ExpressiveTag] = []

    def _has_open_tag(self) -> bool:
        # hold a tag-shaped trailing "<" (partial XML tag) so "3 < 5" isn't stalled, and
        # any unclosed "[" (bracket tags have no such ambiguity)
        last_lt = self._buf.rfind("<")
        if last_lt > self._buf.rfind(">"):
            nxt = self._buf[last_lt + 1 : last_lt + 2]
            if not nxt or nxt == "/" or nxt.isalpha():
                return True
        return self._buf.rfind("[") > self._buf.rfind("]")

    def push(self, text: str) -> str:
        """Feed a chunk; return the clean text ready to emit (may be empty)."""
        self._buf += text
        if self._has_open_tag():
            return ""
        clean, tags = split_all_markup(self._buf)
        self._buf = ""
        self._tags.extend(tags)
        return clean

    def flush(self) -> str:
        """Drain any buffered text at segment end; return the remaining clean text."""
        if not self._buf:
            return ""
        clean, tags = split_all_markup(self._buf)
        self._buf = ""
        self._tags.extend(tags)
        return clean

    @property
    def tags(self) -> list[ExpressiveTag]:
        """The markup tags stripped so far, in document order."""
        return self._tags

    def expression_attribute(self) -> dict[str, str] | None:
        """The ``lk.expression`` attribute for the tags stripped so far, if any."""
        return expression_attribute(self._tags)


_SELF_CLOSING_TAGS: dict[str, list[str]] = {
    "cartesia": ["emotion", "speed", "volume", "break"],
    "inworld": ["expression", "sound", "break"],
}


def normalize_markup(provider: str, text: str) -> str:
    """Fix common LLM markup mistakes for a provider.

    Closes opening tags that should be self-closing (e.g. the LLM writes
    ``<expression value="happy">`` instead of ``<expression value="happy"/>`` — or
    ``<expr type="sound" label="laugh">`` instead of ``<expr type="sound" label="laugh"/>``).
    """
    if provider in _PROVIDER_MARKUP:
        text = _EXPR_UNCLOSED_RE.sub(r"\1/>", text)
    tags = _SELF_CLOSING_TAGS.get(provider)
    if not tags:
        return text
    pattern = "|".join(re.escape(t) for t in tags)
    return re.sub(rf"<({pattern})\b([^>]*[^/])\s*>", r"<\1\2/>", text)


def convert_markup(provider: str, text: str) -> str:
    """Convert framework-standard markup to a provider's native syntax."""
    if provider in _PROVIDER_MARKUP:
        # lower expr markers first; the per-provider conversions below then
        # handle the intermediate framework-standard tags they produce
        text = _convert_expr(provider, text)
    if provider in ("inworld", "xai"):
        # <sound value="X"/> -> [X] (and <expression value="X"/> -> [X]); for xAI this
        # turns inline sounds into its native brackets while emotion/prosody stay <..>
        text = convert_expression_tags(text)
    if provider == "xai":
        # xAI has no <break>; map it to its native [pause]/[long-pause]
        text = _XAI_BREAK_RE.sub(_xai_break_to_bracket, text)
    # <break> is otherwise passed through unchanged: Inworld accepts it as native SSML.
    return text

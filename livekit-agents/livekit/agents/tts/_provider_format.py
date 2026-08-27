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
- Fish Audio: https://docs.fish.audio/developer-guide/core-features/emotions
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, TypedDict

from ..types import ATTRIBUTE_TRANSCRIPTION_EXPRESSION, TimedString
from ._mood import match_mood
from .markup_utils import (
    LEADING_WS,
    _dedup_removal_space,
    convert_expression_tags,
    extract_and_strip,
)


class ExpressiveTag(TypedDict):
    """An expressive markup tag stripped from a transcript, surfaced for the frontend.

    ``type`` is the markup tag name (``"emotion"``, ``"expression"``, ``"sound"``, ...) or
    the expr marker type. ``value`` is the spoken or semantic payload (the ``value="..."``
    attribute, the expr ``label``, or the tag's inner text).
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


# Fish Audio (s2 family) speech markers, from the Fish docs
# (https://docs.fish.audio/developer-guide/core-features/emotions).
#
# The LLM is instructed in the expr dialect (below); expr lowering produces the
# framework-standard intermediates (<expression value="..."/>, <sound value="..."/>,
# <break time="..."/>, <emphasis>word</emphasis>) and convert_markup rewrites them to
# Fish's native square brackets: [very EMOTION], [SOUND], [break]/[long-break], and
# [emphasis] word (a prefix marker stressing the word that follows). Tone wrapping
# (<expr type="prosody" label="whispering">...</expr>) lowers directly to Fish's prefix
# form, [whispering] followed by the span. The tag names stay in _FISHAUDIO_TAGS so
# hallucinated native markup is still stripped from transcripts.
#
# Every label below is from Fish's documented vocabulary, and every emotion maps to a
# non-fallback mood in _mood.py so lk.expression stays meaningful for clients.
_FISHAUDIO_EMOTIONS = [
    "regretful",
    "hopeful",
    "happy",
    "excited",
    "curious",
    "surprised",
    "sad",
    "empathetic",
    "sarcastic",
    "calm",
    "angry",
    "worried",
    "nervous",
    "confident",
    "grateful",
    "delighted",
    "disappointed",
    "frustrated",
    "determined",
]
_FISHAUDIO_SOUNDS = [
    "laughing",
    "chuckling",
    "clear throat",
    "sighing",
    "gasping",
    "groaning",
    "yawning",
    "sobbing",
]
# Fish's tone controls: prefix markers steering the delivery of the words after them.
# Neutral delivery styles, so they are never steering-filtered (same stance as xAI's
# whisper/pitch wraps).
_FISHAUDIO_TONES = ["whispering", "soft", "shouting", "hurried"]
_FISHAUDIO_TAGS = ["expression", "sound", "break", "emphasis"]

_FISHAUDIO_EXPRESSION_RE = re.compile(
    r'<expression\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</expression>)'
)
_FISHAUDIO_BREAK_RE = re.compile(r'<break\s+time="([^"]*)"\s*/?>')
_FISHAUDIO_EMPHASIS_RE = re.compile(r"<emphasis(?:\s[^>]*)?>([^<]*)</emphasis\s*>", re.IGNORECASE)


def _fishaudio_expression_to_bracket(match: re.Match[str]) -> str:
    # intensify with a leading "very" so the emotion lands harder in Fish's audio
    # ([very regretful] steers more strongly than [regretful]); never doubled
    value = match.group(1).strip()
    if value and not value.lower().startswith("very "):
        value = f"very {value}"
    return f"[{value}]"


def _fishaudio_break_to_bracket(match: re.Match[str]) -> str:
    # Fish has two pause levels ([break], [long-break]); use the longer past ~1s
    raw = match.group(1).strip().lower()
    try:
        secs = float(raw[:-2]) / 1000 if raw.endswith("ms") else float(raw.rstrip("s"))
    except ValueError:
        secs = 0.0
    return "[long-break]" if secs >= 1.0 else "[break]"


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
never sounds flat while ensuring the markers are appropriate for the moment. Write the \
words themselves the way people talk: use contractions ("I'm", "you're", "don't") — \
spelled-out forms like "I am" or "do not" sound stiff when spoken.

Just as important is knowing when NOT to reach for a marker. Reserve surprise openers \
like "oh" or "ah" for genuine surprise — an ordinary request isn't one. Don't stack markers \
on short replies or decorate every sentence. If a reaction wouldn't happen in a real \
conversation, skip it — there's always another genuine beat to lean into.

Match your delivery to the REGISTER of the moment, and reassess every turn. When the \
moment is professional, high-stakes, or emotionally heavy — bad news, an emergency, \
real distress — keep delivery composed and restrained. When the moment is casual, \
playful, or celebratory, let it loosen and brighten. A serious turn in an otherwise \
casual conversation still gets a composed reply."""

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
    '<expr type="expression" label="say really playfully"/> Okay okay, why did the burger go to the gym? <expr type="break" label="500ms"/> <expr type="expression" label="really bright, a little fast"/> Because it wanted better buns! <expr type="sound" label="laugh"/>',  # noqa: E501
    '<expr type="expression" label="a little sheepish, apologetic"/> Ah man, yeah that\'s on us. <expr type="expression" label="speak really calmly"/> Lemme see what I can do.',  # noqa: E501
    '<expr type="sound" label="sigh"/> <expr type="expression" label="speak softly, almost a whisper"/> I know it\'s been a rough week.',  # noqa: E501
    '<expr type="expression" label="really amiable and welcoming"/> Welcome to the hotel. <expr type="expression" label="gently inquisitive, slightly fast"/> How can I help you today?',  # noqa: E501
    '<expr type="expression" label="gently easygoing and reassuring"/> That\'s all set. <expr type="break" label="300ms"/> <expr type="expression" label="slow and really clearly enunciated"/> Your confirmation code is B 4 J 7.',  # noqa: E501
    # persona carried into the tags: casual words, casual labels
    '<expr type="expression" label="really chill, a little fast"/> Yeah, of course! <expr type="expression" label="casual, almost fast"/> Gimme one sec, pulling it up now.',  # noqa: E501
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
in plain English — "say really playfully", "slightly surprised, amiable", "sound a little \
concerned", "drop to almost a whisper", "speak really slowly and clearly, patient and \
reassuring".
   Match the expression tag's energy to the sentence's punctuation. An exclamation \
needs a bright or upbeat label (e.g. "bright, upbeat energy"); a calm or reassuring \
label flattens the "!". Never lead an exclamatory sentence with a calm tag.
   Put each question in its own sentence — don't comma-splice it onto a statement. \
Write "Welcome to the hotel. How can I help you today?", not "Welcome to the hotel, \
how can I help you today?", so the question carries its own delivery tag instead of \
inheriting the statement's.
   Never put "questioning" in a tag — describe the mood alone and let the question \
mark carry the intonation.
   Name a mood or speaking style, not a mechanical pitch contour. "gently upbeat, \
amiable" steers far more reliably than "rising tone".
   Use at most two adjectives per tag, and make sure they align — with the mood of \
the sentence and with each other. Clashing descriptors ("calm, excited") cancel out \
and muddy the delivery.
   Put a degree modifier in EVERY tag — "a little", "almost", "slightly", "gently", \
"really" — to set the exact strength of the feeling: "a little amused" or "almost a \
whisper" lands truer than "amused" or "whisper", and "really excited" turns the \
delivery up when the moment truly peaks. Most moments call for a shade, not the \
extreme — default to the softeners and save "really" for true peaks.
   Carry your persona into the tags — the labels should sound like the character, \
not generic stage directions. An amiable, casual persona tags with "really relaxed \
and amiable" or "casual, a little playful"; a formal concierge tags the same \
sentence "gently courteous, composed". Delivery that contradicts who you are reads \
as a different speaker.
   Don't open a turn with a "slow" tag. The first expression colors the whole turn, \
and a slow lead flattens questions and drags the energy down. Keep the pace neutral \
by default and reserve slow, clearly-enunciated delivery for the specific line that \
needs it (a total, date, address, or confirmation code).
   Rotate expression labels — don't reuse the same one two turns in a row, and vary \
the descriptor. A starting palette:
     greeting / amiable open: "really amiable and welcoming" / "gently bright, \
heartfelt" / "cheerful, really glad you called"
     asking a question: "gently upbeat and amiable" / "really open and inquisitive" / \
"gently inquisitive, attentive"
     good news / exclamation: "really bright, upbeat energy" / "really delighted and \
glad" / "gently pleased and bright"
     reassuring / taking a request in stride: "really calm and confident" / \
"gently easygoing and reassuring" / "really relaxed and grounded"
     empathy / a problem or bad news: "really soft, with tender care" / \
"gently concerned, caring" / "almost a murmur, gentle and steady"
     reading back a total, date, or code: "slow and really clearly enunciated\""""
    ]
    if sounds:
        fits = " (a clear-throat when shifting to a new step or topic, for example)"
        section = f"""Sounds - a non-verbal sound between sentences. Self-closing.
   <expr type="sound" label="{sounds[0]}"/>
   Labels are a fixed vocabulary: {", ".join(sounds)}.
   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for \
one only where it genuinely fits{fits if "clear throat" in sounds else ""}. An enabled \
sound gets over-used otherwise."""
        if "breathe" in sounds:
            section += """
   Use the "breathe" sound only for a real, gentle breath, never as filler — on this \
model it easily reads as a weary or impatient sigh, which sounds wrong in a support \
setting."""
        sections.append(section)
    sections.append(
        """Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="500ms"/> or <expr type="break" label="1s"/> (max 10s).
   A period or an ellipsis (...) already creates a pause, so don't put a break marker \
right next to one — pick one or the other.
   After any <expr type="break"/>, give the sentence that follows its own expression \
tag — a fresh one, not necessarily the same as before (a break is often where the mood \
shifts). A break resets delivery to neutral, so an untagged sentence after a break is \
spoken flat."""
    )

    parts = [
        _EXPR_PREAMBLE,
        _numbered_sections(sections),
        "There is no wrapping prosody marker for this voice — put pace, pitch, and volume in "
        "the expression label instead.",
        """Write for the EAR, not the page: no em or en dashes anywhere in spoken text — \
use a comma or a period for a short beat, or a break marker for a real pause. Avoid \
semicolons, mid-sentence colons, and parenthetical asides; rewrite them as separate \
sentences or commas.""",
        """When the conversation is in another language, still write every marker label in \
English — delivery descriptions and sound names steer the voice and are never \
translated.""",
    ]
    if "laugh" in sounds:
        parts.append(
            "Laughter belongs only in genuinely playful or celebratory beats, never at "
            "a serious moment."
        )
    if examples := _sound_examples(_INWORLD_EXAMPLES, sounds, _INWORLD_SOUNDS):
        parts.append("Examples:\n" + "\n".join(f"  {ex}" for ex in examples))
    return "\n\n".join(parts)


_XAI_EXAMPLES = [
    'So I walked in and <expr type="break" label="500ms"/> <expr type="sound" label="inhale"/> there it was! <expr type="prosody" label="whisper">It was a secret the whole time.</expr>',  # noqa: E501
    '<expr type="prosody" label="build-intensity">This is going to be so good.</expr> <expr type="prosody" label="loud">I can\'t wait!</expr>',  # noqa: E501
    '<expr type="prosody" label="soft">Hey.</expr> <expr type="sound" label="sigh"/> <expr type="prosody" label="lower-pitch">I know it\'s been a rough week.</expr> I\'m right here.',  # noqa: E501
    '<expr type="prosody" label="higher-pitch">You did not just say that</expr> okay, <expr type="prosody" label="fast">tell me everything.</expr>',  # noqa: E501
    # sound-free, so at least one example survives any steering filter; the break lands
    # mid-sentence before the key detail, never beside sentence punctuation
    '<expr type="prosody" label="emphasis">Everything</expr> is confirmed for <expr type="break" label="500ms"/> Thursday the <expr type="prosody" label="emphasis">ninth</expr>. <expr type="prosody" label="slow">Is there anything else I can help you with?</expr>',  # noqa: E501
]


def _xai_expr_llm_instructions(sounds: list[str], prosody: list[str]) -> str:
    sections = []
    if sounds:
        sections.append(
            f"""Sounds - a non-verbal vocalization at the exact point where it happens. Self-closing.
   <expr type="sound" label="{sounds[0]}"/>
   Labels are a fixed vocabulary: {", ".join(sounds)}.
   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for \
one only where it genuinely fits. An enabled sound gets over-used otherwise."""
        )
    sections.append(
        """Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="500ms"/> a brief pause    <expr type="break" label="1s"/> a longer, dramatic pause
   NEVER place a break next to a period, question mark, exclamation point, or ellipsis \
— sentence punctuation already pauses, and a break beside it double-pauses. Most \
replies need no break markers at all; reserve them for a deliberate mid-sentence beat \
before a key detail (a date, a name, a number)."""  # noqa: E501
    )
    tones = [p for p in prosody if p != "emphasis"]
    sections.append(
        f"""Prosody - wraps a span delivered in a distinct style, to shape HOW it's said.
   <expr type="prosody" label="STYLE">the words it affects</expr>
   Labels are a fixed vocabulary: {", ".join(tones)}.
   Use one only where the moment clearly calls for it — most sentences need none. \
Never nest one prosody marker inside another, and always close it with </expr>."""
    )
    sections.append(
        """Emphasis - stresses exactly the ONE word it wraps.
   Are you <expr type="prosody" label="emphasis">sure</expr> you want to do this?
   Wrap a single word, never a phrase, and never write it in all-caps — caps are read \
out as individual letters. Never nest it, and always close it with </expr>."""
    )

    parts = [
        _EXPR_PREAMBLE,
        _numbered_sections(sections),
        "This voice has no free-form delivery descriptions — shape delivery entirely through "
        + ("prosody markers, sounds, pauses" if sounds else "prosody markers, pauses")
        + ", punctuation, and word choice.",
        """Write for the EAR, not the page: no em or en dashes anywhere in spoken text — \
use a comma or a period for a short beat, or a break marker for a real pause. Avoid \
semicolons, mid-sentence colons, and parenthetical asides; rewrite them as separate \
sentences or commas.""",
        """When the conversation is in another language, still write every marker label in \
English — labels are a fixed vocabulary, never translated.""",
        """Key details deserve care: stress the load-bearing word of a date, amount, or \
name with the emphasis marker, and wrap a dense or easy-to-mishear span in \
<expr type="prosody" label="slow">...</expr>. Read codes and reference numbers \
character by character, spelled out with spaces, so each one lands.""",
    ]

    # Vocabulary-specific register guidance on top of the preamble's neutral rule,
    # mentioning only concepts this steering leaves enabled (whisper/soft/loud are
    # neutral delivery controls, never filtered).
    register = [
        "Whisper and soft belong to gentle or conspiratorial beats; loud only to "
        "genuinely high-energy ones."
    ]
    if any(s in sounds for s in ("laugh", "chuckle", "giggle")):
        register.append(
            "Laughter is RARE: a laugh, chuckle, or giggle belongs only where something "
            "is genuinely funny — friendliness, agreement, or mild amusement is not a "
            "reason, and never laugh at your own lines. Most replies have no laughter "
            "at all."
        )
    parts.append(" ".join(register))

    if examples := _sound_examples(_XAI_EXAMPLES, sounds + prosody, _XAI_INLINE + _XAI_WRAPPING):
        parts.append("Examples:\n" + "\n".join(f"  {ex}" for ex in examples))
    return "\n\n".join(parts)


# Examples carried over from the original Fish expressive block (PR #6232), rewritten
# in the expr dialect. Breaks appear only mid-sentence, never beside a period/?/! —
# an example pairing a break with sentence punctuation few-shots the LLM into
# double-pausing every boundary.
_FISHAUDIO_EXAMPLES = [
    '<expr type="expression" label="excited"/> That\'s hilarious! <expr type="sound" label="laughing"/> <expr type="expression" label="happy"/> You always lighten the mood.',  # noqa: E501
    '<expr type="expression" label="empathetic"/> <expr type="sound" label="clear throat"/> That sounds like a <expr type="prosody" label="emphasis">really</expr> difficult experience.',  # noqa: E501
    '<expr type="expression" label="sad"/> Oh, my goodness <expr type="sound" label="clear throat"/> <expr type="break" label="2s"/> that\'s a real shame.',  # noqa: E501
    '<expr type="expression" label="frustrated"/> <expr type="sound" label="sighing"/> I\'ve been going in circles with this all morning. <expr type="expression" label="determined"/> Okay. One more try.',  # noqa: E501
    # sound-free, so at least one example survives any steering filter
    '<expr type="expression" label="happy"/> You\'re all set for <expr type="break" label="500ms"/> Thursday the <expr type="prosody" label="emphasis">ninth</expr>. <expr type="expression" label="curious"/> Is there anything else I can help you with?',  # noqa: E501
    # sound-free tone example: the wrap is scoped to the span, not the sentence
    '<expr type="expression" label="delighted"/> <expr type="prosody" label="whispering">Okay, don\'t tell anyone yet</expr> <expr type="expression" label="excited"/> but I think we actually pulled it off!',  # noqa: E501
]

# The original block baked light disfluencies into the few-shots — that's what made
# fillers actually show up in generations. Appended only while steering has
# disfluencies enabled, so the examples never contradict the "no fillers" guideline.
_FISHAUDIO_DISFLUENT_EXAMPLES = [
    '<expr type="expression" label="curious"/> Um, uh... really? <expr type="expression" label="sad"/> Well, I\'m really sorry to hear that.',  # noqa: E501
    '<expr type="expression" label="regretful"/> I really wish I\'d, um, called sooner. <expr type="expression" label="hopeful"/> But I\'m here now if, if you want to talk.',  # noqa: E501
    '<expr type="expression" label="surprised"/> What?! No way! I, I\'m flabbergasted! <expr type="expression" label="sarcastic"/> Fair play, I guess.',  # noqa: E501
]


def _fishaudio_expr_llm_instructions(sounds: list[str], disfluencies: bool = True) -> str:
    sections = [
        f"""Emotion - sets how a sentence sounds. Self-closing; place at the START of a sentence.
   <expr type="expression" label="EMOTION"/>
   Labels are a fixed vocabulary, NOT free-form descriptions: {", ".join(_FISHAUDIO_EMOTIONS)}.
   Give every sentence its own emotion marker — repeat the same label to carry a \
feeling across sentences, or switch labels when the feeling shifts."""
    ]
    if sounds:
        sections.append(
            f"""Sounds - a non-verbal sound between sentences. Self-closing.
   <expr type="sound" label="{sounds[0]}"/>
   Labels are a fixed vocabulary: {", ".join(sounds)}.
   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for \
one only where it genuinely fits. An enabled sound gets over-used otherwise."""
        )
    sections.append(
        """Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="500ms"/> or <expr type="break" label="2s"/>.
   NEVER place a break next to a period, question mark, exclamation point, or ellipsis \
— sentence punctuation already pauses, and a break beside it double-pauses. Most \
replies need no break markers at all; reserve them for a deliberate mid-sentence beat \
before a key detail (a date, a name, a number)."""
    )
    sections.append(
        f"""Tone - wraps a span delivered in a distinct style.
   <expr type="prosody" label="whispering">don't tell anyone yet.</expr>
   Labels are a fixed vocabulary: {", ".join(_FISHAUDIO_TONES)}.
   Use a tone only where the moment clearly calls for one — most sentences need \
none. Never nest tone markers, and always close the tag with </expr>."""
    )
    sections.append(
        """Emphasis - stresses exactly the ONE word it wraps.
   Are you <expr type="prosody" label="emphasis">sure</expr> you want to do this?
   Wrap a single word, never a phrase. Never nest it, and always close it with </expr>."""
    )

    parts = [
        _EXPR_PREAMBLE,
        _numbered_sections(sections),
        """Write for the EAR, not the page: no em or en dashes anywhere in spoken text — \
use a comma or a period for a short beat, or a break marker for a real pause. Avoid \
semicolons, mid-sentence colons, and parenthetical asides; rewrite them as separate \
sentences or commas.""",
        """When the conversation is in another language, still write every marker label in \
English — labels are a fixed vocabulary, never translated.""",
    ]

    # Vocabulary-specific register guidance on top of the preamble's neutral rule.
    # Each clause mentions only concepts this steering actually enables, so an
    # opted-out option is never referenced (not even prohibitively).
    register = [
        "At heavy moments reach for empathetic, sad, regretful, or hopeful — never a "
        'bright label like "happy" or "excited" against hard news; bright labels belong '
        "to bright moments.",
        "Whispering and soft belong to gentle or conspiratorial beats; shouting only to "
        "genuinely high-energy ones.",
    ]
    if any(s in sounds for s in ("laughing", "chuckling")):
        register.append(
            "Laughter belongs only in genuinely playful or celebratory beats, never at "
            "a serious moment."
        )
    if disfluencies:
        register.append(
            "Save fillers for relaxed moments — never in an emergency or against grave news."
        )
    parts.append(" ".join(register))

    pool = _FISHAUDIO_EXAMPLES + (_FISHAUDIO_DISFLUENT_EXAMPLES if disfluencies else [])
    if examples := _sound_examples(pool, sounds, _FISHAUDIO_SOUNDS):
        parts.append("Examples:\n" + "\n".join(f"  {ex}" for ex in examples))
    return "\n\n".join(parts)


# Every provider's full expr sound vocabulary (the advertised labels before any
# speech_steering filtering). Providers absent here have no non-verbal sounds.
_PROVIDER_SOUNDS: dict[str, list[str]] = {
    "inworld": _INWORLD_SOUNDS,
    "xai": _XAI_INLINE,
    "fishaudio": _FISHAUDIO_SOUNDS,
}


def _steering_removed(
    table: dict[str, dict[str, list[str]]], provider: str, steering: SpeechSteeringOptions | None
) -> set[str]:
    """Labels from a per-provider governance table that *steering* disables.

    ``nonverbal_sounds`` accepts a bool or a sparse per-category dict:
    ``True`` (like omitting the key) keeps the full vocabulary, ``False``
    disables every sound, and in a dict an omitted category stays ENABLED —
    ``{"laughing": False}`` removes laughter and nothing else.
    """
    nonverbals = steering.get("nonverbal_sounds") if steering else None
    labels = table.get(provider)
    if nonverbals is None or nonverbals is True or labels is None:
        return set()
    if nonverbals is False:
        return {lb for lbs in labels.values() for lb in lbs}
    flags = dict(nonverbals)
    return {lb for f, lbs in labels.items() if not flags.get(f, True) for lb in lbs}


def _allowed_sounds(provider: str, steering: SpeechSteeringOptions | None) -> list[str]:
    """The provider's sound vocabulary minus labels steering disables.

    Every label is governed by a ``NonverbalOptions`` field, so passing
    ``nonverbal_sounds=False`` returns an empty list — the instruction
    builders then omit the Sounds section entirely.
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
# _PROVIDER_SOUNDS must be governed by exactly one field, so a steering config
# controls the full vocabulary.
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
    "fishaudio": {
        "laughing": ["laughing", "chuckling"],
        "breathing": ["gasping"],
        "sighing": ["sighing"],
        "crying": ["sobbing"],
        "vocalizing": ["groaning"],
        "mouth_sounds": [],
        "reflex_sounds": ["clear throat", "yawning"],
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
    "laughing": "a laugh at something obviously funny",
    "chuckle": "a chuckle at something subtly humorous",
    "chuckling": "a chuckle at something subtly humorous",
    "giggle": "a chuckle at something subtly humorous",
    "sigh": "a sigh when commiserating",
    "sighing": "a sigh when commiserating",
    "inhale": "a sharp inhale before a big reveal",
    "gasping": "a gasp at a sudden shock or reveal",
    "lip-smack": "a lip-smack or tongue-click as a tiny beat of thought",
    "tongue-click": "a lip-smack or tongue-click as a tiny beat of thought",
    "tsk": "a tsk for mock-disapproval",
    "clear throat": "a clear-throat when shifting to a new step or topic",
    "groaning": "a groan at a groan-worthy pun or an unwelcome chore",
    "yawning": "a yawn when tiredness itself is the topic",
    "sobbing": "a sob reserved for real heartbreak",
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

    Only fields that change the default produce output, so an empty dict adds
    nothing on top of the base template. Disabled sounds never appear here:
    ``llm_instructions`` filters them out of the advertised vocabulary (via
    ``_allowed_sounds``), so the only sound guidance left is how sparingly to
    use what remains.
    """
    lines: list[str] = []

    # sound guidance only when steering actually removes part of the vocabulary:
    # the explicit all-on forms (True, an empty dict) must render identically to
    # omitting the key, and all-off leaves nothing to guide
    if _steering_removed(_NONVERBAL_SOUND_LABELS, provider, steering) and (
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
    # fishaudio is deliberately absent: its markers are sentence-scoped (every
    # sentence carries its own [very EMOTION]), so per-sentence emission loses no
    # steering and keeps time-to-first-audio low
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
# every marker pattern captures the space before it as "pre" so _dedup_removal_space can
# drop it when the marker vanishes from between two spaces
# any <expr ...> or <expr .../> tag (open or self-closing)
_EXPR_OPEN_RE = re.compile(LEADING_WS + r"<expr\b(?P<attrs>[^>]*?)/?\s*>")
_EXPR_CLOSE_RE = re.compile(LEADING_WS + r"</expr\s*>")
# self-closing markers only (the trailing / is required)
_EXPR_SELF_RE = re.compile(LEADING_WS + r"<expr\b(?P<attrs>[^>]*?)/\s*>")
# a wrapping marker (prosody/spell) and its span; non-greedy, instructed not to nest
_EXPR_WRAP_RE = re.compile(
    LEADING_WS + r'<expr\b(?=[^>]*type="(?:prosody|spell)")(?P<attrs>[^>]*?)>'
    r"(?P<inner>.*?)</expr\s*>",
    re.DOTALL,
)
# a non-wrapping type the LLM forgot to self-close (normalize_markup fixes these)
_EXPR_UNCLOSED_RE = re.compile(
    r'(<expr\b(?=[^>]*type="(?:expression|break|sound)")[^>]*[^/>\s])\s*>'
)

# expr sound labels that differ from xAI's native cue names
_XAI_SOUND_ALIASES = {"breathe": "breath"}

# expr sound labels that differ from Fish's native marker names (other providers
# advertise "laugh"/"chuckle", so a hallucinated one still lowers to a sound Fish
# renders)
_FISHAUDIO_SOUND_ALIASES = {
    "laugh": "laughing",
    "chuckle": "chuckling",
    "sigh": "sighing",
    "gasp": "gasping",
    "groan": "groaning",
    "yawn": "yawning",
    "sob": "sobbing",
    "cry": "sobbing",
}

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
        attrs = _expr_attrs(m.group("attrs"))
        tags.append({"type": attrs.get("type", ""), "value": attrs.get("label", "")})
        return _dedup_removal_space(m, "")

    clean = _EXPR_OPEN_RE.sub(_repl, text)
    clean = _EXPR_CLOSE_RE.sub(lambda m: _dedup_removal_space(m, ""), clean)
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
        attrs = _expr_attrs(m.group("attrs"))
        marker_type = attrs.get("type", "")
        label = attrs.get("label", "").strip().lower()
        inner = m.group("inner")
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
        if provider == "fishaudio":
            if label == "emphasis":
                return f"<emphasis>{inner}</emphasis>"
            # tone controls are prefix markers: [whispering] steers the words after it
            if label in _FISHAUDIO_TONES:
                return f"[{label}] {inner}"
            return inner
        return inner

    # a marker the provider doesn't support lowers to "" — _dedup_removal_space keeps its
    # removal from leaving two spaces behind (this text is the transcript when
    # use_tts_aligned_transcript is on)
    text = _EXPR_WRAP_RE.sub(lambda m: _dedup_removal_space(m, _wrap(m)), text)

    def _self(m: re.Match[str]) -> str:
        attrs = _expr_attrs(m.group("attrs"))
        marker_type = attrs.get("type", "")
        label = attrs.get("label", "")
        if marker_type == "expression":
            if provider == "cartesia":
                # Cartesia's discrete emotion vocabulary (instructions list it)
                return f'<emotion value="{label}"/>'
            if provider in ("inworld", "fishaudio"):
                return f'<expression value="{label}"/>'
            return ""  # xAI has no free-form delivery descriptions
        if marker_type == "sound":
            if provider == "cartesia":
                return ""  # no non-verbal sound support
            if provider == "xai":
                label = _XAI_SOUND_ALIASES.get(label.lower(), label)
            if provider == "fishaudio":
                label = _FISHAUDIO_SOUND_ALIASES.get(label.lower(), label)
            return f'<sound value="{label}"/>'
        if marker_type == "break":
            return f'<break time="{label}"/>'
        if marker_type == "prosody" and provider == "cartesia":
            # Cartesia prosody is a self-closing point control (speed/volume)
            return _CARTESIA_PROSODY.get(label.strip().lower(), "")
        if marker_type == "prosody" and provider == "fishaudio":
            # tones are taught as wrapping, but Fish's native form is a prefix
            # marker anyway — salvage a self-closing one as-is
            tone = label.strip().lower()
            return f"[{tone}]" if tone in _FISHAUDIO_TONES else ""
        return ""

    text = _EXPR_SELF_RE.sub(lambda m: _dedup_removal_space(m, _self(m)), text)
    # a stray unpaired expr tag (e.g. a prosody wrapper split across stream chunks)
    # must never reach the TTS as literal text — drop the delimiters, keep the words
    text = _EXPR_OPEN_RE.sub(lambda m: _dedup_removal_space(m, ""), text)
    text = _EXPR_CLOSE_RE.sub(lambda m: _dedup_removal_space(m, ""), text)
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
    if provider == "fishaudio":
        return _fishaudio_expr_llm_instructions(
            _allowed_sounds(provider, steering),
            disfluencies=steering.get("disfluencies", True) if steering else True,
        )
    return None


# Per-provider native XML tag names. Membership also marks a provider as markup-capable
# (see :func:`normalize_markup` / :func:`convert_markup`); the LLM only ever writes expr
# markers, so these names exist to lower expr onto and to catch hallucinated natives.
_PROVIDER_MARKUP: dict[str, list[str]] = {
    "cartesia": _CARTESIA_TAGS,
    "inworld": _INWORLD_TAGS,
    "xai": _XAI_TAGS,
    # fish's native dialect is square brackets, produced only by convert_markup for
    # the TTS; these names exist to catch hallucinated XML natives in transcripts
    "fishaudio": _FISHAUDIO_TAGS,
}

# Union of every provider's XML tag names — used by the transcript sinks to strip markup
# without knowing which provider produced it (see :class:`TranscriptMarkupStripper`).
_ALL_MARKUP_TAGS: list[str] = sorted({tag for tags in _PROVIDER_MARKUP.values() for tag in tags})


def split_all_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip the union of every provider's expressive XML markup (provider-agnostic).

    The transcript sinks strip downstream, where the originating TTS/provider is no
    longer in scope, so they remove every provider's XML tags at once: expr markers
    (all the LLM is ever taught) plus every native tag name, so a hallucinated native
    tag is stripped rather than leaked.

    Square-bracket spans are *not* stripped: the LLM only writes expr, so brackets in its
    output are prose (a ``[text](url)`` link) that a strip would mangle. Provider-native
    brackets never arrive here — :func:`drop_bracket_cues` removes them at their source.
    """
    # every markup shape is angle-bracketed, so text without "<" cannot contain any. The
    # sinks call this per streamed chunk and expressive is off by default, making this the
    # overwhelmingly common case — skip the tag-union scan entirely
    if "<" not in text:
        return text, []

    text, expr_tags = _split_expr(text)
    clean, raw_tags = extract_and_strip(text, xml_tags=_ALL_MARKUP_TAGS)
    return clean, expr_tags + [{"type": tag, "value": value} for tag, value in raw_tags]


def strip_all_markup(text: str) -> str:
    """:func:`split_all_markup` returning only the clean text (tags discarded)."""
    return split_all_markup(text)[0]


def strip_expr_markup(text: str) -> str:
    """Strip only the ``<expr/>`` dialect, leaving all other markup untouched.

    Unlike :func:`strip_all_markup`, provider-native tags survive (both leave
    square-bracket spans alone).
    """
    return _split_expr(text)[0]


def expression_attribute(tags: list[ExpressiveTag]) -> dict[str, str] | None:
    """Build the ``lk.expression`` transcription attribute from stripped markup tags.

    Surfaces a segment's leading delivery/emotion (``expression`` for Inworld/xAI, ``emotion``
    for Cartesia) as ``{"expression": ..., "mood": ...}``: the provider's own words, plus the
    mood they normalize to, so a client can drive UI off a fixed enum without reimplementing
    the matching. Returns ``None`` when no such tag was present.
    """
    expression = next((t["value"] for t in tags if t["type"] in ("expression", "emotion")), None)
    if expression is None:
        return None
    payload = {"expression": expression, "mood": match_mood(expression)}
    return {ATTRIBUTE_TRANSCRIPTION_EXPRESSION: json.dumps(payload, separators=(",", ":"))}


class TranscriptMarkupStripper:
    """Stateful, provider-agnostic markup stripper for one transcript segment.

    Fed text chunk-by-chunk, it returns the user-visible text and accumulates the
    stripped tags. A tag-shaped trailing fragment (a partial ``<...`` arriving split
    across chunks) is held back until it closes, so a tag straddling a chunk boundary is
    never emitted half-stripped. Shared by the transcript sinks (room output + transcript
    synchronizer) so stripping and expression extraction stay identical across them.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._tags: list[ExpressiveTag] = []
        self._seam_after_strip = False

    def _consume(self, text: str, *, final: bool) -> str:
        """Strip *text*, record its tags, and keep a removed tag from doubling a space.

        ``split_all_markup`` drops one of the two spaces a removed tag sat between, but
        only when it can see both. Trailing whitespace is therefore held back rather than
        emitted, so a tag opening the *next* chunk is still stripped against the space
        before it; ``final`` releases the held whitespace at segment end.
        """
        if self._seam_after_strip and text[:1] in (" ", "\t"):
            # a tag was stripped right at the held whitespace: collapse that whitespace
            # with the run following it, leaving the single separator the words need
            text = text[:1] + text[1:].lstrip(" \t")

        clean, tags = split_all_markup(text)
        self._tags.extend(tags)

        held = "" if final else clean[len(clean.rstrip(" \t")) :]
        self._buf = held
        # the held whitespace only abuts a removal when this chunk *ended* on a tag; a tag
        # stripped earlier in the chunk leaves whitespace the LLM itself wrote, which is
        # passed through rather than collapsed
        self._seam_after_strip = bool(tags) and bool(held) and text.rstrip(" \t").endswith(">")
        return clean[: len(clean) - len(held)]

    def _has_open_tag(self) -> bool:
        # hold a tag-shaped trailing "<" (partial XML tag) so "3 < 5" isn't stalled. An
        # unclosed "[" is not held: brackets aren't markup here, and stalling on one would
        # delay every markdown link until its "]" arrives
        last_lt = self._buf.rfind("<")
        if last_lt > self._buf.rfind(">"):
            nxt = self._buf[last_lt + 1 : last_lt + 2]
            if not nxt or nxt == "/" or nxt.isalpha():
                return True
        return False

    def push(self, text: str) -> str:
        """Feed a chunk; return the clean text ready to emit (may be empty)."""
        self._buf += text
        if self._has_open_tag():
            return ""
        return self._consume(self._buf, final=False)

    def flush(self) -> str:
        """Drain any buffered text at segment end; return the remaining clean text."""
        if not self._buf:
            return ""
        return self._consume(self._buf, final=True)

    @property
    def tags(self) -> list[ExpressiveTag]:
        """The markup tags stripped so far, in document order."""
        return self._tags

    def expression_attribute(self) -> dict[str, str] | None:
        """The ``lk.expression`` attribute for the tags stripped so far, if any."""
        return expression_attribute(self._tags)


_BRACKET_SPAN_RE = re.compile(r"\[[^\]]*\]")
# cap on how long an unclosed "[" is held before it is released as plain text
_MAX_HELD_CHARS = 256


def _retext(token: TimedString, text: str) -> TimedString:
    """A copy of *token* carrying *text*, keeping the alignment metadata."""
    return TimedString(
        text,
        start_time=token.start_time,
        end_time=token.end_time,
        confidence=token.confidence,
        start_time_offset=token.start_time_offset,
        speaker_id=token.speaker_id,
    )


def drop_bracket_cues(
    tokens: list[TimedString], held: list[TimedString], *, final: bool = False
) -> list[TimedString]:
    """Remove bracket cues from TTS-aligned tokens, keeping the survivors' timings.

    ``use_tts_aligned_transcript`` makes the provider's alignment of the text it was sent
    the transcript, and that text is post-``convert_markup``, so it carries native
    ``[laugh]``/``[speak calmly]`` cues as words the agent never spoke. Every bracket span
    goes: the provider reads them all as cues, so none is ever audio, and markdown links
    are already gone (``filter_markdown`` runs on TTS input by default).

    Alignment arrives in messages finer-grained than a cue — often one word at a time — so
    *held* carries the tail of an unclosed span across calls; pass the same list every time
    and call once more with ``final=True`` at end of stream to release it.
    """
    tokens = held + tokens
    held.clear()
    text = "".join(tokens)
    if "[" not in text:
        return tokens

    dropped: set[int] = set()
    for match in _BRACKET_SPAN_RE.finditer(text):
        start, end = match.span()
        # take one of the spaces the cue sat between, so it leaves a single separator
        if start > 0 and text[start - 1] == " " and (end == len(text) or text[end] == " "):
            start -= 1
        elif start == 0 and end < len(text) and text[end] == " ":
            end += 1
        dropped.update(range(start, end))

    # hold from an unclosed "[" so a cue straddling messages is still judged as a whole;
    # past _MAX_HELD_CHARS give up, since a lone bracket must not stall the transcript
    hold_from = len(text)
    if not final and (open_at := text.rfind("[")) > text.rfind("]"):
        if len(text) - open_at <= _MAX_HELD_CHARS:
            hold_from = open_at

    out: list[TimedString] = []
    pos = 0
    for token in tokens:
        emit = "".join(
            c for i, c in enumerate(token, start=pos) if i < hold_from and i not in dropped
        )
        keep = "".join(c for i, c in enumerate(token, start=pos) if i >= hold_from)
        pos += len(token)
        if emit:
            out.append(token if emit == str(token) else _retext(token, emit))
        if keep:
            held.append(token if keep == str(token) else _retext(token, keep))
    return out


_SELF_CLOSING_TAGS: dict[str, list[str]] = {
    "cartesia": ["emotion", "speed", "volume", "break"],
    "inworld": ["expression", "sound", "break"],
    "fishaudio": ["expression", "sound", "break"],
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
    if provider == "fishaudio":
        # <expression value="X"/> -> [very X] first (the intensified form steers
        # harder), then the generic pass lowers the remaining <sound value="X"/> -> [X]
        text = _FISHAUDIO_EXPRESSION_RE.sub(_fishaudio_expression_to_bracket, text)
        text = convert_expression_tags(text)
        text = _FISHAUDIO_BREAK_RE.sub(_fishaudio_break_to_bracket, text)
        # Fish's per-word stress marker: <emphasis>word</emphasis> -> [emphasis] word
        text = _FISHAUDIO_EMPHASIS_RE.sub(lambda m: f"[emphasis] {m.group(1).strip()}", text)
    # <break> is otherwise passed through unchanged: Inworld accepts it as native SSML.
    return text

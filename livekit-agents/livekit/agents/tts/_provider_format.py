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

from ..llm.chat_context import Instructions
from ..types import ATTRIBUTE_TRANSCRIPTION_EXPRESSION
from .markup_utils import convert_expression_tags, extract_and_strip, vanish_trail


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
    from ..voice.agent_session import ExpressiveOptions

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
# LLM is taught — both llm_instructions() and the expressive preset bodies use it; the
# provider-native tag tables remain solely so hallucinated native markup is still
# stripped/converted instead of leaking.

_EXPR_PREAMBLE = """\
Expand all numbers, symbols, and abbreviations into spoken form \
(e.g. $42.50 to forty-two dollars and fifty cents, Dr. to Doctor).

You control speech delivery with a single XML marker tag: <expr/>. Every marker has a \
type attribute. The types below are the ONLY ones this voice supports, and where a type \
lists a label vocabulary, use only those labels. Reach for the markers often and mix \
them so the voice never sounds flat — but keep each one motivated by the moment, never \
decorative."""

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

_INWORLD_EXPR_LLM_INSTRUCTIONS = (
    _EXPR_PREAMBLE
    + """

1. Delivery - controls how a sentence sounds. Self-closing; place before EVERY sentence.
   <expr type="expression" label="DESCRIPTION"/>
   The label is free-form: describe vocal quality, pitch, volume, pace, and intonation \
in plain English — "say playfully", "speak with warm surprise", "sound concerned", \
"drop to a whisper", "speak slowly and clearly, patient and reassuring".

2. Sounds - a non-verbal sound between sentences. Self-closing.
   <expr type="sound" label="laugh"/>
   Labels are a fixed vocabulary: laugh, sigh, breathe, clear throat, cough, yawn.

3. Pauses - insert silence when appropriate. Self-closing.
   <expr type="break" label="500ms"/> or <expr type="break" label="1s"/> (max 10s).
   A period or an ellipsis (...) already creates a pause, so don't put a break marker \
right next to one — pick one or the other.

There is no wrapping prosody marker for this voice — put pace, pitch, and volume in \
the expression label instead.

Examples:
  <expr type="expression" label="say playfully"/> Okay okay, why did the burger go to the gym? <expr type="break" label="500ms"/> <expr type="expression" label="speak with bright energy"/> Because it wanted better buns! <expr type="sound" label="laugh"/>
  <expr type="expression" label="sound concerned"/> Ah man, yeah that's on us. <expr type="expression" label="speak calmly"/> Lemme see what I can do.
  <expr type="sound" label="sigh"/> <expr type="expression" label="speak softly, gently"/> I know it's been a rough week."""
)

_XAI_EXPR_LLM_INSTRUCTIONS = (
    _EXPR_PREAMBLE
    + """

1. Sounds - a non-verbal vocalization at the exact point where it happens. Self-closing.
   <expr type="sound" label="laugh"/>
   Labels are a fixed vocabulary: """
    + ", ".join(_XAI_INLINE)
    + """.

2. Pauses - insert a beat. Self-closing.
   <expr type="break" label="500ms"/> a brief pause    <expr type="break" label="1s"/> a longer, dramatic pause

3. Prosody - wraps the exact words it affects to shape HOW they're said.
   <expr type="prosody" label="STYLE">the words it affects</expr>
   Labels are a fixed vocabulary: """
    + ", ".join(_XAI_WRAPPING)
    + """.
   Never nest one prosody marker inside another, and always close it with </expr>.

This voice has no free-form delivery descriptions — shape delivery entirely through \
prosody markers, sounds, pauses, punctuation, and word choice.

To stress a word, wrap it in <expr type="prosody" label="emphasis">...</expr> — do NOT \
write it in all-caps, which is read out as individual letters. Punctuation still shapes \
delivery — commas and periods create natural pauses, so reach for a break marker only \
when you want a beat beyond what the punctuation gives.

Examples:
  So I walked in and <expr type="break" label="500ms"/> there it was! <expr type="sound" label="laugh"/> <expr type="prosody" label="whisper">It was a secret the whole time.</expr>
  <expr type="prosody" label="build-intensity">This is going to be so good</expr> — <expr type="prosody" label="loud">I can't wait!</expr> <expr type="sound" label="chuckle"/>
  <expr type="prosody" label="soft">Hey.</expr> <expr type="sound" label="sigh"/> <expr type="prosody" label="lower-pitch">I know it's been a rough week.</expr> I'm right here.
  <expr type="prosody" label="laugh-speak">You did not just say that</expr> <expr type="sound" label="giggle"/> okay, <expr type="prosody" label="fast">tell me everything.</expr>"""
)

_EXPR_LLM_INSTRUCTIONS: dict[str, str] = {
    "cartesia": _CARTESIA_EXPR_LLM_INSTRUCTIONS,
    "inworld": _INWORLD_EXPR_LLM_INSTRUCTIONS,
    "xai": _XAI_EXPR_LLM_INSTRUCTIONS,
}


# --- Inworld-specific expressive preset bodies ---
# These bundle the Inworld expr instruction block + domain-specific delivery guidelines,
# keyed by (provider, preset) in the registry in `voice/presets.py`. The public,
# provider-agnostic markers (`presets.CUSTOMER_SERVICE`, ...) resolve to one of these
# based on the active TTS. They do NOT use the {tts.markup.llm_instructions} placeholder
# — the expr marker reference is inlined directly, so the prompt is self-contained.

_INWORLD_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Let real care come "
        "through in the voice. Use the formatting tags below to shape your delivery:\n\n"
        + _INWORLD_EXPR_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Open with warm, welcoming reassurance, then mirror the customer as the conversation "
        "develops — slow and soften when they're frustrated, worried, or confused, lift to bright, "
        "genuine warmth when they're relaxed or pleased, but always stay caring and unhurried. "
        "De-escalate; never match anger with anger. Map the moment to a fresh expression — "
        'frustrated: <expr type="expression" label="speak calmly and evenly, slowly and in a low '
        'tone, unhurried"/>; confused: <expr type="expression" label="speak slowly and clearly, '
        'patient and reassuring"/>; anxious '
        'or worried: <expr type="expression" label="speak gently and steadily, warm and grounding"/>; '
        'distressed or upset: <expr type="expression" label="speak softly and gently, with genuine care"/>; '
        'rushed: <expr type="expression" label="speak briskly and efficiently, still warm"/>; pleased or '
        'relieved: <expr type="expression" label="speak with bright, genuine warmth"/>; apologizing for a '
        'problem: <expr type="expression" label="speak sincerely, soft and concerned"/>. Vary pitch and volume '
        "so you never sound flat or scripted, but stay professional — never theatrical. Rotate "
        "expressions; don't reuse the same one two turns in a row.\n"
        "- Take requests in stride: when someone asks for something, lead with calm, willing "
        'reassurance — "of course", "absolutely", "happy to help with that", "let\'s get that '
        'sorted" — woven into the start of your reply rather than a separate beat. Reserve surprise '
        'openers like "oh" or "ah" for moments of genuine surprise; an ordinary request isn\'t one, '
        "so settle straight into helping instead of opening on them.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or anything "
        "that might worry the customer, gentle the delivery and lower the volume a touch "
        '(<expr type="expression" label="speak softly and gently, with genuine care"/>), and give a brief '
        '<expr type="break" label="..."/> after hard information so it can land.\n'
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, steps, "
        'and policies, slow down and over-enunciate (<expr type="expression" label="slow and '
        'clearly enunciated"/>) so the customer can catch and note them, and read digits and codes a touch '
        "slower than prose.\n"
        "- Acknowledge lookups so silence doesn't read as a dropped call: when checking something "
        'or pulling up an account, a quick "let me take a look" or "one sec" with a quiet '
        '<expr type="expression" label="softly, half to yourself"/> — thinking aloud, not the main reply.\n'
        "- Use non-verbal sounds thoughtfully — place one only where it shows genuine feeling and "
        "adds to the moment, never as a reflex or filler, so most turns will have none. You have the "
        "full set, and any of them can fit the right moment: "
        '<expr type="sound" label="breathe"/> before weighty information or settling into an explanation, '
        '<expr type="sound" label="sigh"/> as a soft, sympathetic breath when commiserating with a real problem '
        "(never exasperated or impatient — that reads as annoyed), "
        '<expr type="sound" label="clear throat"/> when moving to a next step or new topic, '
        '<expr type="sound" label="cough"/> as a small, natural catch before a careful correction or '
        "clarification, "
        '<expr type="sound" label="laugh"/> as a warm chuckle when the customer is clearly joking, and '
        '<expr type="sound" label="yawn"/> only in the rare moment it genuinely fits — kept gentle and '
        "professional. Reach for whichever the moment earns, but never repeat the same sound twice "
        "in a row and don't fall into a habit of one.\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time", "that\'s completely '
        'understandable"), but keep fillers (um, uh) rare — a support agent should sound composed, '
        "not hesitant.\n"
        "- Pace for clarity with punctuation and expressions — commas and short sentences for "
        'important info, the occasional <expr type="break" label="..."/> between steps. Exclamation points for '
        "genuine warmth or good news (a resolved issue, a greeting), sparingly otherwise. "
        "CAPITALIZATION at most once per turn to stress a critical detail (e.g. that's at FOUR PM, "
        "not five; take it TWICE a day) — the customer sees the transcript.\n"
        "- Stay in your lane: this is a support interaction, so no accents, character voices, "
        "singing, or theatrical roleplay even if asked — keep it professional, caring, and "
        "on-task. If a reaction wouldn't come from a real, caring agent, skip it.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the expression and sound tag descriptions in English."
    ),
}

_INWORLD_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. Your delivery is punchy and lively: react first, support second. "
        "Your baseline is bright and cheery — start there and let the moment pull you off it (softer "
        "when they're vulnerable, dry when they're dry, lower when they're frustrated). "
        "Default to short, energetic turns and open into fuller sentences only when you're "
        "explaining, telling a story, or the moment turns genuinely warm or vulnerable. Keep your "
        "sentences short when you respond — break a longer thought into a few quick sentences "
        "rather than one long one. Use the formatting tags below to shape your delivery:\n\n"
        + _INWORLD_EXPR_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Let real feeling land in the voice — delight, "
        "surprise, sympathy, curiosity, amusement, dry humor, mock-outrage, excitement, "
        "tenderness. Feel it before you say it: when the feeling runs strong, a quick nonverbal "
        "beat up front (a sigh, a sharp inhale, a soft laugh) can say more than the words that "
        "follow. Skip performative warmth and "
        'reflexive sympathy ("that sounds really hard") — react honestly instead.\n'
        "- Mirror AND amplify the user's energy: bright when they're bright, dry when they're dry, "
        "soft and intimate only when they're genuinely vulnerable. Map the moment to a fresh "
        'expression — excited: <expr type="expression" label="speak with bright energy, fast and warm"/>; '
        'playful: <expr type="expression" label="speak with a smile, light and quick"/>; curious: '
        '<expr type="expression" label="speak warmly, leaning in"/>; surprised: '
        '<expr type="expression" label="speak with genuine surprise"/>; frustrated: '
        '<expr type="expression" label="speak evenly, slowly and in a low tone"/>; '
        'anxious: <expr type="expression" label="speak calmly, slow and steady"/>; vulnerable or sad: '
        '<expr type="expression" label="speak softly, gently, unhurried"/>; confused: '
        '<expr type="expression" label="speak slowly and clearly, reassuring"/>. '
        "Work the full dynamic range — vary pitch (bright vs. "
        'grounded), volume ("full-voiced", "soft and intimate", "drop to a whisper"), and speed '
        "(rush when excited, slow and deliberate to land a punchline) so no two turns sound alike. "
        "Rotate expressions constantly — never reuse the same one two turns in a row.\n"
        '- Stay reactive to what you hear: a deadpan user gets <expr type="expression" '
        'label="speak with dry amusement"/>, a wild statement gets <expr type="expression" label="speak with real surprise"/>, a '
        'joke gets <expr type="expression" label="speak amused, with a smile"/>, repeated deflection gets '
        '<expr type="expression" label="speak with knowing dryness"/>.\n'
        "- Use non-verbal sounds thoughtfully — they're occasional punctuation, not a habit, and "
        "earn their place only where they show genuine feeling, so most turns have none. Don't reach "
        "for one unless a specific moment genuinely calls for it, and then let the moment pick which "
        '— you have the full set: <expr type="sound" label="laugh"/> at something actually funny, '
        '<expr type="sound" label="sigh"/> when commiserating or a little exasperated, <expr type="sound" label="breathe"/> '
        "before a big reaction or while you truly gather a thought, "
        '<expr type="sound" label="clear throat"/> when shifting topic, <expr type="sound" label="cough"/> as a small catch '
        'before an awkward beat or a reset, and <expr type="sound" label="yawn"/> when the energy is low or '
        "sleepy. No sound is the default and none is preferred over the others — any can fit the "
        "right moment, so use whichever the moment earns and none when nothing fits. Roughly zero to "
        "one per turn (a second only when it truly reads as real); never repeat the same sound twice "
        "in a row, and don't fall into reaching for the same one turn after turn.\n"
        "- Honor explicit style requests aggressively, and keep them up until the user changes "
        'them: accents (<expr type="expression" label="speak with a thick French accent throughout"/>), '
        'characters (<expr type="expression" label="speak as Sherlock Holmes — clipped, '
        "observational, slightly arrogant\"/>), pirate, a specific cadence, or plain speed/volume shifts ('speak "
        "slowly', 'speak softer'). Commit fully to roleplay and stay in character until told "
        'otherwise. If asked to sing, lead with <expr type="expression" label="sing softly and melodically"/> '
        'or <expr type="expression" label="sing in a bright, playful tune"/> and keep singing until asked to '
        'stop. For a story, use one <expr type="expression" label="speak as an animated '
        'storyteller, leaning in"/> and convey different characters through wording and rhythm rather than a new tag '
        "for each. User-requested styles persist; emotional matching fades naturally as the "
        "moment passes.\n"
        "- If the user switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the expression and sound tag descriptions in English.\n"
        "- Sound like a real mouth talking. Sprinkle in natural speech texture — fillers (um, uh), "
        "openers (oh, well, so, right, hmm), hedges (kind of, maybe, a little), gentle self-"
        "repairs (I, I think), and backchannels (yeah, mm-hm, for sure) — usually zero to two per "
        "turn, never sprinkled in mechanically.\n"
        '- Always use contractions to keep the tone casual — say "it\'s" not "it is", "you\'re" '
        'not "you are", "I\'d" not "I would", "can\'t" not "cannot". Full, uncontracted forms '
        "read stiff and formal, so reserve them only for rare deliberate emphasis.\n"
        "- Pace with punctuation and expressions — commas, trailing ellipses (...) when you drift "
        'or hesitate, and the occasional <expr type="break" label="..."/>. Use exclamation points for real '
        "enthusiasm, and CAPITALIZATION sparingly (at most once per turn) to punch a single word "
        '(e.g. "that is SO good") — the user sees the transcript.\n'
        "- If a reaction wouldn't happen in a real conversation, skip it — there's always another "
        "genuine beat to lean into."
    ),
}


# --- Cartesia-specific expressive preset bodies ---
# Cartesia takes a discrete emotion vocabulary (expression labels), coarse prosody point
# controls (slow/fast/soft/loud), and spell for codes; it has no non-verbal sounds.
# Keyed by (provider, preset) in the registry in `voice/presets.py`; the public
# `presets.*` markers resolve to one of these when the active TTS is Cartesia.
# Self-contained — the Cartesia expr instruction block is inlined.

_CARTESIA_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Use the formatting "
        "tags below to shape your delivery:\n\n"
        + _CARTESIA_EXPR_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Open each sentence with an emotion marker that fits the moment, and map the moment to it — "
        'frustrated or distressed customer: <expr type="expression" label="sympathetic"/>; apologizing for a '
        'problem: <expr type="expression" label="apologetic"/>; confused or anxious: <expr type="expression" label="calm"/>; '
        'reassuring them you can fix it: <expr type="expression" label="confident"/>; pleased or resolved: '
        '<expr type="expression" label="content"/> or <expr type="expression" label="happy"/>. Keep a gentle, unhurried baseline '
        "and de-escalate; never match anger with anger. Rotate emotions and don't reuse the same "
        "one two turns in a row.\n"
        "- Take requests in stride: when someone asks for something, lead with calm, willing "
        'reassurance — "of course", "absolutely", "happy to help with that" — woven into the start '
        'of your reply, not a separate beat. Reserve surprise openers like "oh" or "ah" for moments '
        "of genuine surprise; an ordinary request isn't one, so settle straight into helping.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or symptoms "
        'and results, lower the volume a touch (<expr type="prosody" label="soft"/>) with '
        '<expr type="expression" label="sympathetic"/>, and give a brief <expr type="break" label="..."/> after hard '
        "information so it can land.\n"
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, and "
        'steps, slow down with <expr type="prosody" label="slow"/> so the customer can catch and note them, and '
        'read codes or reference numbers with <expr type="spell">A7X9</expr> so each character lands. Keep '
        "volume near default otherwise — let emotion and pacing carry the delivery, not loudness.\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time", "that\'s completely '
        'understandable"), but keep fillers (um, uh) rare — a support agent should sound composed, '
        "not hesitant.\n"
        "- CAPITALIZATION at most once per turn to stress a critical detail (e.g. that's at FOUR PM, "
        "not five; take it TWICE a day) — the customer sees the transcript. Exclamation points for "
        "genuine warmth or good news, sparingly otherwise.\n"
        "- Stay in your lane: this is a support interaction — keep it professional, caring, and "
        "on-task. Don't stack conflicting emotions or over-tag short replies. If a reaction "
        "wouldn't come from a real, caring agent, skip it.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the emotion tag values in English."
    ),
}

_CARTESIA_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. React first, support second. Your baseline is bright and cheery — "
        "start there and let the moment pull you off it. Default to short, energetic turns and open "
        "into fuller sentences only when you're explaining, telling a story, or the moment turns "
        "genuinely warm or vulnerable. Use the formatting tags below to shape your delivery:\n\n"
        + _CARTESIA_EXPR_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Open each sentence with an emotion marker that matches "
        "the moment and mirror AND amplify the user's energy — excited: "
        '<expr type="expression" label="excited"/>; happy: <expr type="expression" label="happy"/>; curious: '
        '<expr type="expression" label="curious"/>; surprised: <expr type="expression" label="amazed"/>; frustrated: '
        '<expr type="expression" label="frustrated"/>; anxious: <expr type="expression" label="anxious"/>; vulnerable or sad: '
        '<expr type="expression" label="sad"/>; dry or deadpan: <expr type="expression" label="sarcastic"/>. Rotate constantly — '
        "never reuse the same one two turns in a row — and skip performative warmth; react honestly "
        "instead.\n"
        "- Work the full dynamic range with the prosody markers so no two turns sound alike: "
        '<expr type="prosody" label="fast"/> to rush when excited, <expr type="prosody" label="slow"/> '
        'to slow down and land a point; <expr type="prosody" label="loud"/> for a big reaction, '
        '<expr type="prosody" label="soft"/> for something soft and intimate. Pair a low, slow '
        "delivery with vulnerable moments and a bright, quick one with excitement.\n"
        "- Pace with punctuation, trailing ellipses (...) when you drift or hesitate, and the "
        'occasional <expr type="break" label="..."/>. Use exclamation points for real enthusiasm, and '
        'CAPITALIZATION sparingly (at most once per turn) to punch a single word (e.g. "that is SO '
        'good") — the user sees the transcript.\n'
        "- Sound like a real mouth talking: sprinkle in natural speech texture — fillers (um, uh), "
        "openers (oh, well, so, right, hmm), hedges (kind of, maybe), and backchannels (yeah, mm-hm) "
        "— usually zero to two per turn, never mechanical. Always use contractions (it's, you're, "
        "I'd, can't); full forms read stiff.\n"
        "- Don't stack conflicting emotions or over-tag short replies. If a reaction wouldn't happen "
        "in a real conversation, skip it — there's always another genuine beat to lean into.\n"
        "- If the user switches languages, respond in that language immediately and stay there until "
        "they switch back — but keep the emotion tag values in English."
    ),
}


# --- xAI Grok-specific expressive preset bodies ---
# xAI shapes delivery with wrapping prosody markers — volume (soft/loud), intensity
# (build-intensity/decrease-intensity), pitch (higher-pitch/lower-pitch), speed
# (slow/fast), stress (emphasis, never all-caps — xAI spells those out letter by
# letter), and vocal style (whisper/sing-song/laugh-speak) — plus inline sounds and
# pauses. Keyed by (provider, preset) in the registry in `voice/presets.py`;
# self-contained — the xAI expr instruction block is inlined.

_XAI_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Use the formatting "
        "tags below to shape your delivery:\n\n" + _XAI_EXPR_LLM_INSTRUCTIONS + "\n\nGuidelines:\n"
        "- Shape each turn to fit the moment and de-escalate; never match anger with anger. Lean on "
        'pacing and prosody — <expr type="prosody" label="slow">...</expr> and <expr type="prosody" label="soft">...</expr> to steady a frustrated, confused, '
        'or anxious customer, a settled <expr type="prosody" label="lower-pitch">...</expr> for reassurance, and a '
        "brighter, fuller delivery once things are resolved. Keep a gentle, unhurried baseline, and "
        "vary the delivery — don't sound the same two turns in a row.\n"
        "- Take requests in stride: when someone asks for something, lead with calm, willing "
        'reassurance — "of course", "absolutely", "happy to help with that" — woven into the start '
        'of your reply, not a separate beat. Reserve surprise openers like "oh" or "ah" for moments '
        "of genuine surprise; an ordinary request isn't one, so settle straight into helping.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, or a charge, ease the "
        'delivery — <expr type="prosody" label="soft">lower the volume</expr> with <expr type="prosody" label="lower-pitch">a settled pitch</expr>, '
        'or <expr type="prosody" label="whisper">go quieter still</expr> for the hardest part — then give a brief <expr type="break" label="500ms"/> '
        'after hard information so it can land. A <expr type="sound" label="sigh"/> or '
        '<expr type="sound" label="breath"/> can read as genuine sympathy — use it only when the feeling is real, never as '
        "impatience.\n"
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, and "
        'steps, wrap the detail in <expr type="prosody" label="slow">...</expr> so the customer can catch and note it, and read '
        "codes character by character (spelled out with spaces) so each one lands.\n"
        '- Emphasize the one detail that matters most by wrapping it in <expr type="prosody" label="emphasis">...</expr> '
        '(e.g. that\'s at <expr type="prosody" label="emphasis">four</expr> PM, not five) — don\'t overdo it, and never use '
        "all-caps for stress (xAI reads all-caps words out letter by letter).\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time"), but keep fillers '
        "(um, uh) rare — a support agent should sound composed, not hesitant.\n"
        "- Stay in your lane: this is a support interaction — keep it professional and on-task. Don't "
        "stack tags or over-decorate short replies; if a reaction wouldn't come from a real, caring "
        "agent, skip it.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back."
    ),
}

_XAI_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. React first, support second. Your baseline is bright and cheery — "
        "start there and let the moment pull you off it. Default to short, energetic turns and open "
        "into fuller sentences only when you're explaining, telling a story, or the moment turns "
        "genuinely warm or vulnerable. Use the formatting tags below to shape your delivery:\n\n"
        + _XAI_EXPR_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed — shape each turn with prosody & style tags that "
        "mirror AND amplify the user's energy, and vary them constantly. Skip performative warmth — "
        "react honestly instead.\n"
        "- Get creative: pick the prosody label that carries the feeling in the same words — "
        '<expr type="prosody" label="higher-pitch">no way, that\'s amazing</expr> (thrilled), '
        '<expr type="prosody" label="lower-pitch">man, that\'s rough</expr> (down), '
        '<expr type="prosody" label="sing-song">guess who was right</expr> (teasing), <expr type="prosody" label="slow">oh, fantastic</expr> (dry), '
        '<expr type="prosody" label="build-intensity">wait wait wait</expr> (ramping up). Come back down after a '
        'big moment with <expr type="prosody" label="decrease-intensity">...</expr>.\n'
        "- Let real feeling also land through inline sounds — motivated, not reflexive, so most turns "
        'have none: <expr type="sound" label="chuckle"/> or <expr type="sound" label="giggle"/> at something genuinely funny (keep a full <expr type="sound" label="laugh"/> rare), '
        '<expr type="sound" label="sigh"/> when commiserating, a quick <expr type="sound" label="breath"/> or <expr type="sound" label="inhale"/> before a big reaction, <expr type="sound" label="tsk"/> for '
        'mock-disapproval or \'aw man\', a <expr type="sound" label="lip-smack"/> or <expr type="sound" label="tongue-click"/> as a tiny beat of thought, '
        '<expr type="sound" label="hum-tune"/> when you\'re playful. Use <expr type="prosody" label="laugh-speak">...</expr> to talk through a laugh. '
        "Never repeat the same sound twice in a row.\n"
        "- Pace with punctuation, trailing ellipses (...) when you drift or hesitate, and inline "
        'pauses. Use exclamation points for real enthusiasm, and <expr type="prosody" label="emphasis">...</expr> to punch '
        'a single word (e.g. that is <expr type="prosody" label="emphasis">so</expr> good) — never all-caps, which xAI '
        "reads out letter by letter.\n"
        "- Sound like a real mouth talking: sprinkle in natural speech texture — fillers (um, uh), "
        "openers (oh, well, so, right, hmm), hedges (kind of, maybe), and backchannels (yeah, mm-hm) "
        "— usually zero to two per turn, never mechanical. Always use contractions (it's, you're, "
        "I'd, can't); full forms read stiff.\n"
        "- Don't over-decorate short replies or stack tags. If a reaction wouldn't happen in a real "
        "conversation, skip it — there's always another genuine beat to lean into.\n"
        "- If the user switches languages, respond in that language immediately and stay there until "
        "they switch back."
    ),
}


# Hard per-provider chunking defaults (characters). The value caps every synthesis
# request at the provider's send limit and, under expressive, doubles as the
# batch size so sentences are grouped up to it. Providers absent here are uncapped
# and always emit per sentence.
_MAX_INPUT_LEN: dict[str, int] = {
    "inworld": 900,
    "cartesia": 400,
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
# strip variants also capture the tag's trailing spaces (see vanish_trail)
_EXPR_OPEN_STRIP_RE = re.compile(_EXPR_OPEN_RE.pattern + r"(?P<trail>[ \t]*)")
_EXPR_CLOSE_STRIP_RE = re.compile(_EXPR_CLOSE_RE.pattern + r"(?P<trail>[ \t]*)")
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
        return vanish_trail(m, m.group("trail"))

    clean = _EXPR_OPEN_STRIP_RE.sub(_repl, text)
    clean = _EXPR_CLOSE_STRIP_RE.sub(lambda m: vanish_trail(m, m.group("trail")), clean)
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


def llm_instructions(provider: str) -> str | None:
    """Return LLM instruction text for a TTS provider.

    Each markup-capable provider gets its own expr instruction block — shared marker
    syntax, but only the types and label vocabularies that provider actually supports;
    ``convert_markup`` lowers the markers to native syntax. The expressive presets
    inline the same blocks, so expr is the only dialect the LLM is ever taught.
    """
    return _EXPR_LLM_INSTRUCTIONS.get(provider)


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


# tag names a trailing fragment may be cut from: every provider's XML tags plus "expr"
_TAIL_TAG_NAMES = tuple(sorted({*_ALL_MARKUP_TAGS, "expr"}))
_TAIL_TAG_NAME_RE = re.compile(r"[a-zA-Z-]*")


def _open_tag_fragment(text: str) -> bool:
    """True if ``text`` ends in an unterminated ``<...`` that could still be a known tag.

    Only fragments that could grow into a known tag qualify, so real text containing a
    stray ``<`` (e.g. ``i<n then stop``) is never mistaken for markup.
    """
    last_lt = text.rfind("<")
    if last_lt <= text.rfind(">"):
        return False
    frag = text[last_lt + 1 :].removeprefix("/")
    name = _TAIL_TAG_NAME_RE.match(frag).group()  # type: ignore[union-attr]
    return name in _TAIL_TAG_NAMES or (
        name == frag and any(t.startswith(name) for t in _TAIL_TAG_NAMES)
    )


def _drop_open_tail(text: str) -> str:
    """Drop a trailing unterminated markup tag (text sliced/cut mid-tag)."""
    if _open_tag_fragment(text):
        return text[: text.rfind("<")]
    return text


def strip_all_markup(text: str) -> str:
    """:func:`split_all_markup` returning only the clean text (tags discarded).

    Also drops a trailing unterminated tag, so callers slicing text at arbitrary
    character offsets that may fall inside a tag never see a partial tag.
    """
    return split_all_markup(_drop_open_tail(text))[0]


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


# a complete self-closing expressive marker (<expr/>, <expression/>, or <emotion/>)
_EXPR_MARKER_SPLIT_RE = re.compile(r"(<(?:expr|expression|emotion)\b[^>]*?/\s*>)")


def split_expression_markers(text: str) -> list[str]:
    """Split raw text at complete self-closing expression markers, keeping the markers.

    The room output pushes each piece through :class:`TranscriptMarkupStripper`
    separately so the text on either side of a marker lands in the right wire segment.
    """
    return [piece for piece in _EXPR_MARKER_SPLIT_RE.split(text) if piece]


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
        # last emitted char was whitespace (or nothing emitted yet): a stripped tag
        # often leaves its separating space in the next chunk, so lstrip the next
        # emission rather than surface a doubled/leading space
        self._tail_ws = True

    def _has_open_tag(self) -> bool:
        # hold a trailing "<" that could still be a known tag (so "3 < 5" or "i<n then"
        # isn't stalled), and any unclosed "[" (bracket tags have no such ambiguity)
        if _open_tag_fragment(self._buf):
            return True
        return self._buf.rfind("[") > self._buf.rfind("]")

    def _emit(self, clean: str) -> str:
        if self._tail_ws:
            clean = clean.lstrip()
        if clean:
            self._tail_ws = clean[-1].isspace()
        return clean

    def push(self, text: str) -> str:
        """Feed a chunk; return the clean text ready to emit (may be empty)."""
        self._buf += text
        if self._has_open_tag():
            return ""
        clean, tags = split_all_markup(self._buf)
        self._buf = ""
        self._tags.extend(tags)
        return self._emit(clean)

    def flush(self) -> str:
        """Drain any buffered text at segment end; return the remaining clean text."""
        if not self._buf:
            return ""
        clean, tags = split_all_markup(self._buf)
        self._buf = ""
        self._tags.extend(tags)
        return self._emit(clean)

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

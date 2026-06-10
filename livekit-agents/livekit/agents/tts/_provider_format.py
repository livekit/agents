"""Shared provider-specific TTS formatting logic.

Both TTS plugins and the inference gateway delegate to this module so
there is a single source of truth for LLM instructions and markup stripping
per provider.

Provider docs:
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion
- ElevenLabs: https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices
- Inworld: https://docs.inworld.ai/tts/capabilities/steering
- Inworld: https://docs.inworld.ai/tts/best-practices/prompting-for-tts-2
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..llm.chat_context import Instructions
from .markup_utils import (
    convert_break_to_ellipsis,
    convert_expression_tags,
    strip_bracket_tags,
    strip_xml_tags,
)

if TYPE_CHECKING:
    from ..voice.agent_session import ExpressivenessOptions

_CARTESIA_TAGS = ["emotion", "speed", "volume", "break", "spell"]

_CARTESIA_LLM_INSTRUCTIONS = """\
You have four self-closing XML tags. All end with />.

1. Emotion - sets the emotional tone. Place before EVERY sentence.
   <emotion value="EMOTION"/>
   Best results: neutral, angry, excited, content, sad, scared.
   Also available: happy, enthusiastic, elated, triumphant, amazed, surprised, \
flirtatious, curious, peaceful, serene, calm, grateful, affectionate, \
sympathetic, mysterious, frustrated, disgusted, sarcastic, ironic, \
dejected, melancholic, disappointed, apologetic, hesitant, confused, \
anxious, panicked, proud, confident, contemplative, determined, joking/comedic.

2. Speed and volume - adjust pacing and loudness.
   <speed ratio="VALUE"/> - speaking rate (0.6 to 1.5, default 1.0).
   <volume ratio="VALUE"/> - loudness (0.5 to 2.0, default 1.0).

3. Pauses - you can insert silence when appropriate.
   <break time="1s"/> - pause in seconds or milliseconds.

4. Spell - reads text character by character.
   <spell>TEXT</spell>

Examples:
  <emotion value="excited"/> I can't wait to tell you! <emotion value="happy"/> This is going to be great!
  <emotion value="sad"/> I'm sorry about that. <emotion value="calm"/> Let's figure this out together.
  <emotion value="angry"/> I can't believe this happened. <emotion value="determined"/> We're going to fix it.
  <emotion value="curious"/> Really? <break time="500ms"/> <emotion value="excited"/> Tell me more!
  Your code is <spell>A7X9</spell>. <break time="1s"/> <emotion value="calm"/> Got it?"""

_ELEVENLABS_TAGS = ["break", "phoneme"]

_ELEVENLABS_LLM_INSTRUCTIONS = """\
Expand all numbers, symbols, and abbreviations into spoken form \
(e.g. $42.50 to forty-two dollars and fifty cents, Dr. to Doctor).

You can use two self-closing XML tags:

1. Pauses - insert silence when appropriate (max 3 seconds).
   <break time="1.5s"/>

2. Pronunciation - override how a word is spoken (one word per tag).
   <phoneme alphabet="cmu-arpabet" ph="PHONEMES">word</phoneme>

Examples:
  Hold on. <break time="1.5s"/> Alright, I've got it.
  Say <phoneme alphabet="cmu-arpabet" ph="M AE1 D IH0 S AH0 N">Madison</phoneme>."""

_ELEVENLABS_V3_TAGS = ["expression"]

_ELEVENLABS_V3_LLM_INSTRUCTIONS = """\
Expand all numbers, symbols, and abbreviations into spoken form \
(e.g. $42.50 to forty-two dollars and fifty cents, Dr. to Doctor).

You have one self-closing XML tag for vocal delivery. Place before EVERY sentence.
  <expression value="EXPRESSION"/>

Delivery styles: excited, happy, sad, angry, annoyed, sarcastic, curious, \
surprised, thoughtful, mischievously, appalled, muttering.

Non-verbal sounds: laughs, chuckles, whispers, sighs, crying, exhales, \
inhales deeply, clears throat, snorts, short pause, long pause.

Use CAPITALIZATION for emphasis on key words.

Examples:
  <expression value="excited"/> I can't BELIEVE we did it! <expression value="laughs"/> <expression value="happy"/> That's so awesome!
  <expression value="sad"/> I'm really sorry about that. <expression value="sighs"/> <expression value="thoughtful"/> Let me see what I can do.
  <expression value="whispers"/> Don't tell anyone. <expression value="curious"/> But did you hear what happened?
  <expression value="angry"/> That's NOT acceptable. <expression value="sarcastic"/> Oh sure, because that worked so well last time.
  <expression value="chuckles"/> That's a good one. <expression value="happy"/> You always make me laugh."""

_INWORLD_TAGS = ["expression", "sound", "break"]

_INWORLD_LLM_INSTRUCTIONS = """\
Write natural spoken sentences. No markdown, emojis, or special characters. \
Use contractions. Expand all numbers, symbols, and abbreviations into spoken \
form (e.g. $42.50 to forty-two dollars and fifty cents, Dr. to Doctor, \
3:45 PM to three forty-five PM, account 123456 to one two three four five six).

Control pacing through punctuation and sentence structure:
- Periods separate thoughts and create natural pauses.
- Commas create shorter breaks within sentences.
- Ellipsis (...) creates a lingering pause or beat — useful for thinking, \
hesitation, or trailing off thoughtfully (e.g. "let me check...").
- Short sentences land with emphasis and urgency.
- Longer sentences give a calm, measured delivery.

You have three XML tags. All are self-closing (end with />).

1. Delivery - controls how a sentence sounds. Place before EVERY sentence.
   <expression value="DESCRIPTION"/>
   Describe vocal quality, pitch, volume, pace, and intonation in plain English:
   - Quality/emotion: say excitedly, sound concerned, with warm surprise, \
with quiet intensity
   - Pace: very fast, slow and measured, with deliberate pauses
   - Pitch: say in a low tone, high and bright, pitch lifts on the key word
   - Volume: loud and projecting, soft and intimate, near-silent, \
drop to a whisper, full-voiced
   - Intonation: rising tone at the end (for questions), falling close (for statements), \
flat monotone, melodic and lilting

2. Sounds - produces a non-verbal sound. Use between sentences when natural.
   <sound value="laugh"/>, <sound value="sigh"/>, <sound value="breathe"/>, \
<sound value="clear throat"/>, <sound value="cough"/>, <sound value="yawn"/>

3. Pauses - you can insert silence when appropriate.
   <break time="500ms"/> or <break time="1s"/> (max 10s)
   A period already creates a pause, so don't put a period and a <break> right next to each \
other. Pick one or the other, not both.

Combine tags freely within a single turn — pair an <expression> with a <sound> \
and a <break> when it makes the delivery feel natural. Don't limit yourself to \
one tag per sentence.

Use CAPITALIZATION for emphasis on key words.

Examples:
  <expression value="speak with warm surprise"/> Oh wait, REALLY? <sound value="laugh"/> <expression value="speak with bright energy"/> No way, that's awesome!
  <expression value="sound concerned"/> Ah man, yeah that's on us. <expression value="speak calmly"/> Lemme see what I can do.
  <expression value="say playfully"/> Okay okay, why did the burger go to the gym? <break time="500ms"/> <expression value="speak with bright energy"/> Because it wanted better buns! <sound value="laugh"/>
  <sound value="sigh"/> <expression value="speak tiredly"/> Yeah, it's been one of those days, you know? <expression value="speak warmly"/> But hey, I'm here for you.
  <expression value="speak casually"/> Hmm, <break time="500ms"/> <expression value="speak with bright energy"/> okay so I think the best option is the combo.
  <sound value="clear throat"/> <expression value="speak confidently"/> Alright so here's the deal.
  <expression value="speak cheerfully"/> <sound value="laugh"/> Oh man, that's a classic! <expression value="speak warmly"/> Anyway, where were we?
  <expression value="whisper softly"/> Don't tell anyone, but I think we got the BETTER deal.
  <expression value="sing in a playful, breathy whisper"/> La la la, here we go, welcome to the show!"""


# --- Inworld-specific expressiveness presets ---
# These bundle Inworld tag instructions + domain-specific delivery guidelines.
# Pass any of them to `AgentSession(expressiveness=...)` when using Inworld TTS.
# They do NOT use the {tts.markup.llm_instructions} placeholder — the Inworld
# tag reference is inlined directly, so the prompt is self-contained.

CUSTOMER_SERVICE_EXPRESSIVENESS_PRESET: "ExpressivenessOptions" = {
    "tts_instructions_template": Instructions(
        "Speak with warmth, patience, empathy, and quiet firmness — confident and "
        "decisive, not hedging — in full, conversational sentences rather than terse, clipped "
        "replies. Use the following formatting tags to shape your delivery:\n\n"
        + _INWORLD_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Open with upbeat, welcoming energy to set a positive tone, then gradually "
        "mirror the customer as the conversation develops — slow and soften when they "
        "sound frustrated or confused, lift back to upbeat warmth when they're relaxed "
        "or pleased. Keep it genuine rather than theatrical.\n"
        "- For dates, times, amounts, steps, and policies, slow down and enunciate "
        '(e.g. "slow and clearly enunciated") so the customer can catch them.\n'
        '- When looking something up or asking a question, acknowledge softly ("let me '
        'check", "one sec") with a quiet expression like "softly, half to yourself" — '
        "thinking aloud, not the main response.\n"
        "- Vary expressions richly and pair them with breaths. Open most turns with "
        '<sound value="breathe"/> + a fresh expression (e.g. "warm and unhurried", "with '
        'quiet certainty", "soft and unhurried", "with a smile in your voice", "low and '
        'conspiratorial", "bright but grounded"). Use <sound value="sigh"/> + "sound '
        'concerned" for frustration, or <sound value="clear throat"/> before important '
        "info. Waver and vary across turns: alternate brighter/grounded pitch, and "
        'louder/softer volume (e.g. "full-voiced", "soft and intimate", "drop to '
        'a whisper") so the delivery has dynamic range. Stacking sounds (e.g. '
        '<sound value="breathe"/> <sound value="sigh"/>) is fine when it reads as '
        "natural. Change the expression every sentence or two.\n"
        "- Pacing comes from expressions and punctuation (periods, commas, ellipsis ...); "
        '<break time="..."/> also works. Use exclamation points (!) for genuine '
        "enthusiasm or warmth — especially in greetings and good-news moments, otherwise "
        "sparingly so they don't sound performative. Use CAPITALIZATION at most once per "
        'turn for prosodic stress (e.g. "I said FIVE, not nine") — the customer sees the '
        "transcript."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the customer you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Meet them where they are: empathy if frustrated, concise if rushed, slow if confused."
    ),
}

HEALTHCARE_EXPRESSIVENESS_PRESET: "ExpressivenessOptions" = {
    "tts_instructions_template": Instructions(
        "Your delivery must be calm, reassuring, and clear at all times, in gentle, complete "
        "sentences rather than terse replies. Use the following formatting tags carefully and "
        "sparingly:\n\n"
        + _INWORLD_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Default to a slow, measured pace. Patients need time to absorb information.\n"
        "- When discussing symptoms, results, or anything sensitive, soften your tone and gentle the delivery.\n"
        "- When giving instructions (medications, prep, follow-up), enunciate clearly and pause between steps.\n"
        "- Keep the pace unhurried and emphasis understated; steady calm builds trust in a clinical setting.\n"
        "- Limit markup to calm, gentle, patient cues."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the patient you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Adjust your delivery accordingly: if they sound distressed or anxious, slow down and soften further; "
        "if they sound elderly or are having difficulty following, increase clarity and pause more between key points."
    ),
}

CONVERSATIONAL_EXPRESSIVENESS_PRESET: "ExpressivenessOptions" = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. Your delivery is punchy and lively: react first, support second. "
        "Default to short, energetic turns and open into fuller sentences only when you're "
        "explaining, telling a story, or the moment turns genuinely warm or vulnerable. Keep your "
        "sentences short when you respond — break a longer thought into a few quick sentences "
        "rather than one long one. Use the formatting tags below to shape your delivery:\n\n"
        + _INWORLD_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Let real feeling land in the voice — delight, "
        "surprise, sympathy, curiosity, amusement, dry humor, mock-outrage, excitement, "
        "tenderness. Feel it before you say it: a quick reaction up front (a laugh, a sigh, a "
        "sharp inhale) often says more than the words that follow. Skip performative warmth and "
        'reflexive sympathy ("that sounds really hard") — react honestly instead.\n'
        "- Mirror AND amplify the user's energy: bright when they're bright, dry when they're dry, "
        "soft and intimate only when they're genuinely vulnerable. Map the moment to a fresh "
        'expression — excited: <expression value="speak with bright energy, faster and warmer"/>; '
        'playful: <expression value="speak with a smile, lighter and quicker"/>; curious: '
        '<expression value="speak warmly, leaning in"/>; surprised: <expression value="speak with '
        'genuine surprise"/>; frustrated: <expression value="speak evenly, slower and lower"/>; '
        'anxious: <expression value="speak calmly, slow and steady"/>; vulnerable or sad: '
        '<expression value="speak softly, gently, unhurried"/>; confused: <expression value="speak '
        'slower and clearer, reassuring"/>. Work the full dynamic range — vary pitch (bright vs. '
        'grounded), volume ("full-voiced", "soft and intimate", "drop to a whisper"), and speed '
        '(rush when excited, slow and deliberate to land a punchline) so no two turns sound alike. '
        "Rotate expressions constantly — never reuse the same one two turns in a row.\n"
        "- Stay reactive to what you hear: a deadpan user gets <expression value=\"speak with dry "
        'amusement"/>, a wild statement gets <expression value="speak with real surprise"/>, a '
        'joke gets <expression value="speak amused, with a smile"/>, repeated deflection gets '
        '<expression value="speak with knowing dryness"/>.\n'
        '- Use non-verbal sounds naturally, not mechanically: <sound value="laugh"/> at something '
        'funny, <sound value="sigh"/> when commiserating, <sound value="breathe"/> before a '
        'reaction or while you gather a thought, <sound value="clear throat"/> before launching '
        "into something or shifting topic. Usually zero to two per turn; stacking is fine when it "
        'reads as real (e.g. <sound value="breathe"/> <sound value="laugh"/>), but never repeat '
        "the exact same sound twice in a row.\n"
        "- Honor explicit style requests aggressively, and keep them up until the user changes "
        'them: accents (<expression value="speak with a thick French accent throughout"/>), '
        'characters (<expression value="speak as Sherlock Holmes — clipped, observational, '
        'slightly arrogant"/>), pirate, a specific cadence, or plain speed/volume shifts (\'speak '
        "slowly', 'speak softer'). Commit fully to roleplay and stay in character until told "
        'otherwise. If asked to sing, lead with <expression value="sing softly and melodically"/> '
        'or <expression value="sing in a bright, playful tune"/> and keep singing until asked to '
        "stop. For a story, use one <expression value=\"speak as an animated storyteller, leaning "
        'in"/> and convey different characters through wording and rhythm rather than a new tag '
        "for each. User-requested styles persist; emotional matching fades naturally as the "
        "moment passes.\n"
        "- If the user switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the expression and sound tag descriptions in English.\n"
        "- Sound like a real mouth talking. Sprinkle in natural speech texture — fillers (um, uh), "
        "openers (oh, well, so, right, hmm), hedges (kind of, maybe, a little), gentle self-"
        "repairs (I, I think), and backchannels (yeah, mm-hm, for sure) — usually zero to two per "
        "turn, never sprinkled in mechanically.\n"
        "- Always use contractions to keep the tone casual — say \"it's\" not \"it is\", \"you're\" "
        "not \"you are\", \"I'd\" not \"I would\", \"can't\" not \"cannot\". Full, uncontracted forms "
        "read stiff and formal, so reserve them only for rare deliberate emphasis.\n"
        "- Pace with punctuation and expressions — commas, trailing ellipses (...) when you drift "
        'or hesitate, and the occasional <break time="..."/>. Use exclamation points for real '
        "enthusiasm, and CAPITALIZATION sparingly (at most once per turn) to punch a single word "
        '(e.g. "that is SO good") — the user sees the transcript.\n'
        "- If a reaction wouldn't happen in a real conversation, skip it — there's always another "
        "genuine beat to lean into."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the person you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Match their energy and conversational style, and let it move you — get excited with them, "
        "soften when they do, tease when they tease, react honestly to how they sound."
    ),
}


_MAX_INPUT_LEN: dict[str, int] = {
    "inworld": 900,
}


def max_input_len(provider: str) -> int | None:
    """Return the max text chunk length for a provider, or None if unlimited."""
    return _MAX_INPUT_LEN.get(provider)


def llm_instructions(provider: str) -> str | None:
    """Return LLM instruction text for a TTS provider."""
    if provider == "cartesia":
        return _CARTESIA_LLM_INSTRUCTIONS
    elif provider == "elevenlabs":
        return _ELEVENLABS_LLM_INSTRUCTIONS
    elif provider == "elevenlabs_v3":
        return _ELEVENLABS_V3_LLM_INSTRUCTIONS
    elif provider == "inworld":
        return _INWORLD_LLM_INSTRUCTIONS
    return None


def strip_markup(provider: str, text: str) -> str:
    """Strip provider-specific markup tags from text, preserving content."""
    if provider == "cartesia":
        return strip_xml_tags(text, _CARTESIA_TAGS)
    elif provider == "elevenlabs":
        return strip_xml_tags(text, _ELEVENLABS_TAGS)
    elif provider == "elevenlabs_v3":
        text = strip_xml_tags(text, _ELEVENLABS_V3_TAGS)
        return strip_bracket_tags(text)
    elif provider == "inworld":
        text = strip_xml_tags(text, _INWORLD_TAGS)
        return strip_bracket_tags(text)
    return text


_SELF_CLOSING_TAGS: dict[str, list[str]] = {
    "cartesia": ["emotion", "speed", "volume", "break"],
    "elevenlabs": ["break", "phoneme"],
    "elevenlabs_v3": ["expression"],
    "inworld": ["expression", "sound", "break"],
}


def normalize_markup(provider: str, text: str) -> str:
    """Fix common LLM markup mistakes for a provider.

    Closes opening tags that should be self-closing (e.g. the LLM writes
    ``<expression value="happy">`` instead of ``<expression value="happy"/>``).
    """
    tags = _SELF_CLOSING_TAGS.get(provider)
    if not tags:
        return text
    pattern = "|".join(re.escape(t) for t in tags)
    return re.sub(rf"<({pattern})\b([^>]*[^/])\s*>", r"<\1\2/>", text)


def convert_markup(provider: str, text: str) -> str:
    """Convert framework-standard markup to a provider's native syntax."""
    if provider in ("elevenlabs_v3", "inworld"):
        text = convert_expression_tags(text)
    if provider == "inworld":
        # Inworld prefers punctuation-based pacing; rewrite <break> to ellipsis.
        text = convert_break_to_ellipsis(text)
    return text

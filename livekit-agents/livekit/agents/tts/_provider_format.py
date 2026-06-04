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

from .markup_utils import (
    convert_break_to_ellipsis,
    convert_expression_tags,
    strip_bracket_tags,
    strip_xml_tags,
)

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

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

from .markup_utils import convert_expression_tags, strip_bracket_tags, strip_xml_tags

_CARTESIA_TAGS = ["emotion", "speed", "volume", "break", "spell"]

_CARTESIA_LLM_INSTRUCTIONS = """\
Control speech with self-closing XML tags placed before the text they affect.

<emotion value="EMOTION"/> — sets emotional tone. Available: neutral, happy, \
excited, enthusiastic, elated, euphoric, triumphant, amazed, surprised, \
flirtatious, curious, content, peaceful, serene, calm, grateful, affectionate, \
sympathetic, anticipation, mysterious, angry, mad, outraged, frustrated, \
agitated, threatened, disgusted, contempt, envious, sarcastic, ironic, sad, \
dejected, melancholic, disappointed, hurt, guilty, bored, tired, rejected, \
nostalgic, wistful, apologetic, hesitant, insecure, confused, resigned, \
anxious, panicked, alarmed, scared, proud, confident, distant, skeptical, \
contemplative, determined, joking/comedic.
<speed ratio="VALUE"/> — speaking rate (0.6–2.0, default 1.0).
<volume ratio="VALUE"/> — loudness (0.5–2.0, default 1.0).
<break time="1s"/> — pause (seconds or milliseconds).
<spell>TEXT</spell> — reads character by character (only wrapping tag).

Examples:
  <emotion value="excited"/> I can't wait to tell you!
  <emotion value="sad"/> I'm sorry. <emotion value="calm"/> Let's figure this out.
  Your code is <spell>A7X9</spell>. <break time="1s"/> Got it?"""

_ELEVENLABS_TAGS = ["break", "phoneme"]

_ELEVENLABS_LLM_INSTRUCTIONS = """\
Expand all numbers, symbols, and abbreviations into spoken form \
(e.g. $42.50 → forty-two dollars and fifty cents, Dr. → Doctor).

<break time="1.5s"/> — pause (max 3 seconds, use sparingly).
<phoneme alphabet="cmu-arpabet" ph="PHONEMES">word</phoneme> — pronunciation override (one word per tag).

Examples:
  Hold on. <break time="1.5s"/> Alright, I've got it.
  Say <phoneme alphabet="cmu-arpabet" ph="M AE1 D IH0 S AH0 N">Madison</phoneme>."""

_ELEVENLABS_V3_TAGS = ["expression"]

_ELEVENLABS_V3_LLM_INSTRUCTIONS = """\
Expand all numbers, symbols, and abbreviations into spoken form \
(e.g. $42.50 → forty-two dollars and fifty cents, Dr. → Doctor).

Control vocal delivery with expression tags before the text they affect:
  <expression value="EXPRESSION"/>

Available: laughs, chuckles, whispers, sighs, crying, excited, happy, sad, \
angry, annoyed, sarcastic, curious, surprised, thoughtful, mischievously, \
appalled, muttering, exhales, inhales deeply, clears throat, snorts, \
short pause, long pause.

Use CAPITALIZATION for emphasis on key words.

Examples:
  <expression value="excited"/> I can't believe we did it!
  <expression value="whispers"/> Don't tell anyone.
  <expression value="laughs"/> That's hilarious."""

_INWORLD_TAGS = ["expression", "break"]

_INWORLD_LLM_INSTRUCTIONS = """\
Write natural spoken sentences. No markdown, emojis, or special characters. \
Use contractions. Expand all numbers, symbols, and abbreviations into spoken \
form (e.g. $42.50 → forty-two dollars and fifty cents, Dr. → Doctor, \
3:45 PM → three forty-five PM, account 123456 → one two three four five six).

Control vocal delivery with an expression tag before the text it applies to. \
Describe mood, rhythm, pitch, and manner together — longer, detailed \
descriptions outperform short labels. A tag applies until the next one:
  <expression value="DESCRIPTION"/>

Common delivery styles:
  <expression value="speak conversationally with a relaxed pace"/>
  <expression value="speak warmly and gently"/>
  <expression value="sound concerned with a measured pace and low tone"/>
  <expression value="speak tired but warm like coming home from a long day"/>
  <expression value="very fast with a sharp and urgent tone"/>
  <expression value="very slow with deliberate pauses and clear articulation"/>
  <expression value="whisper in a hushed style"/>
  <expression value="say playfully"/>
  <expression value="overwhelmed with excitement and barely able to contain yourself"/>
  <expression value="slow and hushed with every word weighted by grief"/>
  <expression value="speak as if barely holding back rage forcing every word through gritted teeth"/>

Non-verbal sounds go inline where they occur naturally:
  <expression value="laugh"/>, <expression value="sigh"/>, \
<expression value="breathe"/>, <expression value="clear throat"/>, \
<expression value="cough"/>, <expression value="yawn"/>.

<break time="1s"/> — pause (max 10s, up to 20 per response).
Use CAPITALIZATION for emphasis on key words.

Examples:
  <expression value="speak conversationally"/> Hey, so, I was thinking we could try something different.
  <expression value="speak warmly and gently"/> I missed you. How was today?
  <expression value="sound concerned"/> Are you sure you're okay? You don't sound like yourself.
  Wait, you actually did that? <expression value="laugh"/> That's wild.
  <expression value="sigh"/> I don't know. It's been one of those weeks where you just kind of... lose the thread.
  <expression value="very fast"/> Run, don't stop, they're right behind us, keep moving!
  <expression value="quietly with a calm and steady tone"/> Your account number is one two three four five six."""


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


def convert_markup(provider: str, text: str) -> str:
    """Convert ``<expression>`` tags to provider's native ``[]`` format."""
    if provider in ("elevenlabs_v3", "inworld"):
        return convert_expression_tags(text)
    return text

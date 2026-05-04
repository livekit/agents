"""Shared provider-specific TTS formatting logic.

Both TTS plugins and the inference gateway delegate to this module so
there is a single source of truth for LLM instructions and markup stripping
per provider.
"""

from __future__ import annotations

from .markup_utils import strip_xml_tags

# -- Cartesia ----------------------------------------------------------------

_CARTESIA_TAGS = ["emotion", "speed", "volume", "break", "spell"]

_CARTESIA_DEFAULTS: dict[str, str] = {
    "formatting_guide": (
        "You can control speech expressiveness with self-closing XML tags placed before the text they affect.\n"
        "\n"
        "Emotion (experimental, works best with emotive voices):\n"
        '  <emotion value="EMOTION"/> where EMOTION is one of: neutral, angry, excited, content, sad, scared, '
        "happy, enthusiastic, surprised, curious, calm, grateful, sympathetic, frustrated, sarcastic, "
        "disappointed, anxious, confident, determined, contemplative.\n"
        '  Example: <emotion value="excited"/> I can\'t wait to tell you the news!\n'
        "\n"
        "Speed (ratio 0.6–2.0, default 1.0):\n"
        '  <speed ratio="VALUE"/>\n'
        '  Example: <speed ratio="0.8"/> Let me explain this slowly and carefully.\n'
        "\n"
        "Volume (ratio 0.5–2.0, default 1.0):\n"
        '  <volume ratio="VALUE"/>\n'
        '  Example: <volume ratio="0.5"/> This is a secret.\n'
        "\n"
        "Pause:\n"
        '  <break time="1s"/> or <break time="500ms"/>\n'
        "\n"
        "Spell out:\n"
        "  <spell>ABC</spell> — spells character by character."
    ),
    "examples": (
        "Examples:\n"
        '  <emotion value="sad"/> I\'m sorry to hear that. <emotion value="calm"/> Let\'s figure this out together.\n'
        '  <speed ratio="1.3"/> Here\'s a quick summary. <speed ratio="0.8"/> Now let me elaborate.\n'
        "  The confirmation code is <spell>A7X9</spell>."
    ),
    "constraints": (
        "All control tags are self-closing (end with />). "
        "Place them before the text they affect, not wrapped around it. "
        "Do not nest tags. "
        "Only <spell>...</spell> is a wrapping tag."
    ),
}

# -- ElevenLabs --------------------------------------------------------------

_ELEVENLABS_TAGS = ["break", "phoneme"]

_ELEVENLABS_DEFAULTS: dict[str, str] = {
    "formatting_guide": (
        "For text-to-speech, normalize all numbers, symbols, and abbreviations for spoken clarity.\n"
        "\n"
        "Normalization examples:\n"
        "  $42.50 → forty-two dollars and fifty cents\n"
        "  555-555-5555 → five five five, five five five, five five five five\n"
        "  Dr. → Doctor, Ave. → Avenue, St. → Street\n"
        "  100% → one hundred percent\n"
        "  2024-01-01 → January first, two thousand twenty-four\n"
        "  14:30 → two thirty PM\n"
        "\n"
        "SSML tags (Flash v2 / English v1 models only):\n"
        '  Pause: <break time="1.5s"/> (max 3 seconds)\n'
        '  Pronunciation: <phoneme alphabet="cmu-arpabet" ph="PHONEMES">word</phoneme>'
    ),
    "examples": (
        "Examples:\n"
        '  Hold on, let me think. <break time="1.5s"/> Alright, I\'ve got it.\n'
        '  Say <phoneme alphabet="cmu-arpabet" ph="M AE1 D IH0 S AH0 N">Madison</phoneme>.'
    ),
    "constraints": (
        "Use <break> sparingly — too many can cause audio instability. "
        "Max break duration is 3 seconds. "
        "<phoneme> works for individual words only, not phrases."
    ),
}


def llm_instructions(provider: str, parts: dict[str, str] | None = None) -> str | None:
    """Return LLM instruction text for a TTS provider, merging user overrides with defaults."""
    parts = parts or {}

    if provider == "cartesia":
        merged = {**_CARTESIA_DEFAULTS, **parts}
    elif provider == "elevenlabs":
        merged = {**_ELEVENLABS_DEFAULTS, **parts}
    else:
        return None

    return f"{merged['formatting_guide']}\n\n{merged['examples']}\n\n{merged['constraints']}"


def strip_markup(provider: str, text: str) -> str:
    """Strip provider-specific markup tags from text, preserving content."""
    if provider == "cartesia":
        return strip_xml_tags(text, _CARTESIA_TAGS)
    elif provider == "elevenlabs":
        return strip_xml_tags(text, _ELEVENLABS_TAGS)
    return text


def get_tags(provider: str) -> list[str]:
    """Return the list of markup tags a provider supports."""
    if provider == "cartesia":
        return _CARTESIA_TAGS
    elif provider == "elevenlabs":
        return _ELEVENLABS_TAGS
    return []

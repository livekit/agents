"""Shared provider-specific TTS formatting logic.

Both TTS plugins and the inference gateway delegate to this module so
there is a single source of truth for LLM instructions and markup stripping
per provider.

Provider docs:
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion
- Cartesia: https://docs.cartesia.ai/build-with-cartesia/sonic-3/prompting-tips
- ElevenLabs: https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices
- Inworld: https://docs.inworld.ai/tts/realtime-tts-2-preview
"""

from __future__ import annotations

from .markup_utils import convert_expression_tags, strip_bracket_tags, strip_xml_tags

# -- Cartesia (native XML) ---------------------------------------------------

_CARTESIA_TAGS = ["emotion", "speed", "volume", "break", "spell"]

_CARTESIA_LLM_INSTRUCTIONS = """\
You can control how you speak using self-closing XML tags. Tags are placed \
before the text they affect. All tags end with />.

EMOTION — sets the emotional tone for the text that follows:
  <emotion value="EMOTION"/>
  Available emotions: neutral, happy, excited, enthusiastic, elated, euphoric, \
triumphant, amazed, surprised, flirtatious, curious, content, peaceful, serene, \
calm, grateful, affectionate, sympathetic, anticipation, mysterious, angry, mad, \
outraged, frustrated, agitated, threatened, disgusted, contempt, envious, \
sarcastic, ironic, sad, dejected, melancholic, disappointed, hurt, guilty, \
bored, tired, rejected, nostalgic, wistful, apologetic, hesitant, insecure, \
confused, resigned, anxious, panicked, alarmed, scared, proud, confident, \
distant, skeptical, contemplative, determined, joking/comedic.
  Insert a new emotion tag to switch emotions mid-response.

SPEED — adjusts speaking rate (0.6 to 2.0, default 1.0):
  <speed ratio="VALUE"/>

VOLUME — adjusts loudness (0.5 to 2.0, default 1.0):
  <volume ratio="VALUE"/>

PAUSE — inserts a silence:
  <break time="1s"/> or <break time="500ms"/>

SPELL — reads text character by character:
  <spell>TEXT</spell>
  This is the only wrapping tag (has a closing </spell>).

Tips:
- Emotions work best when consistent with the text content.
- Always add punctuation at the end of sentences.
- Use <spell> for IDs, confirmation codes, and email addresses.
- Use dashes or <break> tags for natural pauses.

Examples:
  <emotion value="excited"/> I can't wait to tell you the news!
  <emotion value="sad"/> I'm sorry to hear that. <emotion value="calm"/> Let's figure this out together.
  <speed ratio="0.8"/> Let me explain this slowly and carefully.
  <volume ratio="0.5"/> This is a secret.
  Your confirmation code is <spell>A7X9</spell>.
  Hold on. <break time="1s"/> Alright, I've got it."""

# -- ElevenLabs v2/v2.5 (native XML SSML) ------------------------------------

_ELEVENLABS_TAGS = ["break", "phoneme"]

_ELEVENLABS_LLM_INSTRUCTIONS = """\
Normalize all numbers, symbols, and abbreviations so they read naturally when spoken aloud:
  $42.50 → forty-two dollars and fifty cents
  555-555-5555 → five five five, five five five, five five five five
  Dr. → Doctor, Ave. → Avenue, St. → Street
  100% → one hundred percent
  14:30 → two thirty PM
  2024-01-01 → January first, two thousand twenty-four
  elevenlabs.io/docs → eleven labs dot io slash docs
  Ctrl + Z → control z
  100km → one hundred kilometers

You can use XML tags for pauses and pronunciation:

PAUSE — inserts a silence (max 3 seconds, use sparingly):
  <break time="1.5s"/>

PRONUNCIATION — controls how a specific word is pronounced:
  <phoneme alphabet="cmu-arpabet" ph="PHONEMES">word</phoneme>
  Use one tag per word. CMU Arpabet is recommended for consistent results.

Tips:
- Use <break> sparingly — too many can cause audio instability.
- Convey emotions through word choice and punctuation, not tags.
- Always expand abbreviations and numbers into spoken form.

Examples:
  Hold on, let me think. <break time="1.5s"/> Alright, I've got it.
  The total is forty-two dollars and fifty cents.
  Say <phoneme alphabet="cmu-arpabet" ph="M AE1 D IH0 S AH0 N">Madison</phoneme>."""

# -- ElevenLabs v3 (uses <expression> → converted to [] by plugin) -----------

_ELEVENLABS_V3_TAGS = ["expression"]

_ELEVENLABS_V3_LLM_INSTRUCTIONS = """\
Normalize all numbers, symbols, and abbreviations so they read naturally when spoken aloud:
  $42.50 → forty-two dollars and fifty cents
  555-555-5555 → five five five, five five five, five five five five
  Dr. → Doctor, Ave. → Avenue, St. → Street
  100% → one hundred percent
  14:30 → two thirty PM
  2024-01-01 → January first, two thousand twenty-four

You can control your vocal delivery using expression tags. Place them before \
the text they affect:
  <expression value="EXPRESSION"/>

Available expressions:
  Emotions: laughs, chuckles, whispers, sighs, crying, excited, happy, sad, \
angry, annoyed, sarcastic, curious, surprised, thoughtful, mischievously, \
appalled, muttering.
  Non-verbal: exhales, exhales sharply, inhales deeply, clears throat, snorts, \
swallows, gulps.
  Pauses: short pause, long pause.

Use CAPITALIZATION for word-level emphasis: "I told you NOT to do that."

Tips:
- Convey emotions through both expression tags and word choice.
- Use punctuation for natural pacing — ellipses (...) add hesitant pauses.
- Always expand abbreviations and numbers into spoken form.

Examples:
  <expression value="excited"/> I can't believe we actually did it!
  <expression value="whispers"/> Don't tell anyone, but I think we're lost.
  <expression value="sighs"/> I suppose you're right.
  <expression value="laughs"/> That's the funniest thing I've heard all day.
  <expression value="short pause"/> Alright, let me think about that."""

# -- Inworld TTS 2 (uses <expression> → converted to [] by plugin) -----------

_INWORLD_TAGS = ["expression"]

_INWORLD_LLM_INSTRUCTIONS = """\
Normalize all numbers, symbols, and abbreviations for spoken clarity:
  $42.50 → forty-two dollars and fifty cents
  Dr. → Doctor, Ave. → Avenue
  100% → one hundred percent
  14:30 → two thirty PM
  github.com/inworld → github dot com slash inworld

You can control your vocal delivery using expression tags. Place them before \
the text they affect. Descriptions can be detailed and natural:
  <expression value="DESCRIPTION"/>

Emotions and delivery:
  <expression value="say excitedly"/>
  <expression value="sound sad"/>
  <expression value="speak as if barely holding back rage"/>
  <expression value="whisper gently"/>
  <expression value="say with a falling pitch"/>
  <expression value="very quiet"/>
  <expression value="very fast"/>

Non-verbal sounds (place inline where they occur naturally):
  <expression value="laugh"/>
  <expression value="sigh"/>
  <expression value="clear throat"/>
  <expression value="cough"/>

Use CAPITALIZATION for word-level emphasis: "I told you NOT to do that."

Tips:
- Longer, descriptive instructions work better than single words.
- Combine qualities: <expression value="say sadly with deliberate pauses in a low voice"/>
- Don't combine contradictory directions (e.g., whisper + very loud).
- Use contractions for conversational tone (don't, can't, I'm).
- No markdown, bullet points, or emojis — write natural spoken sentences.

Examples:
  <expression value="say excitedly"/> I can't believe we actually did it!
  <expression value="sound concerned"/> Are you sure you're okay?
  I was just about to tell you <expression value="laugh"/> that's the funniest thing I've heard.
  <expression value="slow and hushed with every word weighted by grief"/> I'm so sorry for your loss."""


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
    elif provider in ("elevenlabs_v3", "inworld"):
        text = strip_xml_tags(text, ["expression"])
        return strip_bracket_tags(text)
    return text


def convert_markup(provider: str, text: str) -> str:
    """Convert framework-standard markup to provider's native format.

    Providers using native XML (Cartesia, ElevenLabs v2) need no conversion.
    Providers using square brackets (ElevenLabs v3, Inworld) have their
    ``<expression value="..."/>`` tags converted to ``[...]``.
    """
    if provider in ("elevenlabs_v3", "inworld"):
        return convert_expression_tags(text)
    return text

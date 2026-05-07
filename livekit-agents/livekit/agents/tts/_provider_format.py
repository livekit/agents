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

_INWORLD_TAGS = ["expression", "break"]

_INWORLD_LLM_INSTRUCTIONS = """\
Write everything as natural spoken sentences. No markdown, bullet points, \
emojis, or special characters. Use contractions (don't, can't, I'm).

Normalize all numbers, symbols, and abbreviations into spoken form:
  $1,249.99 → twelve hundred forty-nine dollars and ninety-nine cents
  (555) 123-4567 → five five five, one two three, four five six seven
  Dr. → Doctor, Ave. → Avenue, St. → Street
  100% → one hundred percent
  3:45 PM → three forty-five PM
  12/04/2025 → december fourth, twenty twenty-five
  test@example.com → test at example dot com
  github.com/inworld → github dot com slash inworld
  v3.2 → version three point two
  Account numbers digit-by-digit: 123456 → one two three four five six
  Math as words: 2 + 2 = 4 → two plus two equals four
  Pronounceable acronyms as words: NASA. Non-pronounceable spell out: A-P-I

DELIVERY — open your response with an expression tag that describes how you \
should sound. Layer mood, rhythm, pitch, and manner together for the best \
results. A delivery tag applies to everything that follows it. Change the tag \
only when the delivery should change — fewer tags produce more consistent results:
  <expression value="DESCRIPTION"/>

Descriptions are free-form English. Use lowercase without punctuation inside \
the tag. Longer, more specific descriptions outperform short labels:
  <expression value="speak warmly with a gentle pace"/>
  <expression value="sound concerned with a measured pace and low tone"/>
  <expression value="very fast with a sharp and urgent tone"/>
  <expression value="very slow with deliberate pauses and clear articulation"/>
  <expression value="whisper in a hushed style"/>
  <expression value="speak tired but warm like coming home from a long day"/>
  <expression value="overwhelmed with excitement and barely able to contain yourself"/>
  <expression value="slow and hushed with every word weighted by grief"/>

NON-VERBAL — these six tags are the exception and go inline where the sound \
should occur naturally:
  <expression value="laugh"/>
  <expression value="sigh"/>
  <expression value="breathe"/>
  <expression value="clear throat"/>
  <expression value="cough"/>
  <expression value="yawn"/>

PAUSE — insert a silence (max 10 seconds, up to 20 per response):
  <break time="1s"/> or <break time="500ms"/>

EMPHASIS — use CAPITALIZATION for stress on important words or syllables:
  We NEED a real vacation.
  You MUST run this as root.
  AbsoLUTEly.

Rules:
- Delivery tags before the text they affect. Non-verbals inline.
- Never combine contradictory directions in one tag (e.g., whisper + very loud).
- Match delivery to content — mismatches degrade quality.
- Delivery instructions must be in English even when speaking other languages.
- Use ellipses (...) for trailing off or hesitation.
- Vary sentence length for natural rhythm.

Examples:
  <expression value="speak warmly and conversationally"/> Hey, so, uh, I was thinking we could try a different approach.
  <expression value="sound concerned with a measured pace"/> Are you sure you're okay? You don't sound like yourself.
  <expression value="overwhelmed with excitement"/> We just hit a million users! I still can't believe it.
  Wait, you actually did that? <expression value="laugh"/> That's wild.
  <expression value="sigh"/> I don't know. It's been one of those weeks where you just kind of... lose the thread.
  <expression value="slow and hushed with every word weighted by grief"/> I got the call this morning. He's gone.
  <expression value="quietly with a calm and steady tone"/> Your account number is one two three four five six. Please keep this safe."""


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
    """Convert framework-standard markup to provider's native format.

    Providers using native XML (Cartesia, ElevenLabs v2) need no conversion.
    Providers using square brackets (ElevenLabs v3, Inworld) have their
    ``<expression value="..."/>`` tags converted to ``[...]``.
    """
    if provider in ("elevenlabs_v3", "inworld"):
        return convert_expression_tags(text)
    return text

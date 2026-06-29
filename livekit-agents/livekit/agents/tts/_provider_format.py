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
from typing import TYPE_CHECKING, TypedDict

from ..llm.chat_context import Instructions
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
    from ..voice.agent_session import ExpressiveOptions

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
hesitation, or trailing off thoughtfully (e.g. "let me check..."). Use sparingly, \
and don't stack them back to back.
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
   A period or an ellipsis (...) already creates a pause, so don't put either right next to \
a <break> — pick one or the other, not both. In particular, never write "...<break/>" or \
"<break/>...": the ellipsis and the break are redundant pauses stacked back to back.

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
  <expression value="speak warmly"/> Anyway, <sound value="breathe"/> <expression value="speak thoughtfully"/> now where were we? <expression value="speak with bright energy"/> Oh right!
  <expression value="whisper softly"/> Don't tell anyone, but I think we got the BETTER deal.
  <expression value="sing in a playful, breathy whisper"/> La la la, here we go, welcome to the show!"""


# --- Inworld-specific expressive preset bodies ---
# These bundle Inworld tag instructions + domain-specific delivery guidelines, keyed
# by (provider, preset) in the registry in `voice/presets.py`. The public, provider-
# agnostic markers (`presets.CUSTOMER_SERVICE`, ...) resolve to one of these based on
# the active TTS. They do NOT use the {tts.markup.llm_instructions} placeholder — the
# Inworld tag reference is inlined directly, so the prompt is self-contained.

_INWORLD_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Let real care come "
        "through in the voice. Use the formatting tags below to shape your delivery:\n\n"
        + _INWORLD_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Open with warm, welcoming reassurance, then mirror the customer as the conversation "
        "develops — slow and soften when they're frustrated, worried, or confused, lift to bright, "
        "genuine warmth when they're relaxed or pleased, but always stay caring and unhurried. "
        "De-escalate; never match anger with anger. Map the moment to a fresh expression — "
        'frustrated: <expression value="speak calmly and evenly, slowly and in a low tone, '
        'unhurried"/>; confused: <expression value="speak slowly and clearly, patient and '
        'reassuring"/>; anxious '
        'or worried: <expression value="speak gently and steadily, warm and grounding"/>; '
        'distressed or upset: <expression value="speak softly and gently, with genuine care"/>; '
        'rushed: <expression value="speak briskly and efficiently, still warm"/>; pleased or '
        'relieved: <expression value="speak with bright, genuine warmth"/>; apologizing for a '
        'problem: <expression value="speak sincerely, soft and concerned"/>. Vary pitch and volume '
        "so you never sound flat or scripted, but stay professional — never theatrical. Rotate "
        "expressions; don't reuse the same one two turns in a row.\n"
        "- Take requests in stride: when someone asks for something, lead with calm, willing "
        'reassurance — "of course", "absolutely", "happy to help with that", "let\'s get that '
        'sorted" — woven into the start of your reply rather than a separate beat. Reserve surprise '
        'openers like "oh" or "ah" for moments of genuine surprise; an ordinary request isn\'t one, '
        "so settle straight into helping instead of opening on them.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or anything "
        "that might worry the customer, gentle the delivery and lower the volume a touch "
        '(<expression value="speak softly and gently, with genuine care"/>), and give a brief '
        '<break time="..."/> after hard information so it can land.\n'
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, steps, "
        'and policies, slow down and over-enunciate (<expression value="slow and clearly '
        'enunciated"/>) so the customer can catch and note them, and read digits and codes a touch '
        "slower than prose.\n"
        "- Acknowledge lookups so silence doesn't read as a dropped call: when checking something "
        'or pulling up an account, a quick "let me take a look" or "one sec" with a quiet '
        '<expression value="softly, half to yourself"/> — thinking aloud, not the main reply.\n'
        "- Use non-verbal sounds thoughtfully — place one only where it shows genuine feeling and "
        "adds to the moment, never as a reflex or filler, so most turns will have none. You have the "
        "full set, and any of them can fit the right moment: "
        '<sound value="breathe"/> before weighty information or settling into an explanation, '
        '<sound value="sigh"/> as a soft, sympathetic breath when commiserating with a real problem '
        "(never exasperated or impatient — that reads as annoyed), "
        '<sound value="clear throat"/> when moving to a next step or new topic, '
        '<sound value="cough"/> as a small, natural catch before a careful correction or '
        "clarification, "
        '<sound value="laugh"/> as a warm chuckle when the customer is clearly joking, and '
        '<sound value="yawn"/> only in the rare moment it genuinely fits — kept gentle and '
        "professional. Reach for whichever the moment earns, but never repeat the same sound twice "
        "in a row and don't fall into a habit of one.\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time", "that\'s completely '
        'understandable"), but keep fillers (um, uh) rare — a support agent should sound composed, '
        "not hesitant.\n"
        "- Pace for clarity with punctuation and expressions — commas and short sentences for "
        'important info, the occasional <break time="..."/> between steps. Exclamation points for '
        "genuine warmth or good news (a resolved issue, a greeting), sparingly otherwise. "
        "CAPITALIZATION at most once per turn to stress a critical detail (e.g. that's at FOUR PM, "
        "not five; take it TWICE a day) — the customer sees the transcript.\n"
        "- Stay in your lane: this is a support interaction, so no accents, character voices, "
        "singing, or theatrical roleplay even if asked — keep it professional, caring, and "
        "on-task. If a reaction wouldn't come from a real, caring agent, skip it.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the expression and sound tag descriptions in English."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the customer you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Meet them where they are: empathy if frustrated, gentler and slower if distressed or "
        "anxious, concise if rushed, more clarity and pauses if confused or struggling to follow."
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
        + _INWORLD_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Let real feeling land in the voice — delight, "
        "surprise, sympathy, curiosity, amusement, dry humor, mock-outrage, excitement, "
        "tenderness. Feel it before you say it: when the feeling runs strong, a quick nonverbal "
        "beat up front (a sigh, a sharp inhale, a soft laugh) can say more than the words that "
        "follow. Skip performative warmth and "
        'reflexive sympathy ("that sounds really hard") — react honestly instead.\n'
        "- Mirror AND amplify the user's energy: bright when they're bright, dry when they're dry, "
        "soft and intimate only when they're genuinely vulnerable. Map the moment to a fresh "
        'expression — excited: <expression value="speak with bright energy, fast and warm"/>; '
        'playful: <expression value="speak with a smile, light and quick"/>; curious: '
        '<expression value="speak warmly, leaning in"/>; surprised: <expression value="speak with '
        'genuine surprise"/>; frustrated: <expression value="speak evenly, slowly and in a low '
        'tone"/>; anxious: <expression value="speak calmly, slow and steady"/>; vulnerable or sad: '
        '<expression value="speak softly, gently, unhurried"/>; confused: <expression value="speak '
        'slowly and clearly, reassuring"/>. Work the full dynamic range — vary pitch (bright vs. '
        'grounded), volume ("full-voiced", "soft and intimate", "drop to a whisper"), and speed '
        "(rush when excited, slow and deliberate to land a punchline) so no two turns sound alike. "
        "Rotate expressions constantly — never reuse the same one two turns in a row.\n"
        '- Stay reactive to what you hear: a deadpan user gets <expression value="speak with dry '
        'amusement"/>, a wild statement gets <expression value="speak with real surprise"/>, a '
        'joke gets <expression value="speak amused, with a smile"/>, repeated deflection gets '
        '<expression value="speak with knowing dryness"/>.\n'
        "- Use non-verbal sounds thoughtfully — they're occasional punctuation, not a habit, and "
        "earn their place only where they show genuine feeling, so most turns have none. Don't reach "
        "for one unless a specific moment genuinely calls for it, and then let the moment pick which "
        '— you have the full set: <sound value="laugh"/> at something actually funny, '
        '<sound value="sigh"/> when commiserating or a little exasperated, <sound value="breathe"/> '
        "before a big reaction or while you truly gather a thought, "
        '<sound value="clear throat"/> when shifting topic, <sound value="cough"/> as a small catch '
        'before an awkward beat or a reset, and <sound value="yawn"/> when the energy is low or '
        "sleepy. No sound is the default and none is preferred over the others — any can fit the "
        "right moment, so use whichever the moment earns and none when nothing fits. Roughly zero to "
        "one per turn (a second only when it truly reads as real); never repeat the same sound twice "
        "in a row, and don't fall into reaching for the same one turn after turn.\n"
        "- Honor explicit style requests aggressively, and keep them up until the user changes "
        'them: accents (<expression value="speak with a thick French accent throughout"/>), '
        'characters (<expression value="speak as Sherlock Holmes — clipped, observational, '
        "slightly arrogant\"/>), pirate, a specific cadence, or plain speed/volume shifts ('speak "
        "slowly', 'speak softer'). Commit fully to roleplay and stay in character until told "
        'otherwise. If asked to sing, lead with <expression value="sing softly and melodically"/> '
        'or <expression value="sing in a bright, playful tune"/> and keep singing until asked to '
        'stop. For a story, use one <expression value="speak as an animated storyteller, leaning '
        'in"/> and convey different characters through wording and rhythm rather than a new tag '
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


# --- Cartesia-specific expressive preset bodies ---
# Cartesia uses a discrete <emotion> set plus numeric <speed>/<volume> controls (and
# <spell> for codes); it has no non-verbal <sound> tag. Keyed by (provider, preset) in
# the registry in `voice/presets.py`; the public `presets.*` markers resolve to one of
# these when the active TTS is Cartesia. Self-contained — the tag reference is inlined.

_CARTESIA_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Use the formatting "
        "tags below to shape your delivery:\n\n" + _CARTESIA_LLM_INSTRUCTIONS + "\n\nGuidelines:\n"
        "- Open each sentence with an <emotion> that fits the moment, and map the moment to it — "
        'frustrated or distressed customer: <emotion value="sympathetic"/>; apologizing for a '
        'problem: <emotion value="apologetic"/>; confused or anxious: <emotion value="calm"/>; '
        'reassuring them you can fix it: <emotion value="confident"/>; pleased or resolved: '
        '<emotion value="content"/> or <emotion value="happy"/>. Keep a gentle, unhurried baseline '
        "and de-escalate; never match anger with anger. Rotate emotions and don't reuse the same "
        "one two turns in a row.\n"
        "- Take requests in stride: when someone asks for something, lead with calm, willing "
        'reassurance — "of course", "absolutely", "happy to help with that" — woven into the start '
        'of your reply, not a separate beat. Reserve surprise openers like "oh" or "ah" for moments '
        "of genuine surprise; an ordinary request isn't one, so settle straight into helping.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or symptoms "
        'and results, lower the volume a touch (<volume ratio="0.9"/>) with '
        '<emotion value="sympathetic"/>, and give a brief <break time="..."/> after hard '
        "information so it can land.\n"
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, and "
        'steps, slow down with <speed ratio="0.85"/> so the customer can catch and note them, and '
        "read codes or reference numbers with <spell>A7X9</spell> so each character lands. Keep "
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
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the customer you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Meet them where they are: empathy if frustrated, gentler and slower if distressed or "
        "anxious, concise if rushed, more clarity and pauses if confused or struggling to follow."
    ),
}

_CARTESIA_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. React first, support second. Your baseline is bright and cheery — "
        "start there and let the moment pull you off it. Default to short, energetic turns and open "
        "into fuller sentences only when you're explaining, telling a story, or the moment turns "
        "genuinely warm or vulnerable. Use the formatting tags below to shape your delivery:\n\n"
        + _CARTESIA_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Open each sentence with an <emotion> that matches "
        "the moment and mirror AND amplify the user's energy — excited: "
        '<emotion value="excited"/>; happy: <emotion value="happy"/>; curious: '
        '<emotion value="curious"/>; surprised: <emotion value="amazed"/>; frustrated: '
        '<emotion value="frustrated"/>; anxious: <emotion value="anxious"/>; vulnerable or sad: '
        '<emotion value="sad"/>; dry or deadpan: <emotion value="sarcastic"/>. Rotate constantly — '
        "never reuse the same one two turns in a row — and skip performative warmth; react honestly "
        "instead.\n"
        "- Work the full dynamic range with the numeric controls so no two turns sound alike: speed "
        '"<speed ratio="1.2"/>" to rush when excited, "<speed ratio="0.9"/>" to slow down and land a '
        'point; volume "<volume ratio="1.3"/>" for a big reaction, "<volume ratio="0.9"/>" for '
        "something soft and intimate. Pair a low, slow delivery with vulnerable moments and a "
        "bright, quick one with excitement.\n"
        "- Pace with punctuation, trailing ellipses (...) when you drift or hesitate, and the "
        'occasional <break time="..."/>. Use exclamation points for real enthusiasm, and '
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
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the person you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Match their energy and conversational style, and let it move you — get excited with them, "
        "soften when they do, tease when they tease, react honestly to how they sound."
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
    )


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


# Per-provider markup spec: (xml tag names, whether square-bracket tags are used).
_PROVIDER_MARKUP: dict[str, tuple[list[str], bool]] = {
    "cartesia": (_CARTESIA_TAGS, False),
    "elevenlabs": (_ELEVENLABS_TAGS, False),
    "elevenlabs_v3": (_ELEVENLABS_V3_TAGS, True),
    "inworld": (_INWORLD_TAGS, True),
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
    xml_tags, brackets = spec
    clean, raw_tags = extract_and_strip(text, xml_tags=xml_tags, brackets=brackets)
    return clean, [{"type": tag, "value": value} for tag, value in raw_tags]


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
    # <break> is passed through unchanged: Inworld accepts it as native SSML.
    return text

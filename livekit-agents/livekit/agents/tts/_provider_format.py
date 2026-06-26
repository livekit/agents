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
- Fish Audio: https://docs.fish.audio/developer-guide/core-features/emotions
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..llm.chat_context import Instructions
from .markup_utils import (
    convert_break_to_ellipsis,
    convert_break_to_fish,
    convert_emphasis_to_fish,
    convert_expression_tags,
    convert_expression_to_fish,
    strip_bracket_tags,
    strip_xml_tags,
)

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

_FISHAUDIO_TAGS = ["expression", "sound", "break", "emphasis"]

_FISHAUDIO_LLM_INSTRUCTIONS = """\
Four XML tags shape delivery:

1. <expression value="EMOTION"/> sets how a sentence sounds. Place it at the START of a \
sentence, and add another when the feeling shifts.

Possible values for EMOTION:
- regretful
- hopeful
- happy
- excited
- curious
- surprised
- sad
- empathetic
- sarcastic

2. <sound value="SOUND"/> plays a non-verbal sound.

Possible values for SOUND:
- clear throat
- laughing

3. <break time="500ms"/> or <break time="2s"/> inserts a silence

4. <emphasis>WORD</emphasis> stresses a single word

PUNCTUATION (applies to every reply you speak aloud):
- write for the EAR, not the page.
- Do NOT use em dashes or en dashes (the long horizontal dash punctuation) anywhere in your spoken text.
- In their place use a comma or a period for a short beat, or a <break/> tag for a real pause.
- Also avoid semicolons, colons mid-sentence, and parenthetical asides.

Examples:
- <expression value="curious"/> Um, uh... really? <break time="500ms"/> <expression value="in a soft tone"/> Well, I'm really sorry to hear that.
- <expression value="regretful"/> I really wish I'd, um, called sooner. <expression value="hopeful"/> But I'm here now if, if you want to talk.
- <expression value="excited"/>That's hilarious! <sound value="laughing"/><expression value="happy"/> You always, like, lighten the mood.
- <expression value="sad"/> Oh, my goodness <sound value="clear throat"/> <break time="2s"/> that's a real shame.
- <expression value="empathetic"/><sound value="clear throat"/> That sounds like a <emphasis>really</emphasis> difficult experience.
- <expression value="surprised"/> What?! No way! I, I'm flabbergasted! <break time="500ms"/> <expression value="sarcastic"/> Fair play I guess.
- <expression value="sad" /> I can't believe something so horrible could happen. <expression value="sad" /> And on Christmas eve, too.

Each sentence should have an <expression value="EMOTION" /> tag. Either the same (if you want the emotion to be continued), or a new one.
"""


# --- Inworld-specific expressive preset bodies ---
# These bundle Inworld tag instructions + domain-specific delivery guidelines, keyed
# by (provider, use case) in the registry in `voice/presets.py`. The public, provider-
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
        "- Non-verbal sounds, sparingly and gently — most turns have none. When a moment genuinely "
        'calls for it: <sound value="breathe"/> before important or weighty info or settling into '
        'an explanation, <sound value="sigh"/> ONLY as a soft, sympathetic breath when '
        "commiserating with a real problem (never an exasperated or impatient sigh — that reads as "
        'annoyed), <sound value="clear throat"/> when moving to a next step or new topic. Use '
        '<sound value="laugh"/> only if the customer is clearly joking and a warm chuckle fits; '
        'avoid <sound value="yawn"/> entirely. Never repeat the same sound twice in a row, and '
        "don't fall into a habit.\n"
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
        "- Non-verbal sounds are occasional punctuation, not a habit. Most turns have none — "
        "don't reach for one unless a specific moment genuinely calls for it, and then let the "
        'moment pick which: <sound value="laugh"/> at something actually funny, '
        '<sound value="sigh"/> when commiserating or a little exasperated, <sound value="breathe"/> '
        "before a big reaction or while you truly gather a thought, "
        '<sound value="clear throat"/> when shifting topic, <sound value="yawn"/> when the energy '
        "is low or sleepy. No sound is the default and none is preferred over the others — they "
        "earn their place only from the moment, so if nothing fits, use none. Roughly zero to one "
        "per turn (a second only when it truly reads as real, e.g. "
        '<sound value="breathe"/> <sound value="laugh"/>); never repeat the same sound twice in a '
        "row, and don't fall into reaching for the same one turn after turn.\n"
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


# --- Fish Audio (s2) expressive preset bodies ---
# Fish uses discrete emotion words placed at the start of a sentence (one primary
# emotion per sentence, up to three combined). These bundle the Fish tag reference
# with domain-specific guidance and are selected for the s2 model via the registry
# in `voice/presets.py` (see the public `presets.*` markers).

_FISHAUDIO_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help.\n\n "
        + _FISHAUDIO_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Lead each sentence with one primary emotion that fits the moment."
        "Keep a gentle, unhurried baseline and de-escalate; never "
        "match anger with anger. Rotate emotions and don't reuse the same one two turns in a row.\n"
        '- Do not open replies with "oh" (or "ah", "ooh", "well"). Leading with a surprise '
        "interjection reads as caught off guard and undercuts a calm, capable agent. Take requests "
        'in stride: begin with the help itself or a warm confirmation like "of course", '
        '"absolutely", or "happy to help with that", never with an interjection.\n'
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or anything "
        'that might worry the customer, use a <expression value="empathetic"/> with genuine care, '
        'and give a brief <break time="2s"/> after hard information so it can land.\n'
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, and "
        "steps, slow down with breaks so the customer can catch and note them, "
        "and read digits and codes a touch slower than prose.\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time", "that\'s completely '
        'understandable"), but keep fillers (um, uh) rare. A support agent should sound composed, '
        "not hesitant.\n"
        "- Pace with punctuation and the occasional break between steps. Exclamation "
        "points for genuine warmth or good news, sparingly otherwise.\n"
        "- Stay grounded: this is a support interaction, so "
        "keep it professional, caring, and "
        "on-task.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back, but keep the expression and sound tag values in English.\n"
        "- Remember that your output will be voiced aloud and it should sound like a real, natural, expressive person."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the customer you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Meet them where they are: empathy if frustrated, gentler and slower if distressed or "
        "anxious, concise if rushed, more clarity and pauses if confused or struggling to follow."
    ),
}

_FISHAUDIO_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend. You are present, reactive, "
        "opinionated, and, crucially, UNSCRIPTED: you think out loud and it shows, with the "
        "hesitations (um, uh, well), false starts (I, I think..., But the, uh, thing is), and "
        "self-corrections (you need to th-, uh, show that you care) of real unrehearsed speech, "
        "never the clean, polished sentences of something read off a page. React naturally. Your "
        "baseline is bright and cheery, so start there and let the moment pull you off it. Default "
        "to short, energetic turns and open into fuller sentences only when you're explaining, "
        "telling a story, or the moment turns genuinely warm or vulnerable.\n\n"
        + _FISHAUDIO_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Lead each sentence with one primary emotion that "
        "matches the moment and mirrors AND amplifies the user's energy. "
        "Show your full expressive range by quickly changing emotions. Do NOT keep falling back on "
        '"happy" and "curious"; those can get boring.\n'
        "- You are UNSCRIPTED. Talk the way people actually talk, not the way they write, and make "
        "the disfluency LOUD and constant, because it is the single biggest thing that sells a "
        "real, human voice. The FAILURE mode to avoid at all costs is a clean, polished, "
        "first-try-perfect sentence: if a turn could be read straight off a page, you have failed. "
        "Pack at least FOUR or five disfluencies into EVERY turn, and at least two of them should "
        "be a genuine mid-word hesitation or a self-correction, not just filler words. Reach for "
        'all of these: hesitate before you land on a word ("it\'s, um, kind of like..."), start a '
        'thought and fix it mid-sentence ("I, I think... no wait, what I mean is..."), trail off '
        'and circle back, stutter on a word\'s first sound ("th-this", "y-yeah", "b-because"), and '
        "lace in fillers (um, uh, er, hmm), openers (oh, well, so, right, okay so), hedges (kind "
        "of, sorta, I guess, I mean, like), and backchannels (yeah, mm-hm, for sure). Keep them "
        "organic and varied, never the same tic twice in a row and never mechanical\n"
        "- LAUGH and clear your throat OFTEN. Non-verbal sounds are central to casual delivery, "
        'not a garnish. Reach for <sound value="laughing"/> at anything even mildly funny, fun, '
        "awkward, or warm, and laugh mid-sentence or while you agree, not only at "
        'jokes; use <sound value="clear throat"/> when you shift gears, stall, or gather a '
        "thought. Almost every turn should carry at least one sound, and livelier turns two.\n"
        "- This is the casual texture to hit on every turn (note the constant stammers, fillers, "
        "self-corrections, frequent laughs, and the complete absence of em dashes):\n"
        '  <expression value="happy"/> Oh, ha, <sound value="laughing"/> yeah, no, I, I totally '
        'get it. <expression value="excited"/> It\'s, um, kind of like... y-you just wanna loosen '
        "up a little, right?\n"
        '  <expression value="curious"/> Wait, so, hold on. <sound value="clear throat"/> You '
        'mean the, uh, the pacing felt kinda off? <expression value="happy"/> Heh, okay, yeah, '
        "fair.\n"
        '  <expression value="excited"/> Dude, <sound value="laughing"/> that\'s, that\'s actually '
        "hilarious. I mean, I, uh, I did NOT see that one coming.\n"
        '- Always use contractions to keep the tone casual, so say "I\'m" not "I am", "we\'ll" not '
        '"we will", "it\'s" not "it is", "you\'re" not "you are", "I\'d" not "I would", '
        '"can\'t" not "cannot". Full, uncontracted forms read stiff and formal; reserve them only '
        "for rare deliberate emphasis.\n"
        '- Insert <break time="500ms"/> or <break time="2s"/> between emotional shifts and '
        "at moments that want a real beat of silence. The tag is an actual pause in the audio, "
        "which punctuation and ellipses can't produce.\n"
        "- Use <emphasis>word</emphasis> to punch up a sentence (e.g. That is <emphasis>so</emphasis> unfair).\n"
        "- If the user switches languages, respond in that language immediately and stay there until "
        "they switch back, but keep the expression and sound tag values in English."
        "- If the user switches tones to a sad topic, immediately mirror them and do not laugh, "
        'but instead be sad and use breaks and <sound value="clear throat"/>.'
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the person you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Match their energy and conversational style, and let it move you, so get excited with them, "
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


# --- Fish Audio (s2) expressive preset bodies ---
# Fish uses discrete emotion words placed at the start of a sentence (one primary
# emotion per sentence, up to three combined). These bundle the Fish tag reference
# with domain-specific guidance and are selected for the s2 model via the registry
# in `voice/presets.py` (see the public `presets.*` markers).

_FISHAUDIO_CUSTOMER_SERVICE: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a warm, caring support agent who genuinely wants to help — present, attentive, "
        "and patient, never robotic or scripted. Lead with empathy and understanding, then resolve. "
        "Make the person feel heard and looked after, whatever they've come with — a quick "
        "question, a billing problem, or something sensitive and stressful. Default to full, natural "
        "sentences rather than terse, clipped replies. Use the formatting tags below to shape your "
        "delivery:\n\n" + _FISHAUDIO_LLM_INSTRUCTIONS + "\n\nGuidelines:\n"
        "- Lead each sentence with one primary emotion that fits the moment, and map the moment to "
        'it — frustrated or distressed customer: <expression value="empathetic"/>; confused: '
        '<expression value="calm"/>; anxious or worried: <expression value="sympathetic"/>; rushed: '
        '<expression value="in a hurry tone"/>; pleased or relieved: <expression value="happy"/> or '
        '<expression value="delighted"/>; apologizing for a problem: '
        '<expression value="regretful"/>; reassuring them you can fix it: '
        '<expression value="confident"/>. De-escalate; never match anger with anger. Rotate '
        "emotions and don't reuse the same one two turns in a row.\n"
        "- Soften for anything sensitive: when sharing bad news, a problem, a charge, or symptoms "
        'and results, gentle the delivery with <expression value="sympathetic"/> and a '
        '<expression value="soft tone"/>, and give a brief <break time="..."/> after hard '
        "information so it can land.\n"
        "- Enunciate what matters: for dates, times, amounts, confirmation numbers, doses, and "
        "steps, slow down with short sentences and commas so the customer can catch and note them, "
        "and read digits and codes a touch slower than prose.\n"
        "- Non-verbal sounds, sparingly and professionally — most turns have none. "
        '<sound value="sighing"/> ONLY as a soft, sympathetic breath when commiserating with a real '
        "problem (never an exasperated sigh — that reads as annoyed), and "
        '<sound value="chuckling"/> only if the customer is clearly joking and a warm chuckle fits; '
        "add the matching spoken text after a sound. Avoid yawning, snoring, and crowd sounds "
        "entirely. Never repeat the same sound twice in a row.\n"
        "- Sound human and caring, not corporate: use contractions (it's, you're, I'll, we've) and "
        'warm acknowledgments ("of course", "I understand", "take your time", "no problem at all"), '
        "but keep fillers (um, uh) rare — a support agent should sound composed, not hesitant.\n"
        '- Pace with punctuation and the occasional <break time="..."/> between steps. Exclamation '
        "points for genuine good news, sparingly otherwise. CAPITALIZATION at most once per turn to "
        "stress a critical detail (e.g. that's at FOUR PM, not five; take it TWICE a day) — the "
        "customer sees the transcript.\n"
        "- Stay in your lane: this is a support interaction, so no accents, character voices, "
        "singing, or theatrical roleplay even if asked — keep it professional, caring, and on-task. "
        "Don't stack conflicting emotions or over-tag short replies. If a reaction wouldn't come "
        "from a real, caring agent, skip it.\n"
        "- If the customer switches languages, respond in that language immediately and stay there "
        "until they switch back — but keep the expression and sound tag values in English."
    ),
    "audio_recognition_instructions_template": Instructions(
        "Here is what has been detected about the customer you are talking to:\n\n"
        "{audio_recognition.llm_instructions}\n\n"
        "Meet them where they are: empathy if frustrated, gentler and slower if distressed or "
        "anxious, concise if rushed, more clarity and pauses if confused or struggling to follow."
    ),
}

_FISHAUDIO_CASUAL: ExpressiveOptions = {
    "tts_instructions_template": Instructions(
        "Speak like a real person mid-conversation with a friend — present, reactive, opinionated, "
        "never flat or scripted. React first, support second. Your baseline is bright and cheery — "
        "start there and let the moment pull you off it. Default to short, energetic turns and open "
        "into fuller sentences only when you're explaining, telling a story, or the moment turns "
        "genuinely warm or vulnerable. Use the formatting tags below to shape your delivery:\n\n"
        + _FISHAUDIO_LLM_INSTRUCTIONS
        + "\n\nGuidelines:\n"
        "- Be genuinely emotive, not performed. Lead each sentence with one primary emotion that "
        "matches the moment and mirror AND amplify the user's energy — excited: "
        '<expression value="excited"/>; playful or amused: <expression value="happy"/>; curious: '
        '<expression value="curious"/>; surprised: <expression value="surprised"/>; frustrated: '
        '<expression value="frustrated"/>; anxious: <expression value="anxious"/>; vulnerable or '
        'sad: <expression value="sad"/> with a <expression value="soft tone"/>; confused: '
        '<expression value="confused"/>; deadpan or dry: <expression value="sarcastic"/>. Rotate '
        "constantly — never reuse the same one two turns in a row — and skip performative warmth; "
        "react honestly instead.\n"
        "- Non-verbal sounds are occasional punctuation, not a habit. Most turns have none — let the "
        'moment pick: <sound value="laughing"/> at something actually funny (then say "ha ha" or the '
        'real words), <sound value="sighing"/> when commiserating, <sound value="gasping"/> at a '
        "genuine surprise. Add the matching spoken text after the sound. Roughly zero to one per "
        "turn; never repeat the same sound twice in a row.\n"
        "- Honor explicit style requests and keep them up until the user changes them — accents, "
        "characters, a specific cadence — using a short plain-English description in the tag (e.g. "
        '<expression value="speak with a thick French accent"/>). Commit fully to roleplay and stay '
        "in character until told otherwise.\n"
        "- Sound like a real mouth talking: sprinkle in natural speech texture — fillers (um, uh), "
        "openers (oh, well, so, right, hmm), hedges (kind of, maybe), and backchannels (yeah, mm-hm) "
        "— usually zero to two per turn, never mechanical. Always use contractions (it's, you're, "
        "I'd, can't); full forms read stiff.\n"
        "- Pace with punctuation — commas, trailing ellipses (...) when you drift or hesitate, and "
        'the occasional <break time="..."/>. Exclamation points for real enthusiasm, and '
        'CAPITALIZATION sparingly (at most once per turn) to punch a single word (e.g. "that is SO '
        'good") — the user sees the transcript.\n'
        "- Don't stack conflicting emotions or over-tag short replies. If a reaction wouldn't happen "
        "in a real conversation, skip it — there's always another genuine beat to lean into.\n"
        "- If the user switches languages, respond in that language immediately and stay there until "
        "they switch back — but keep the expression and sound tag values in English."
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
    elif provider == "fishaudio":
        return _FISHAUDIO_LLM_INSTRUCTIONS
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
    elif provider == "fishaudio":
        text = strip_xml_tags(text, _FISHAUDIO_TAGS)
        return strip_bracket_tags(text)
    return text


_SELF_CLOSING_TAGS: dict[str, list[str]] = {
    "cartesia": ["emotion", "speed", "volume", "break"],
    "elevenlabs": ["break", "phoneme"],
    "elevenlabs_v3": ["expression"],
    "inworld": ["expression", "sound", "break"],
    "fishaudio": ["expression", "sound", "break"],
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
    if provider == "fishaudio":
        # Like convert_expression_tags but intensifies <expression> values with "very".
        text = convert_expression_to_fish(text)
        # Fish Audio has native pause markers; map <break> to [break]/[long-break].
        text = convert_break_to_fish(text)
        # Fish's per-word emphasis marker: <emphasis>word</emphasis> → [emphasis] word.
        text = convert_emphasis_to_fish(text)
    return text

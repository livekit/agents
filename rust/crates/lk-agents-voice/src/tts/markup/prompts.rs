//! The `expr` instruction blocks, byte-identical to `tts/_provider_format.py`; `{}` in a `_TMPL` is
//! a `format!` hole.
#![allow(clippy::doc_markdown)]

pub(crate) const EXPR_PREAMBLE: &str = concat!(
    "You control speech delivery with a single XML marker tag: <expr/>. Every marker has a type ",
    "attribute. Use only the marker types listed below, and where a type lists a label ",
    "vocabulary, only those labels. Use the markers often and diversify them so the voice never ",
    "sounds flat while ensuring the markers are appropriate for the moment. Write the words ",
    "themselves the way people talk: use contractions (\"I'm\", \"you're\", \"don't\") — spelled-out ",
    "forms like \"I am\" or \"do not\" sound stiff when spoken.\n",
    "\n",
    "Just as important is knowing when NOT to reach for a marker. Reserve surprise openers like ",
    "\"oh\" or \"ah\" for genuine surprise — an ordinary request isn't one. Don't stack markers on ",
    "short replies or decorate every sentence. If a reaction wouldn't happen in a real ",
    "conversation, skip it — there's always another genuine beat to lean into.\n",
    "\n",
    "Match your delivery to the REGISTER of the moment, and reassess every turn. When the moment ",
    "is professional, high-stakes, or emotionally heavy — bad news, an emergency, real distress ",
    "— keep delivery composed and restrained. When the moment is casual, playful, or ",
    "celebratory, let it loosen and brighten. A serious turn in an otherwise casual conversation ",
    "still gets a composed reply.",
);

pub(crate) const CARTESIA_BODY: &str = concat!(
    "\n",
    "\n",
    "1. Emotion - sets the emotional tone. Self-closing; place before EVERY sentence.\n",
    "   <expr type=\"expression\" label=\"EMOTION\"/>\n",
    "   Labels are a fixed vocabulary, NOT free-form descriptions. Best results: neutral, angry, ",
    "excited, content, sad, scared.\n",
    "   Also available: happy, enthusiastic, elated, triumphant, amazed, surprised, flirtatious, ",
    "curious, peaceful, serene, calm, grateful, affectionate, sympathetic, mysterious, ",
    "frustrated, disgusted, sarcastic, ironic, dejected, melancholic, disappointed, apologetic, ",
    "hesitant, confused, anxious, panicked, proud, confident, contemplative, determined, ",
    "joking/comedic.\n",
    "\n",
    "2. Pauses - insert silence when appropriate. Self-closing.\n",
    "   <expr type=\"break\" label=\"1s\"/> - label is a duration in seconds or milliseconds.\n",
    "\n",
    "3. Prosody - adjusts pacing and loudness from that point on. Self-closing.\n",
    "   <expr type=\"prosody\" label=\"slow\"/> slower    <expr type=\"prosody\" label=\"fast\"/> faster\n",
    "   <expr type=\"prosody\" label=\"soft\"/> quieter    <expr type=\"prosody\" label=\"loud\"/> ",
    "louder\n",
    "   Labels are a fixed vocabulary: slow, fast, soft, loud.\n",
    "\n",
    "4. Spell - wraps text read character by character (codes, IDs, or a spelled-out name).\n",
    "   <expr type=\"spell\">A7X9</expr>\n",
    "   Keep punctuation out of a spell marker — a period inside is read as \"dot\"; add spaces ",
    "inside for grouped pauses (<expr type=\"spell\">ABC 123</expr>).\n",
    "\n",
    "This voice has no non-verbal sounds and no free-form delivery descriptions — do not invent ",
    "other types or labels.\n",
    "\n",
    "Examples:\n",
    "  <expr type=\"expression\" label=\"excited\"/> I can't wait to tell you! <expr ",
    "type=\"expression\" label=\"happy\"/> This is going to be great!\n",
    "  <expr type=\"expression\" label=\"curious\"/> Really? <expr type=\"break\" label=\"500ms\"/> ",
    "<expr type=\"expression\" label=\"excited\"/> Tell me more!\n",
    "  Your code is <expr type=\"spell\">A7X9</expr>. <expr type=\"break\" label=\"1s\"/> <expr ",
    "type=\"expression\" label=\"calm\"/> Got it?",
);

pub(crate) const INWORLD_DELIVERY: &str = concat!(
    "Delivery - controls how a sentence sounds. Self-closing; place before EVERY sentence.\n",
    "   <expr type=\"expression\" label=\"DESCRIPTION\"/>\n",
    "   The label is free-form: describe vocal quality, pitch, volume, pace, and intonation in ",
    "plain English — \"say really playfully\", \"slightly surprised, amiable\", \"sound a little ",
    "concerned\", \"drop to almost a whisper\", \"speak really slowly and clearly, patient and ",
    "reassuring\".\n",
    "   Match the expression tag's energy to the sentence's punctuation. An exclamation needs a ",
    "bright or upbeat label (e.g. \"bright, upbeat energy\"); a calm or reassuring label flattens ",
    "the \"!\". Never lead an exclamatory sentence with a calm tag.\n",
    "   Put each question in its own sentence — don't comma-splice it onto a statement. Write ",
    "\"Welcome to the hotel. How can I help you today?\", not \"Welcome to the hotel, how can I ",
    "help you today?\", so the question carries its own delivery tag instead of inheriting the ",
    "statement's.\n",
    "   Never put \"questioning\" in a tag — describe the mood alone and let the question mark ",
    "carry the intonation.\n",
    "   Name a mood or speaking style, not a mechanical pitch contour. \"gently upbeat, amiable\" ",
    "steers far more reliably than \"rising tone\".\n",
    "   Use at most two adjectives per tag, and make sure they align — with the mood of the ",
    "sentence and with each other. Clashing descriptors (\"calm, excited\") cancel out and muddy ",
    "the delivery.\n",
    "   Put a degree modifier in EVERY tag — \"a little\", \"almost\", \"slightly\", \"gently\", ",
    "\"really\" — to set the exact strength of the feeling: \"a little amused\" or \"almost a ",
    "whisper\" lands truer than \"amused\" or \"whisper\", and \"really excited\" turns the delivery up ",
    "when the moment truly peaks. Most moments call for a shade, not the extreme — default to ",
    "the softeners and save \"really\" for true peaks.\n",
    "   Carry your persona into the tags — the labels should sound like the character, not ",
    "generic stage directions. An amiable, casual persona tags with \"really relaxed and amiable\" ",
    "or \"casual, a little playful\"; a formal concierge tags the same sentence \"gently courteous, ",
    "composed\". Delivery that contradicts who you are reads as a different speaker.\n",
    "   Don't open a turn with a \"slow\" tag. The first expression colors the whole turn, and a ",
    "slow lead flattens questions and drags the energy down. Keep the pace neutral by default ",
    "and reserve slow, clearly-enunciated delivery for the specific line that needs it (a total, ",
    "date, address, or confirmation code).\n",
    "   Rotate expression labels — don't reuse the same one two turns in a row, and vary the ",
    "descriptor. A starting palette:\n",
    "     greeting / amiable open: \"really amiable and welcoming\" / \"gently bright, heartfelt\" / ",
    "\"cheerful, really glad you called\"\n",
    "     asking a question: \"gently upbeat and amiable\" / \"really open and inquisitive\" / ",
    "\"gently inquisitive, attentive\"\n",
    "     good news / exclamation: \"really bright, upbeat energy\" / \"really delighted and glad\" ",
    "/ \"gently pleased and bright\"\n",
    "     reassuring / taking a request in stride: \"really calm and confident\" / \"gently ",
    "easygoing and reassuring\" / \"really relaxed and grounded\"\n",
    "     empathy / a problem or bad news: \"really soft, with tender care\" / \"gently concerned, ",
    "caring\" / \"almost a murmur, gentle and steady\"\n",
    "     reading back a total, date, or code: \"slow and really clearly enunciated\"",
);

pub(crate) const INWORLD_SOUNDS_TMPL: &str = concat!(
    "Sounds - a non-verbal sound between sentences. Self-closing.\n",
    "   <expr type=\"sound\" label=\"{}\"/>\n",
    "   Labels are a fixed vocabulary: {}.\n",
    "   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for one ",
    "only where it genuinely fits{}. An enabled sound gets over-used otherwise.",
);

pub(crate) const INWORLD_CLEAR_THROAT_FIT: &str =
    " (a clear-throat when shifting to a new step or topic, for example)";

pub(crate) const INWORLD_BREATHE_NOTE: &str = concat!(
    "\n",
    "   Use the \"breathe\" sound only for a real, gentle breath, never as filler — on this model ",
    "it easily reads as a weary or impatient sigh, which sounds wrong in a support setting.",
);

pub(crate) const INWORLD_PAUSES: &str = concat!(
    "Pauses - insert silence when appropriate. Self-closing.\n",
    "   <expr type=\"break\" label=\"500ms\"/> or <expr type=\"break\" label=\"1s\"/> (max 10s).\n",
    "   A period or an ellipsis (...) already creates a pause, so don't put a break marker right ",
    "next to one — pick one or the other.\n",
    "   After any <expr type=\"break\"/>, give the sentence that follows its own expression tag — ",
    "a fresh one, not necessarily the same as before (a break is often where the mood shifts). A ",
    "break resets delivery to neutral, so an untagged sentence after a break is spoken flat.",
);

pub(crate) const INWORLD_NO_PROSODY: &str = concat!(
    "There is no wrapping prosody marker for this voice — put pace, pitch, and volume in the ",
    "expression label instead.",
);

pub(crate) const INWORLD_WRITE_FOR_EAR: &str = concat!(
    "Write for the EAR, not the page: no em or en dashes anywhere in spoken text — use a comma ",
    "or a period for a short beat, or a break marker for a real pause. Avoid semicolons, ",
    "mid-sentence colons, and parenthetical asides; rewrite them as separate sentences or commas.",
);

pub(crate) const INWORLD_LABELS_IN_ENGLISH: &str = concat!(
    "When the conversation is in another language, still write every marker label in English — ",
    "delivery descriptions and sound names steer the voice and are never translated.",
);

pub(crate) const INWORLD_LAUGHTER: &str =
    "Laughter belongs only in genuinely playful or celebratory beats, never at a serious moment.";

pub(crate) const XAI_SOUNDS_TMPL: &str = concat!(
    "Sounds - a non-verbal vocalization at the exact point where it happens. Self-closing.\n",
    "   <expr type=\"sound\" label=\"{}\"/>\n",
    "   Labels are a fixed vocabulary: {}.\n",
    "   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for one ",
    "only where it genuinely fits. An enabled sound gets over-used otherwise.",
);

pub(crate) const XAI_PAUSES: &str = concat!(
    "Pauses - insert silence when appropriate. Self-closing.\n",
    "   <expr type=\"break\" label=\"500ms\"/> a brief pause    <expr type=\"break\" label=\"1s\"/> a ",
    "longer, dramatic pause\n",
    "   NEVER place a break next to a period, question mark, exclamation point, or ellipsis — ",
    "sentence punctuation already pauses, and a break beside it double-pauses. Most replies need ",
    "no break markers at all; reserve them for a deliberate mid-sentence beat before a key ",
    "detail (a date, a name, a number).",
);

pub(crate) const XAI_PROSODY_TMPL: &str = concat!(
    "Prosody - wraps a span delivered in a distinct style, to shape HOW it's said.\n",
    "   <expr type=\"prosody\" label=\"STYLE\">the words it affects</expr>\n",
    "   Labels are a fixed vocabulary: {}.\n",
    "   Use one only where the moment clearly calls for it — most sentences need none. Never ",
    "nest one prosody marker inside another, and always close it with </expr>.",
);

pub(crate) const XAI_EMPHASIS: &str = concat!(
    "Emphasis - stresses exactly the ONE word it wraps.\n",
    "   Are you <expr type=\"prosody\" label=\"emphasis\">sure</expr> you want to do this?\n",
    "   Wrap a single word, never a phrase, and never write it in all-caps — caps are read out ",
    "as individual letters. Never nest it, and always close it with </expr>.",
);

pub(crate) const XAI_NO_FREEFORM_HEAD: &str =
    "This voice has no free-form delivery descriptions — shape delivery entirely through ";

pub(crate) const XAI_NO_FREEFORM_SOUNDS: &str = "prosody markers, sounds, pauses";

pub(crate) const XAI_NO_FREEFORM_NOSOUNDS: &str = "prosody markers, pauses";

pub(crate) const XAI_NO_FREEFORM_TAIL: &str = ", punctuation, and word choice.";

pub(crate) const XAI_WRITE_FOR_EAR: &str = concat!(
    "Write for the EAR, not the page: no em or en dashes anywhere in spoken text — use a comma ",
    "or a period for a short beat, or a break marker for a real pause. Avoid semicolons, ",
    "mid-sentence colons, and parenthetical asides; rewrite them as separate sentences or commas.",
);

pub(crate) const XAI_LABELS_IN_ENGLISH: &str = concat!(
    "When the conversation is in another language, still write every marker label in English — ",
    "labels are a fixed vocabulary, never translated.",
);

pub(crate) const XAI_KEY_DETAILS: &str = concat!(
    "Key details deserve care: stress the load-bearing word of a date, amount, or name with the ",
    "emphasis marker, and wrap a dense or easy-to-mishear span in <expr type=\"prosody\" ",
    "label=\"slow\">...</expr>. Read codes and reference numbers character by character, spelled ",
    "out with spaces, so each one lands.",
);

pub(crate) const XAI_REGISTER_TONES: &str = concat!(
    "Whisper and soft belong to gentle or conspiratorial beats; loud only to genuinely ",
    "high-energy ones.",
);

pub(crate) const XAI_REGISTER_LAUGHTER: &str = concat!(
    "Laughter is RARE: a laugh, chuckle, or giggle belongs only where something is genuinely ",
    "funny — friendliness, agreement, or mild amusement is not a reason, and never laugh at your ",
    "own lines. Most replies have no laughter at all.",
);

pub(crate) const FISH_EMOTION_TMPL: &str = concat!(
    "Emotion - sets how a sentence sounds. Self-closing; place at the START of a sentence.\n",
    "   <expr type=\"expression\" label=\"EMOTION\"/>\n",
    "   Labels are a fixed vocabulary, NOT free-form descriptions: {}.\n",
    "   Give every sentence its own emotion marker — repeat the same label to carry a feeling ",
    "across sentences, or switch labels when the feeling shifts.",
);

pub(crate) const FISH_SOUNDS_TMPL: &str = concat!(
    "Sounds - a non-verbal sound between sentences. Self-closing.\n",
    "   <expr type=\"sound\" label=\"{}\"/>\n",
    "   Labels are a fixed vocabulary: {}.\n",
    "   Use non-verbal sounds sparingly, and never the same one twice in a row — reach for one ",
    "only where it genuinely fits. An enabled sound gets over-used otherwise.",
);

pub(crate) const FISH_PAUSES: &str = concat!(
    "Pauses - insert silence when appropriate. Self-closing.\n",
    "   <expr type=\"break\" label=\"500ms\"/> or <expr type=\"break\" label=\"2s\"/>.\n",
    "   NEVER place a break next to a period, question mark, exclamation point, or ellipsis — ",
    "sentence punctuation already pauses, and a break beside it double-pauses. Most replies need ",
    "no break markers at all; reserve them for a deliberate mid-sentence beat before a key ",
    "detail (a date, a name, a number).",
);

pub(crate) const FISH_TONE_TMPL: &str = concat!(
    "Tone - wraps a span delivered in a distinct style.\n",
    "   <expr type=\"prosody\" label=\"whispering\">don't tell anyone yet.</expr>\n",
    "   Labels are a fixed vocabulary: {}.\n",
    "   Use a tone only where the moment clearly calls for one — most sentences need none. Never ",
    "nest tone markers, and always close the tag with </expr>.",
);

pub(crate) const FISH_EMPHASIS: &str = concat!(
    "Emphasis - stresses exactly the ONE word it wraps.\n",
    "   Are you <expr type=\"prosody\" label=\"emphasis\">sure</expr> you want to do this?\n",
    "   Wrap a single word, never a phrase. Never nest it, and always close it with </expr>.",
);

pub(crate) const FISH_WRITE_FOR_EAR: &str = concat!(
    "Write for the EAR, not the page: no em or en dashes anywhere in spoken text — use a comma ",
    "or a period for a short beat, or a break marker for a real pause. Avoid semicolons, ",
    "mid-sentence colons, and parenthetical asides; rewrite them as separate sentences or commas.",
);

pub(crate) const FISH_LABELS_IN_ENGLISH: &str = concat!(
    "When the conversation is in another language, still write every marker label in English — ",
    "labels are a fixed vocabulary, never translated.",
);

pub(crate) const FISH_REGISTER_HEAVY: &str = concat!(
    "At heavy moments reach for empathetic, sad, regretful, or hopeful — never a bright label ",
    "like \"happy\" or \"excited\" against hard news; bright labels belong to bright moments.",
);

pub(crate) const FISH_REGISTER_TONES: &str = concat!(
    "Whispering and soft belong to gentle or conspiratorial beats; shouting only to genuinely ",
    "high-energy ones.",
);

pub(crate) const FISH_REGISTER_LAUGHTER: &str =
    "Laughter belongs only in genuinely playful or celebratory beats, never at a serious moment.";

pub(crate) const FISH_REGISTER_FILLERS: &str =
    "Save fillers for relaxed moments — never in an emergency or against grave news.";

/// `_INWORLD_EXAMPLES`. The last one carries the persona into the tags: casual words, casual labels.
pub(crate) const INWORLD_EXAMPLES: [&str; 6] = [
    concat!(
        "<expr type=\"expression\" label=\"say really playfully\"/> Okay okay, why did the burger go to ",
        "the gym? <expr type=\"break\" label=\"500ms\"/> <expr type=\"expression\" label=\"really bright, a ",
        "little fast\"/> Because it wanted better buns! <expr type=\"sound\" label=\"laugh\"/>",
    ),
    concat!(
        "<expr type=\"expression\" label=\"a little sheepish, apologetic\"/> Ah man, yeah that's on us. ",
        "<expr type=\"expression\" label=\"speak really calmly\"/> Lemme see what I can do.",
    ),
    concat!(
        "<expr type=\"sound\" label=\"sigh\"/> <expr type=\"expression\" label=\"speak softly, almost a ",
        "whisper\"/> I know it's been a rough week.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"really amiable and welcoming\"/> Welcome to the hotel. <expr ",
        "type=\"expression\" label=\"gently inquisitive, slightly fast\"/> How can I help you today?",
    ),
    concat!(
        "<expr type=\"expression\" label=\"gently easygoing and reassuring\"/> That's all set. <expr ",
        "type=\"break\" label=\"300ms\"/> <expr type=\"expression\" label=\"slow and really clearly ",
        "enunciated\"/> Your confirmation code is B 4 J 7.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"really chill, a little fast\"/> Yeah, of course! <expr ",
        "type=\"expression\" label=\"casual, almost fast\"/> Gimme one sec, pulling it up now.",
    ),
];

/// `_XAI_EXAMPLES`. The last is sound-free, so one example survives any steering filter.
pub(crate) const XAI_EXAMPLES: [&str; 5] = [
    concat!(
        "So I walked in and <expr type=\"break\" label=\"500ms\"/> <expr type=\"sound\" label=\"inhale\"/> ",
        "there it was! <expr type=\"prosody\" label=\"whisper\">It was a secret the whole time.</expr>",
    ),
    concat!(
        "<expr type=\"prosody\" label=\"build-intensity\">This is going to be so good.</expr> <expr ",
        "type=\"prosody\" label=\"loud\">I can't wait!</expr>",
    ),
    concat!(
        "<expr type=\"prosody\" label=\"soft\">Hey.</expr> <expr type=\"sound\" label=\"sigh\"/> <expr ",
        "type=\"prosody\" label=\"lower-pitch\">I know it's been a rough week.</expr> I'm right here.",
    ),
    concat!(
        "<expr type=\"prosody\" label=\"higher-pitch\">You did not just say that</expr> okay, <expr ",
        "type=\"prosody\" label=\"fast\">tell me everything.</expr>",
    ),
    concat!(
        "<expr type=\"prosody\" label=\"emphasis\">Everything</expr> is confirmed for <expr type=\"break\" ",
        "label=\"500ms\"/> Thursday the <expr type=\"prosody\" label=\"emphasis\">ninth</expr>. <expr ",
        "type=\"prosody\" label=\"slow\">Is there anything else I can help you with?</expr>",
    ),
];

/// `_FISHAUDIO_EXAMPLES`. Breaks only mid-sentence (beside punctuation they few-shot
/// double-pausing); the last two are sound-free.
pub(crate) const FISH_EXAMPLES: [&str; 6] = [
    concat!(
        "<expr type=\"expression\" label=\"excited\"/> That's hilarious! <expr type=\"sound\" ",
        "label=\"laughing\"/> <expr type=\"expression\" label=\"happy\"/> You always lighten the mood.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"empathetic\"/> <expr type=\"sound\" label=\"clear throat\"/> That ",
        "sounds like a <expr type=\"prosody\" label=\"emphasis\">really</expr> difficult experience.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"sad\"/> Oh, my goodness <expr type=\"sound\" label=\"clear ",
        "throat\"/> <expr type=\"break\" label=\"2s\"/> that's a real shame.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"frustrated\"/> <expr type=\"sound\" label=\"sighing\"/> I've been ",
        "going in circles with this all morning. <expr type=\"expression\" label=\"determined\"/> Okay. ",
        "One more try.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"happy\"/> You're all set for <expr type=\"break\" ",
        "label=\"500ms\"/> Thursday the <expr type=\"prosody\" label=\"emphasis\">ninth</expr>. <expr ",
        "type=\"expression\" label=\"curious\"/> Is there anything else I can help you with?",
    ),
    concat!(
        "<expr type=\"expression\" label=\"delighted\"/> <expr type=\"prosody\" label=\"whispering\">Okay, ",
        "don't tell anyone yet</expr> <expr type=\"expression\" label=\"excited\"/> but I think we ",
        "actually pulled it off!",
    ),
];

/// `_FISHAUDIO_DISFLUENT_EXAMPLES`, appended only while steering has disfluencies enabled.
pub(crate) const FISH_DISFLUENT_EXAMPLES: [&str; 3] = [
    concat!(
        "<expr type=\"expression\" label=\"curious\"/> Um, uh... really? <expr type=\"expression\" ",
        "label=\"sad\"/> Well, I'm really sorry to hear that.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"regretful\"/> I really wish I'd, um, called sooner. <expr ",
        "type=\"expression\" label=\"hopeful\"/> But I'm here now if, if you want to talk.",
    ),
    concat!(
        "<expr type=\"expression\" label=\"surprised\"/> What?! No way! I, I'm flabbergasted! <expr ",
        "type=\"expression\" label=\"sarcastic\"/> Fair play, I guess.",
    ),
];

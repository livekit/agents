//! Speech markup vocabulary and language-neutral export contracts.

/// Returned because the blingfire tokenizer lives outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentenceTokenizerConfig {
    pub max_token_len: Option<usize>,
    /// Under expressive this equals `max_token_len`, so sentences batch up to that size.
    pub min_token_len: Option<usize>,
    /// Only under expressive: xml-aware tokenization would stall streaming on a stray `<`.
    pub xml_aware: bool,
}

/// The selected provider dialect, retained in resource descriptions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Markup(pub Option<Provider>);

impl Markup {
    pub fn from_provider_key(provider_key: &str) -> Self {
        Self(Provider::from_wire(provider_key))
    }

    pub fn from_provider(provider: Provider) -> Self {
        Self(Some(provider))
    }

    pub fn provider_key(&self) -> &'static str {
        self.0.map_or("", Provider::as_str)
    }
}

/// An expressive markup tag stripped from a transcript: the tag name (or the expr marker's type)
/// and its payload (the ``value`` attribute, the expr ``label`` or the inner text).
// A `TypedDict` in the reference; no field defaults, since `t["type"]` must raise on a missing key.
#[crate::host(record(module = "tts", path = "livekit.agents.tts", typed_dict, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressiveTag {
    /// The markup tag name (``"emotion"``, ``"sound"``, ...) or the expr marker type.
    #[host(rename = "type")]
    pub kind: String,
    /// The payload: the ``value="..."`` attribute, the expr ``label``, or the tag's inner text.
    pub value: String,
}

impl ExpressiveTag {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        ExpressiveTag { kind: kind.into(), value: value.into() }
    }
}

/// `AgentMood`: the closed set a delivery label normalizes to, in declaration order. The wire
/// strings are the exact `AgentMood` literals, a wire contract (`lk.expression`).
#[crate::host(wire_enum)]
#[crate::host(codec(extract, emit, host = "str"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mood {
    Excited,
    Happy,
    Playful,
    Curious,
    Surprised,
    Hopeful,
    Empathetic,
    Sad,
    Angry,
    Anxious,
    Calm,
}

/// The keys of `_PROVIDER_MARKUP`: `_provider_key()`, the key into every table in this module.
/// `from_wire` is `None` for a key with no dialect (`""`, `"rime"`, ...).
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Cartesia,
    Inworld,
    Xai,
    #[host(rename = "fishaudio")]
    FishAudio,
    Gemini,
}

/// The unit steering turns non-verbal sounds off by.
///
/// Declaration order is the wire order of the supported nonverbal matrix.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NonverbalField {
    /// laugh, chuckle, giggle -- and laugh-speak delivery
    Laughing,
    /// audible breath, inhale, exhale
    Breathing,
    Sighing,
    Crying,
    /// non-lexical voiced sounds -- humming a tune, sing-song or sung delivery
    Vocalizing,
    /// tsk, tongue-click, lip-smack
    MouthSounds,
    /// cough, clearing the throat, yawn
    ReflexSounds,
}

/// `nonverbal_sounds: bool | NonverbalOptions`. `Enabled(true)` and `Per(default)` stay apart so a
/// round-tripped config is not rewritten.
#[crate::host(record(module = "expressive", transparent, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonverbalSounds {
    /// `True` is identical to omitting the key; `False` omits the Sounds section.
    Enabled(bool),
    /// An omitted category stays enabled.
    Per(NonverbalOptions),
}

/// Non-verbal vocalizations the TTS may produce (sounds that aren't words).
///
/// A sparse opt-out: omitted keys default to ON, and a category set to
/// ``False`` is never advertised to the LLM — ``{"laughing": False}``
/// removes laughter and nothing else. Together the keys cover every sound
/// the providers offer. Each key may later widen to ``bool | <Sound>Options``
/// (e.g. ``laughing: bool | LaughingOptions``) for per-sound control without
/// breaking existing callers.
// The `///` above is the reference's docstring, verbatim. `None` on a field is an absent key.
#[crate::host(record(
    module = "expressive",
    path = "livekit.agents.voice.agent_session",
    typed_dict,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NonverbalOptions {
    /// laugh, chuckle, giggle — and laugh-speak delivery
    #[host(default = None)]
    pub laughing: Option<bool>,
    /// audible breath, inhale, exhale
    #[host(default = None)]
    pub breathing: Option<bool>,
    #[host(default = None)]
    pub sighing: Option<bool>,
    #[host(default = None)]
    pub crying: Option<bool>,
    /// non-lexical voiced sounds — humming a tune, sing-song or sung delivery
    #[host(default = None)]
    pub vocalizing: Option<bool>,
    /// tsk, tongue-click, lip-smack
    #[host(default = None)]
    pub mouth_sounds: Option<bool>,
    /// cough, clearing the throat, yawn
    #[host(default = None)]
    pub reflex_sounds: Option<bool>,
}

#[crate::host(record(module = "expressive", extract))]
#[crate::host(wire_strs(msg = format!("unknown pace {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    Slow,
    Normal,
    Fast,
}

/// Steers verbal delivery and non-verbal sounds in generated speech.
///
/// Every key is a sparse override on the default (full sound vocabulary,
/// light fillers): the expressive instructions already tell the LLM to match
/// its delivery to the register of the moment, so most agents need no
/// steering at all — set a key only to take an option away regardless of
/// context.
// The `///` above is the reference's docstring, verbatim: it is the published docs. Every key
// is optional: absent and `None` alike are unset.
#[crate::host(record(
    module = "expressive",
    name = "SpeechSteeringOptions",
    path = "livekit.agents.voice.agent_session",
    typed_dict,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeechSteering {
    /// Filler words such as "um" / "uh". On by default
    /// (``DEFAULT_SPEECH_STEERING_OPTIONS``); set ``False`` to opt out.
    // Tested with `is not None`: `Some(false)` renders the opposite guideline, not none.
    #[host(default = None)]
    pub disfluencies: Option<bool>,
    /// Which non-verbal sounds the TTS may make. ``True`` (and omitting the
    /// key) keeps the provider's full vocabulary, ``False`` disables every sound,
    /// and a ``NonverbalOptions`` dict toggles per category (omitted categories
    /// stay enabled).
    #[host(default = None)]
    pub nonverbal_sounds: Option<NonverbalSounds>,
    #[host(default = None)]
    pub pace: Option<Pace>,
}

use std::collections::BTreeMap;

#[crate::host(functions(pid = false))]
pub trait MarkupFunctions {
    #[host(function, pid = true)]
    fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String>;
    #[host(function)]
    fn supported_nonverbals(provider: String) -> crate::MarkupInfo;
    #[host(function)]
    fn split_all_markup(text: String) -> (String, Vec<ExpressiveTag>);
    #[host(function)]
    fn strip_all_markup(text: String) -> String;
    #[host(function)]
    fn strip_expr_markup(text: String) -> String;
    #[host(function)]
    fn split_expr_markup(text: String) -> (String, Vec<ExpressiveTag>);
    #[host(function)]
    fn to_xml(
        tag_name: String,
        #[host(default = None)] content: Option<String>,
        #[host(default = None)] attrs: Option<Vec<(String, String)>>,
    ) -> String;
}

/// The markup dialect a TTS speaks, for the expressive pipeline: what the LLM is taught to write
/// and how it is normalized and lowered before synthesis.
///
/// Plugins opt in by overriding `_provider_key`.
#[crate::host(class(name = "Markup", pid = false))]
pub trait MarkupClass: Send + Sized + 'static {
    #[host(new)]
    fn from_provider_key(provider_key: String) -> Self;
    /// The provider's key into the markup tables, or `""` for no markup.
    #[host(call, name = "_provider_key")]
    fn provider_key(&self) -> String;
    /// The instructions teaching the LLM this voice's markup, `None` without markup.
    #[host(call)]
    fn llm_instructions(
        &self,
        #[host(kw, default = None)] speech_steering: Option<SpeechSteering>,
    ) -> Option<String>;
    /// Fix common LLM markup mistakes (e.g. unclosed self-closing tags).
    #[host(call)]
    fn normalize(&self, text: String) -> String;
    /// Lower the framework's markup to the provider's native syntax.
    #[host(call)]
    fn convert(&self, text: String) -> String;
}

/// Stateful, provider-agnostic markup stripper for one transcript segment.
#[crate::host(class(name = "TranscriptMarkupStripper", pid = false))]
pub trait TranscriptStripper: Send + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn push(&mut self, text: String) -> String;
    #[host(call)]
    fn flush(&mut self) -> String;
    #[host(getter)]
    fn tags(&self) -> Vec<ExpressiveTag>;
    #[host(call)]
    fn expression_attribute(&self) -> Option<BTreeMap<String, String>>;
}

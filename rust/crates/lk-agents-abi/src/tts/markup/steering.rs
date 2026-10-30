//! Sparse speech steering configuration.

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

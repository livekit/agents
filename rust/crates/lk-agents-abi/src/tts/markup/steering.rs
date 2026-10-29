//! Sparse speech steering configuration.

use lk_agents_host_derive::Host;

/// `nonverbal_sounds: bool | NonverbalOptions`. `Enabled(true)` and `Per(default)` stay apart so a
/// round-tripped config is not rewritten.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, untagged)]
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "expressive",
    path = "livekit.agents.voice.agent_session",
    name = "NonverbalOptions"
)]
#[binding(extract, from_items)]
pub struct NonverbalOptions {
    /// laugh, chuckle, giggle — and laugh-speak delivery
    #[python(ty = "bool")]
    pub laughing: Option<bool>,
    /// audible breath, inhale, exhale
    #[python(ty = "bool")]
    pub breathing: Option<bool>,
    #[python(ty = "bool")]
    pub sighing: Option<bool>,
    #[python(ty = "bool")]
    pub crying: Option<bool>,
    /// non-lexical voiced sounds — humming a tune, sing-song or sung delivery
    #[python(ty = "bool")]
    pub vocalizing: Option<bool>,
    /// tsk, tongue-click, lip-smack
    #[python(ty = "bool")]
    pub mouth_sounds: Option<bool>,
    /// cough, clearing the throat, yawn
    #[python(ty = "bool")]
    pub reflex_sounds: Option<bool>,
}

#[crate::host(wire_strs(msg = format!("unknown pace {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    Slow,
    Normal,
    Fast,
}

#[cfg(feature = "python")]
crate::python::from_host_str!(Pace);

/// Steers verbal delivery and non-verbal sounds in generated speech.
///
/// Every key is a sparse override on the default (full sound vocabulary,
/// light fillers): the expressive instructions already tell the LLM to match
/// its delivery to the register of the moment, so most agents need no
/// steering at all — set a key only to take an option away regardless of
/// context.
// The `///` above is the reference's docstring, verbatim: it is the published docs. Explicit
// `ty`s because Rust `Option` means absent, and none of these keys is nullable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "expressive",
    path = "livekit.agents.voice.agent_session",
    name = "SpeechSteeringOptions"
)]
#[binding(extract, from_items)]
pub struct SpeechSteering {
    /// Filler words such as "um" / "uh". On by default
    /// (``DEFAULT_SPEECH_STEERING_OPTIONS``); set ``False`` to opt out.
    // Tested with `is not None`: `Some(false)` renders the opposite guideline, not none.
    #[python(ty = "bool")]
    pub disfluencies: Option<bool>,
    /// Which non-verbal sounds the TTS may make. ``True`` (and omitting the
    /// key) keeps the provider's full vocabulary, ``False`` disables every sound,
    /// and a ``NonverbalOptions`` dict toggles per category (omitted categories
    /// stay enabled).
    //
    // `bool | NonverbalOptions`: `NonverbalSounds`' own reader.
    #[python(ty = "bool | NonverbalOptions")]
    pub nonverbal_sounds: Option<NonverbalSounds>,
    // `None` is absent; a string parses via `FromStr`, an unknown one raising.
    #[python(ty = "Literal[\"slow\", \"normal\", \"fast\"]")]
    pub pace: Option<Pace>,
}

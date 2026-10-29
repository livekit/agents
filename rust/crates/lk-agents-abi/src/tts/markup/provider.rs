//! The provider and nonverbal vocabularies crossing the speech boundary.

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

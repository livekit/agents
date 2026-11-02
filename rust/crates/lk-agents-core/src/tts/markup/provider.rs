//! The per-provider dialect tables from `tts/_provider_format.py`.
//!
//! Provider docs:
//! * Cartesia: <https://docs.cartesia.ai/build-with-cartesia/sonic-3/ssml-tags>
//! * Inworld: <https://docs.inworld.ai/tts/capabilities/steering>
//! * xAI: <https://docs.x.ai/developers/model-capabilities/audio/text-to-speech>
//! * Fish Audio: <https://docs.fish.audio/developer-guide/core-features/emotions>
//! * Gemini: <https://ai.google.dev/gemini-api/docs/speech-generation#controllable>

use NonverbalField::*;
use std::collections::BTreeSet;
use std::sync::LazyLock;

pub use lk_agents_abi::tts::markup::{NonverbalField, Provider};

pub trait ProviderExt {
    fn native_tags(self) -> &'static [&'static str];
    fn sounds(self) -> &'static [&'static str];
    fn prosody(self) -> &'static [&'static str];
    fn max_input_len(self) -> Option<usize>;
    fn self_closing_tags(self) -> &'static [&'static str];
}

impl ProviderExt for Provider {
    /// `_PROVIDER_MARKUP[provider]`: what `expr` lowers to, and what catches hallucinated natives.
    fn native_tags(self) -> &'static [&'static str] {
        match self {
            Provider::Cartesia => &CARTESIA_TAGS,
            Provider::Inworld => &INWORLD_TAGS,
            Provider::Xai => XAI_TAGS.as_slice(),
            // Fish's native dialect is brackets; these names catch hallucinated XML natives.
            Provider::FishAudio => &FISHAUDIO_TAGS,
            Provider::Gemini => GEMINI_TAGS.as_slice(),
        }
    }

    /// `_PROVIDER_SOUNDS`, before steering (empty for Cartesia).
    fn sounds(self) -> &'static [&'static str] {
        match self {
            Provider::Inworld => &INWORLD_SOUNDS,
            Provider::Xai => &XAI_INLINE,
            Provider::FishAudio => &FISHAUDIO_SOUNDS,
            Provider::Gemini => &GEMINI_SOUNDS,
            Provider::Cartesia => &[],
        }
    }

    /// `_PROVIDER_PROSODY`. Only xAI has one.
    fn prosody(self) -> &'static [&'static str] {
        match self {
            Provider::Xai => &XAI_WRAPPING,
            _ => &[],
        }
    }

    /// `_MAX_INPUT_LEN`: the chunking cap in characters, also the expressive batch size.
    fn max_input_len(self) -> Option<usize> {
        match self {
            Provider::Inworld => Some(900),
            Provider::Cartesia => Some(400),
            // well under xAI's 15,000-char request limit; sized as an expressive batch target
            Provider::Xai => Some(1000),
            // Fish markers are sentence-scoped, so per-sentence emission keeps TTFA low.
            Provider::FishAudio | Provider::Gemini => None,
        }
    }

    /// `_SELF_CLOSING_TAGS`.
    fn self_closing_tags(self) -> &'static [&'static str] {
        match self {
            Provider::Cartesia => &["emotion", "speed", "volume", "break"],
            Provider::Inworld => &["expression", "sound", "break"],
            Provider::FishAudio => &["expression", "sound", "break"],
            Provider::Xai | Provider::Gemini => &[],
        }
    }
}

pub(crate) const CARTESIA_TAGS: [&str; 5] = ["emotion", "speed", "volume", "break", "spell"];

pub(crate) const INWORLD_TAGS: [&str; 3] = ["expression", "sound", "break"];

// `XAI_WRAPPING` is what expr prosody lowers to; all stay in `XAI_TAGS` so hallucinated natives
// are stripped.
pub(crate) const XAI_EMOTIONS: [&str; 12] = [
    "happy",
    "sad",
    "angry",
    "excited",
    "calm",
    "surprised",
    "sympathetic",
    "curious",
    "sarcastic",
    "confident",
    "playful",
    "nervous",
];
pub(crate) const XAI_INLINE: [&str; 12] = [
    "breath",
    "inhale",
    "exhale",
    "sigh",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "hum-tune",
];
pub(crate) const XAI_WRAPPING: [&str; 13] = [
    "emphasis",           // stress the wrapped words
    "whisper",            // quiet, intimate
    "soft",               // lower volume
    "loud",               // higher volume
    "build-intensity",    // ramp energy up over the span
    "decrease-intensity", // ease energy off over the span
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",   // playful, musical lilt
    "singing",     // actually sung
    "laugh-speak", // talk through a laugh
];
// All tags are XML in the transcript, so all are stripped.
pub(crate) static XAI_TAGS: LazyLock<Vec<&str>> =
    LazyLock::new(|| [&XAI_EMOTIONS[..], &XAI_WRAPPING[..], &["sound", "break"]].concat());

// `convert_markup` lowers expr to Fish's brackets; the names stay in `FISHAUDIO_TAGS` to strip
// hallucinated natives. Every emotion maps to a non-fallback mood in `mood.rs`.
pub(crate) const FISHAUDIO_EMOTIONS: [&str; 19] = [
    "regretful",
    "hopeful",
    "happy",
    "excited",
    "curious",
    "surprised",
    "sad",
    "empathetic",
    "sarcastic",
    "calm",
    "angry",
    "worried",
    "nervous",
    "confident",
    "grateful",
    "delighted",
    "disappointed",
    "frustrated",
    "determined",
];
pub(crate) const FISHAUDIO_SOUNDS: [&str; 8] = [
    "laughing",
    "chuckling",
    "clear throat",
    "sighing",
    "gasping",
    "groaning",
    "yawning",
    "sobbing",
];
/// Neutral prefix markers, never steering-filtered.
pub(crate) const FISHAUDIO_TONES: [&str; 4] = ["whispering", "soft", "shouting", "hurried"];
pub(crate) const FISHAUDIO_TAGS: [&str; 4] = ["expression", "sound", "break", "emphasis"];

// Gemini's direction rides two channels: an `expression` marker becomes the part's
// `speech_metadata.style` (the plugin lifts it out), while sounds, breaks and emphasis lower to
// inline tags. Angle brackets: the docs measured them more reliable than square ones.
pub(crate) const GEMINI_SOUNDS: [&str; 9] = [
    // the model's own documented list
    "laugh", "chuckle", "sigh", "breath", "cough", "argh",
    // named by the general TTS docs, which warn there is no exhaustive list; the rest of what
    // they name is sustained delivery, which belongs in the style
    "gasp", "giggle", "cry",
];
/// Gemini documents one pause length, so every break lowers to it.
pub(crate) const GEMINI_PAUSE: &str = "short pause";
pub(crate) static GEMINI_TAGS: LazyLock<Vec<&str>> =
    LazyLock::new(|| [&GEMINI_SOUNDS[..], &[GEMINI_PAUSE]].concat());

pub(crate) const INWORLD_SOUNDS: [&str; 6] =
    ["laugh", "sigh", "breathe", "clear throat", "cough", "yawn"];

/// Union of every provider's XML tag names. Sorted: it is the alternation order, and CPython
/// alternation is leftmost-first.
pub(crate) static ALL_MARKUP_TAGS: LazyLock<Vec<&str>> = LazyLock::new(|| {
    let providers = [
        Provider::Cartesia,
        Provider::Inworld,
        Provider::Xai,
        Provider::FishAudio,
        Provider::Gemini,
    ];
    let tags: BTreeSet<&str> =
        providers.iter().flat_map(|p| p.native_tags().iter().copied()).collect();
    tags.into_iter().collect()
});

/// `_NONVERBAL_SOUND_LABELS[provider]`: every label in [`Provider::sounds`] is governed by exactly
/// one field. Insertion order, not sorted.
pub(crate) fn nonverbal_sound_labels(
    provider: Provider,
) -> &'static [(NonverbalField, &'static [&'static str])] {
    match provider {
        Provider::Inworld => &[
            (Laughing, &["laugh"]),
            (Breathing, &["breathe"]),
            (Sighing, &["sigh"]),
            (Crying, &[]),
            (Vocalizing, &[]),
            (MouthSounds, &[]),
            (ReflexSounds, &["cough", "clear throat", "yawn"]),
        ],
        Provider::Xai => &[
            (Laughing, &["laugh", "chuckle", "giggle"]),
            (Breathing, &["breath", "inhale", "exhale"]),
            (Sighing, &["sigh"]),
            (Crying, &["cry"]),
            (Vocalizing, &["hum-tune"]), // non-lexical voiced sounds
            (MouthSounds, &["tsk", "tongue-click", "lip-smack"]),
            (ReflexSounds, &[]), // xAI has no cough/yawn sounds
        ],
        Provider::FishAudio => &[
            (Laughing, &["laughing", "chuckling"]),
            (Breathing, &["gasping"]),
            (Sighing, &["sighing"]),
            (Crying, &["sobbing"]),
            (Vocalizing, &["groaning"]),
            (MouthSounds, &[]),
            (ReflexSounds, &["clear throat", "yawning"]),
        ],
        Provider::Gemini => &[
            (Laughing, &["laugh", "chuckle", "giggle"]),
            (Breathing, &["breath", "gasp"]),
            (Sighing, &["sigh"]),
            (Crying, &["cry"]),
            (Vocalizing, &["argh"]),
            (MouthSounds, &[]),
            (ReflexSounds, &["cough"]),
        ],
        Provider::Cartesia => &[],
    }
}

/// `_NONVERBAL_PROSODY_LABELS[provider]`: only vocal-style prosody is steerable.
pub(crate) fn nonverbal_prosody_labels(
    provider: Provider,
) -> &'static [(NonverbalField, &'static [&'static str])] {
    match provider {
        Provider::Xai => &[(Laughing, &["laugh-speak"]), (Vocalizing, &["sing-song", "singing"])],
        _ => &[],
    }
}

/// Only surviving labels' hints are advertised.
pub(crate) fn sound_usage_hint(sound: &str) -> Option<&'static str> {
    Some(match sound {
        "laugh" | "laughing" => "a laugh at something obviously funny",
        "chuckle" | "chuckling" | "giggle" => "a chuckle at something subtly humorous",
        "sigh" | "sighing" => "a sigh when commiserating",
        "inhale" => "a sharp inhale before a big reveal",
        "gasp" | "gasping" => "a gasp at a sudden shock or reveal",
        "lip-smack" | "tongue-click" => "a lip-smack or tongue-click as a tiny beat of thought",
        "tsk" => "a tsk for mock-disapproval",
        "clear throat" => "a clear-throat when shifting to a new step or topic",
        "argh" | "groaning" => "a groan at a groan-worthy pun or an unwelcome chore",
        "cough" => "a cough only when a cough is the point",
        "yawning" => "a yawn when tiredness itself is the topic",
        "cry" | "sobbing" => "a sob reserved for real heartbreak",
        _ => return None,
    })
}

/// `_XAI_SOUND_ALIASES`.
pub(crate) fn xai_sound_alias(label: &str) -> Option<&'static str> {
    (label == "breathe").then_some("breath")
}

/// `_FISHAUDIO_SOUND_ALIASES`.
pub(crate) fn fishaudio_sound_alias(label: &str) -> Option<&'static str> {
    Some(match label {
        "laugh" => "laughing",
        "chuckle" => "chuckling",
        "sigh" => "sighing",
        "gasp" => "gasping",
        "groan" => "groaning",
        "yawn" => "yawning",
        "sob" | "cry" => "sobbing",
        _ => return None,
    })
}

/// `_GEMINI_SOUND_ALIASES`: expr labels that differ from Gemini's native tag names.
pub(crate) fn gemini_sound_alias(label: &str) -> Option<&'static str> {
    Some(match label {
        "breathe" => "breath",
        "laughing" | "laughs" => "laugh",
        "chuckling" => "chuckle",
        "sighing" | "sighs" => "sigh",
        "groan" | "groaning" => "argh",
        "giggles" | "giggling" => "giggle",
        "gasping" => "gasp",
        "crying" | "sob" | "sobbing" => "cry",
        _ => return None,
    })
}

/// `_CARTESIA_PROSODY`.
pub(crate) fn cartesia_prosody(label: &str) -> Option<&'static str> {
    Some(match label {
        "slow" => r#"<speed ratio="0.85"/>"#,
        "fast" => r#"<speed ratio="1.2"/>"#,
        "soft" => r#"<volume ratio="0.9"/>"#,
        "loud" => r#"<volume ratio="1.3"/>"#,
        _ => return None,
    })
}

/// Empty fields omitted.
pub fn supported_nonverbals(
    provider: Option<Provider>,
) -> Vec<(NonverbalField, Vec<&'static str>)> {
    let Some(provider) = provider else { return Vec::new() };
    let mut merged: Vec<(NonverbalField, Vec<&'static str>)> = Vec::new();
    for table in [nonverbal_sound_labels(provider), nonverbal_prosody_labels(provider)] {
        for (field, labels) in table {
            if labels.is_empty() {
                continue;
            }
            match merged.iter_mut().find(|(f, _)| f == field) {
                Some((_, acc)) => acc.extend_from_slice(labels),
                None => merged.push((*field, labels.to_vec())),
            }
        }
    }
    merged
}

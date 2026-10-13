//! `SpeechSteeringOptions`: a sparse opt-out, a key is set only to take an option away.

use lk_agents_host_derive::Host;

use crate::tts::markup::provider::{
    NonverbalField, Provider, nonverbal_prosody_labels, nonverbal_sound_labels, sound_usage_hint,
};

/// `On` and `Per(default)` are kept apart so a round-tripped config is not rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonverbalSounds {
    /// Identical to omitting the key.
    On,
    /// The Sounds section is omitted.
    Off,
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

impl NonverbalOptions {
    fn enabled(&self, field: NonverbalField) -> bool {
        match field {
            NonverbalField::Laughing => self.laughing,
            NonverbalField::Breathing => self.breathing,
            NonverbalField::Sighing => self.sighing,
            NonverbalField::Crying => self.crying,
            NonverbalField::Vocalizing => self.vocalizing,
            NonverbalField::MouthSounds => self.mouth_sounds,
            NonverbalField::ReflexSounds => self.reflex_sounds,
        }
        .unwrap_or(true)
    }
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

/// `True` stays distinct from an empty dict so a round-tripped config is not rewritten.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for NonverbalSounds {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::{PyAnyMethods as _, PyBoolMethods as _};
        if let Ok(flag) = obj.cast::<pyo3::types::PyBool>() {
            return Ok(if flag.is_true() { NonverbalSounds::On } else { NonverbalSounds::Off });
        }
        if !obj.hasattr("get")? {
            return Err(pyo3::exceptions::PyTypeError::new_err("nonverbal_sounds must be a bool or a NonverbalOptions mapping"));
        }
        Ok(NonverbalSounds::Per(obj.extract()?))
    }
}

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

fn steering_removed(
    table: &'static [(NonverbalField, &'static [&'static str])],
    steering: Option<&SpeechSteering>,
) -> Vec<&'static str> {
    let nonverbals = steering.and_then(|s| s.nonverbal_sounds.as_ref());
    // `labels is None`: the provider has no table, so there is nothing to remove.
    if table.is_empty() {
        return Vec::new();
    }
    match nonverbals {
        None | Some(NonverbalSounds::On) => Vec::new(),
        Some(NonverbalSounds::Off) => table.iter().flat_map(|(_, l)| l.iter().copied()).collect(),
        Some(NonverbalSounds::Per(flags)) => table
            .iter()
            .filter(|(f, _)| !flags.enabled(*f))
            .flat_map(|(_, l)| l.iter().copied())
            .collect(),
    }
}

pub fn allowed_sounds(provider: Provider, steering: Option<&SpeechSteering>) -> Vec<&'static str> {
    let removed = steering_removed(nonverbal_sound_labels(provider), steering);
    provider.sounds().iter().copied().filter(|s| !removed.contains(s)).collect()
}

/// Only vocal-style labels are governed, so the result is never empty.
pub fn allowed_prosody(provider: Provider, steering: Option<&SpeechSteering>) -> Vec<&'static str> {
    let removed = steering_removed(nonverbal_prosody_labels(provider), steering);
    provider.prosody().iter().copied().filter(|p| !removed.contains(p)).collect()
}

fn sound_guidance(sounds: &[&str]) -> String {
    let mut hints: Vec<&'static str> = Vec::new();
    for sound in sounds {
        if let Some(hint) = sound_usage_hint(sound)
            && !hints.contains(&hint)
        {
            hints.push(hint);
        }
    }
    let mut line =
        String::from("Non-verbal sounds: use one only where the moment genuinely earns it");
    if !hints.is_empty() {
        line.push_str(" — ");
        line.push_str(&hints.join(", "));
    }
    line.push_str(". Most turns have none; never repeat the same sound twice in a row.");
    line
}

/// `""` when nothing changes the default (the caller tests truthiness).
pub fn steering_instructions(provider: Option<Provider>, steering: &SpeechSteering) -> String {
    let mut lines: Vec<String> = Vec::new();

    // Only when steering removes part of the vocabulary and something remains: both are container
    // truthiness in the reference, so `True`, `{}` and `False` all render nothing.
    if let Some(provider) = provider
        && !steering_removed(nonverbal_sound_labels(provider), Some(steering)).is_empty()
    {
        let allowed = allowed_sounds(provider, Some(steering));
        if !allowed.is_empty() {
            lines.push(sound_guidance(&allowed));
        }
    }

    if let Some(disfluencies) = steering.disfluencies {
        lines.push(
            if disfluencies {
                "Sprinkle in natural fillers (um, uh) and openers (oh, well, so), zero to two per \
                 turn, never mechanical."
            } else {
                "No fillers (um, uh). Sound composed and fluent."
            }
            .to_string(),
        );
    }

    if let Some(pace) = steering.pace
        && pace != Pace::Normal
    {
        lines.push(format!("Keep a {} overall speaking pace.", pace.as_str()));
    }

    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::from("Delivery guidelines:\n");
    out.push_str(&lines.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `FromStr` and `as_str` disagreeing, or an unknown pace read as one of the three.
    #[test]
    fn pace_round_trips_its_three_words_and_refuses_the_rest() {
        for p in Pace::ALL {
            assert_eq!(p.as_str().parse::<Pace>(), Ok(p));
        }
        assert_eq!("brisk".parse::<Pace>().unwrap_err().to_string(), "unknown pace \"brisk\"");
    }
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use super::*;
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;

    // Catches: `nonverbal_sounds` disagreeing with the declaration about absent-vs-`None`, or the
    // bool half being read through the dict reader.
    #[test]
    fn steering_reads_absent_and_none_alike_and_keeps_the_three_nonverbal_cases_apart() {
        Python::initialize();
        Python::attach(|py| {
            let read = |src: &str| -> PyResult<SpeechSteering> {
                py.eval(&std::ffi::CString::new(src).unwrap(), None, None)?.extract::<SpeechSteering>()
            };
            assert_eq!(read("{}").unwrap(), SpeechSteering::default());
            assert_eq!(
                read("{'disfluencies': None, 'nonverbal_sounds': None, 'pace': None}").unwrap(),
                SpeechSteering::default(),
                "an explicit None is absent, as the reference's `is not None` makes it"
            );
            let full = read("{'disfluencies': False, 'nonverbal_sounds': {'laughing': False, 'other': 1}, 'pace': 'slow'}")
                .unwrap();
            assert_eq!(full.disfluencies, Some(false));
            assert_eq!(full.pace, Some(Pace::Slow));
            match full.nonverbal_sounds {
                Some(NonverbalSounds::Per(opts)) => {
                    assert_eq!(opts.laughing, Some(false));
                    assert_eq!(opts.breathing, None, "an omitted category stays absent");
                }
                other => panic!("a dict is the per-category case, got {other:?}"),
            }
            assert_eq!(read("{'nonverbal_sounds': True}").unwrap().nonverbal_sounds, Some(NonverbalSounds::On));
            assert_eq!(read("{'nonverbal_sounds': False}").unwrap().nonverbal_sounds, Some(NonverbalSounds::Off));
            let err = read("{'pace': 'brisk'}").unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(err.value(py).to_string(), "unknown pace \"brisk\"");
            let err = read("{'nonverbal_sounds': 3}").unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }
}

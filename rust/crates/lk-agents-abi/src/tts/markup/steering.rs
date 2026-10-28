//! Sparse speech steering configuration.

use lk_agents_host_derive::Host;

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
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "nonverbal_sounds must be a bool or a NonverbalOptions mapping",
            ));
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
                py.eval(&std::ffi::CString::new(src).unwrap(), None, None)?
                    .extract::<SpeechSteering>()
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
            assert_eq!(
                read("{'nonverbal_sounds': True}").unwrap().nonverbal_sounds,
                Some(NonverbalSounds::On)
            );
            assert_eq!(
                read("{'nonverbal_sounds': False}").unwrap().nonverbal_sounds,
                Some(NonverbalSounds::Off)
            );
            let err = read("{'pace': 'brisk'}").unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
            assert_eq!(err.value(py).to_string(), "unknown pace \"brisk\"");
            let err = read("{'nonverbal_sounds': 3}").unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }
}

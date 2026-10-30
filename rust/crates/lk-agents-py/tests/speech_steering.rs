//! `SpeechSteeringOptions` as the host writes it: absent and `None` keys alike, and
//! `nonverbal_sounds` as `True`, `False` or a per-category mapping.

use lk_agents_abi::tts::markup::{NonverbalSounds, Pace, SpeechSteering};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

// Catches: `nonverbal_sounds` disagreeing with the declaration about absent-vs-`None`, or the
// bool half being read through the dict reader.
#[test]
fn steering_reads_absent_and_none_alike_and_keeps_the_three_nonverbal_cases_apart() {
    Python::initialize();
    Python::attach(|py| {
        // `NOT_GIVEN` is the package's: its dependencies' `.pth` installs need a site directory.
        let setup = concat!(
            "import site, sys\n",
            "for path in [p for p in sys.path if p.endswith('site-packages')]:\n",
            "    site.addsitedir(path)\n",
            "sys.path.insert(0, '",
            env!("CARGO_MANIFEST_DIR"),
            "/../../../livekit-agents-next')\n",
        );
        py.run(&std::ffi::CString::new(setup).unwrap(), None, None).unwrap();
        let read = |src: &str| -> PyResult<SpeechSteering> {
            _core::generated::records::expressive::speech_steering_options_from_py(
                &py.eval(&std::ffi::CString::new(src).unwrap(), None, None)?,
            )
        };
        assert_eq!(read("{}").unwrap(), SpeechSteering::default());
        assert_eq!(
            read("{'disfluencies': None, 'nonverbal_sounds': None, 'pace': None}").unwrap(),
            SpeechSteering::default(),
            "an explicit None is absent, as the reference's `is not None` makes it"
        );
        let full = read(
            "{'disfluencies': False, 'nonverbal_sounds': {'laughing': False, 'other': 1}, 'pace': 'slow'}",
        )
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
            Some(NonverbalSounds::Enabled(true))
        );
        assert_eq!(
            read("{'nonverbal_sounds': False}").unwrap().nonverbal_sounds,
            Some(NonverbalSounds::Enabled(false))
        );
        let err = read("{'pace': 'brisk'}").unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
        assert_eq!(err.value(py).to_string(), "Pace: \"brisk\" is not one of 'slow', 'normal', 'fast'");
        let err = read("{'nonverbal_sounds': 3}").unwrap_err();
        assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
    });
}

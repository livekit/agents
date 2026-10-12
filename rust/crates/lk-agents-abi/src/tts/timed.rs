//! `TimedString` (`types.py:152`). The Python type stays a `str` subclass (pyo3 cannot subclass
//! `str`). An explicit `start_time=None` from the host reads as not given.

use lk_agents_host_derive::Host;

/// A word (or chunk) with optional start/end times for alignment.
///
/// Built as the host's own `TimedString`, an absent timing an absent keyword. Read with
/// `NOT_GIVEN`, `None` and anything unreadable folded to `None`; `speaker_id` may be missing on
/// older subclasses. Declared for the Node binding's record; the Python crossing is written below.
#[derive(Debug, Clone, Default, PartialEq, Host)]
pub struct TimedString {
    /// The text itself: `TimedString(text=word, ..)` out, and the `str` the object is on the way in.
    pub text: String,
    /// Word start time in seconds.
    pub start_time: Option<f64>,
    /// Word end time in seconds.
    pub end_time: Option<f64>,
    /// Per-word confidence score.
    pub confidence: Option<f64>,
    /// Offset in seconds relative to the start of the audio input stream or session. Used by STT
    /// plugins to align words against the session timeline.
    pub start_time_offset: Option<f64>,
    /// Speaker identifier when the provider supports diarization.
    pub speaker_id: Option<String>,
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &TimedString {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        static CLASS: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("text", &self.text)?;
        for (key, v) in [
            ("start_time", self.start_time),
            ("end_time", self.end_time),
            ("confidence", self.confidence),
            ("start_time_offset", self.start_time_offset),
        ] {
            if let Some(v) = v {
                kwargs.set_item(key, v)?;
            }
        }
        if let Some(v) = &self.speaker_id {
            kwargs.set_item("speaker_id", v)?;
        }
        CLASS.import(py, "livekit.agents.types", "TimedString")?.call((), Some(&kwargs))
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for TimedString {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        (&self).into_pyobject(py)
    }
}

/// `NOT_GIVEN`, `None` and anything else unreadable are absent.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for TimedString {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr_or, attr_with};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "TimedStringFromPy";
        let obj = &*obj;
        let lenient = |key: &str| -> pyo3::PyResult<Option<f64>> { attr_or(obj, OWNER, key, |v| Ok(v.extract().ok()), || None) };
        Ok(TimedString {
            text: attr_with(obj, OWNER, "__str__", |v| v.call0()?.extract())?,
            start_time: lenient("start_time")?,
            end_time: lenient("end_time")?,
            confidence: lenient("confidence")?,
            start_time_offset: lenient("start_time_offset")?,
            speaker_id: attr_or(obj, OWNER, "speaker_id", |v| Ok(v.extract().ok()), || None)?,
        })
    }
}

impl TimedString {
    pub fn new(text: impl Into<String>) -> Self {
        TimedString { text: text.into(), ..Default::default() }
    }
}

/// The text only, as `str.__eq__` does.
impl PartialEq<str> for TimedString {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for TimedString {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `PartialEq` comparing the timings as well as the text.
    #[test]
    fn comparison_against_a_bare_str_ignores_the_timings() {
        let token = TimedString { start_time: Some(1.5), ..TimedString::new("hello") };
        assert_eq!(token, *"hello");
        assert_eq!(token, "hello");
        assert_ne!(token, *"hell");
        assert_ne!(token, "hell");
    }
}

//! `TimedString` (`types.py:152`). The Python type stays a `str` subclass (pyo3 cannot subclass
//! `str`). An explicit `start_time=None` from the host reads as not given.

use lk_agents_host_derive::Host;

/// A word (or chunk) with optional start/end times for alignment.
///
/// Built as the host's own `TimedString`, an absent timing an absent keyword. Read with
/// `NOT_GIVEN`, `None` and anything unreadable folded to `None`; `speaker_id` may be missing on
/// older subclasses. Declared for the Node binding's record; the Python crossing is
/// `lk-agents-py`'s `conversions::timed_string_{from,to}_py`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[crate::host(handwritten(extract, emit, host = "..types.TimedString", py))]
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

//! The VAD pipeline events (`vad.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::MediaRef;

use lk_agents_host_derive::Host;

// `vad.py:19`, a `(str, Enum)`: the wire strings are public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Host)]
#[python(enum, module = "vad", path = "livekit.agents.vad", name = "VADEventType")]
#[node(enum)]
#[host(wire(
    err = crate::llm::chat_context::UnknownLiteral,
    msg = crate::llm::chat_context::UnknownLiteral { kind: "VADEventType", value: s.to_owned() },
))]
pub enum VadEventType {
    #[host(rename = "start_of_speech")]
    StartOfSpeech,
    #[host(rename = "inference_done")]
    InferenceDone,
    #[host(rename = "end_of_speech")]
    EndOfSpeech,
}

///
/// Represents an event detected by the Voice Activity Detector (VAD).
//
// The `///` above is the reference's docstring, blank first line included. Fires ~every 32 ms per
// stream, so the audio is one [`MediaRef`] rather than a `Vec` of frames; `None` is an empty list.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(dataclass, module = "vad", path = "livekit.agents.vad", name = "VADEvent")]
#[node(interface)]
// Read by hand (`FromPyObject` below): `frames` is interned only when it holds audio.
#[binding(emit)]
pub struct VadEvent {
    /// Type of the VAD event (e.g., start of speech, end of speech, inference done).
    // A string both ways, as `SpeechEvent::kind`.
    #[host(rename = "type")]
    pub kind: VadEventType,
    /// Index of the audio sample where the event occurred, relative to the inference sample rate.
    pub samples_index: u64,
    /// Timestamp (in seconds) when the event was fired.
    pub timestamp: f64,
    /// Duration of the speech segment in seconds.
    pub speech_duration: f64,
    /// Duration of the silence segment in seconds.
    pub silence_duration: f64,
    ///
    /// List of audio frames associated with the speech.
    ///
    /// - For `start_of_speech` events, this contains the audio chunks that triggered the detection.
    /// - For `inference_done` events, this contains the audio chunks that were processed.
    /// - For `end_of_speech` events, this contains the complete user speech.
    //
    // Not emitted: `VadOutput.frames` owns the `frames` key. Read without interning an empty list
    // (`inference_done` fires ~30/s; a handle each would leak).
    #[python(ty = "list[rtc.AudioFrame]", import = "livekit:rtc", default_factory = "list")]
    #[node(ty = "unknown[]")]
    #[binding(skip)]
    pub frames: Option<MediaRef>,
    /// Probability that speech is present (only for `INFERENCE_DONE` events).
    #[python(default = "0.0")]
    pub probability: f64,
    /// Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events).
    #[python(default = "0.0")]
    pub inference_duration: f64,
    /// Indicates whether speech was detected in the frames.
    #[python(default = "False")]
    pub speaking: bool,
    /// Threshold used to detect silence.
    // (The reference's words; it holds the accumulated silence in seconds.)
    #[python(default = "0.0")]
    pub raw_accumulated_silence: f64,
    /// Threshold used to detect speech.
    // (Likewise: the accumulated speech in seconds.)
    #[python(default = "0.0")]
    pub raw_accumulated_speech: f64,
}

/// `frames` is one [`MediaRef`] when it holds audio, `None` for `None` or `[]`.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for VadEvent {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{attr, attr_with, intern_media};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "VadEventFromPy";
        let obj = &*obj;
        Ok(VadEvent {
            kind: attr(obj, OWNER, "type")?,
            samples_index: attr(obj, OWNER, "samples_index")?,
            timestamp: attr(obj, OWNER, "timestamp")?,
            speech_duration: attr(obj, OWNER, "speech_duration")?,
            silence_duration: attr(obj, OWNER, "silence_duration")?,
            frames: attr_with(obj, OWNER, "frames", |v| {
                Ok((!v.is_none() && v.len().unwrap_or(0) > 0).then(|| intern_media(v.clone().unbind())))
            })?,
            probability: attr(obj, OWNER, "probability")?,
            inference_duration: attr(obj, OWNER, "inference_duration")?,
            speaking: attr(obj, OWNER, "speaking")?,
            raw_accumulated_silence: attr(obj, OWNER, "raw_accumulated_silence")?,
            raw_accumulated_speech: attr(obj, OWNER, "raw_accumulated_speech")?,
        })
    }
}

impl VadEvent {
    pub fn new(
        kind: VadEventType,
        samples_index: u64,
        timestamp: f64,
        speech_duration: f64,
        silence_duration: f64,
    ) -> Self {
        VadEvent {
            kind,
            samples_index,
            timestamp,
            speech_duration,
            silence_duration,
            frames: None,
            probability: 0.0,
            inference_duration: 0.0,
            speaking: false,
            raw_accumulated_silence: 0.0,
            raw_accumulated_speech: 0.0,
        }
    }
}

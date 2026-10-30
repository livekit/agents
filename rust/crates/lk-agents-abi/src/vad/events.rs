//! The VAD pipeline events (`vad.py`). A `///` on a Python-targeted declaration is the
//! reference's docstring verbatim; notes for a Rust reader are `//`.

use crate::AudioFrame;

// `vad.py:19`, a `(str, Enum)`: the wire strings are public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, lk_agents_host_derive::Vocabulary)]
#[crate::host(record(
    module = "vad",
    path = "livekit.agents",
    name = "VADEventType",
    enum,
    extract,
    emit
))]
#[crate::host(wire_strs(
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
// The `///` above is the reference's docstring, blank first line included.
// Read by hand (`conversions::vad_event_from_py`): `frames` is read only for an end of speech.
#[crate::host(record(
    module = "vad",
    path = "livekit.agents",
    name = "VADEvent",
    dataclass,
    extract(py),
    emit,
))]
#[derive(Debug, Clone, PartialEq)]
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
    // Read from a host VAD only for an end of speech, the one event whose audio is recognized
    // (`inference_done` fires ~30/s).
    #[host(default_factory = "list")]
    pub frames: Vec<AudioFrame>,
    /// Probability that speech is present (only for `INFERENCE_DONE` events).
    #[host(default = 0.0)]
    pub probability: f64,
    /// Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events).
    #[host(default = 0.0)]
    pub inference_duration: f64,
    /// Indicates whether speech was detected in the frames.
    #[host(default = false)]
    pub speaking: bool,
    /// Threshold used to detect silence.
    // (The reference's words; it holds the accumulated silence in seconds.)
    #[host(default = 0.0)]
    pub raw_accumulated_silence: f64,
    /// Threshold used to detect speech.
    // (Likewise: the accumulated speech in seconds.)
    #[host(default = 0.0)]
    pub raw_accumulated_speech: f64,
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
            frames: Vec::new(),
            probability: 0.0,
            inference_duration: 0.0,
            speaking: false,
            raw_accumulated_silence: 0.0,
            raw_accumulated_speech: 0.0,
        }
    }
}

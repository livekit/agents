//! Telemetry contracts: the wire attribute names, the policy and provider metric declarations, the
//! session report's records. Rust emits completed records; the host materialises them in its own
//! OTel SDK.

pub mod policy;
pub mod provider;

/// Every string is a dashboard wire contract. None carries user content.
pub mod attrs {
    pub const SPEECH_ID: &str = "lk.speech_id";
    /// Provider correlation ids, for joining against provider logs.
    pub const PROVIDER_REQUEST_IDS: &str = "lk.provider_request_ids";
    pub const TTS_LABEL: &str = "lk.tts.label";
    pub const TTS_STREAMING: &str = "lk.tts.streaming";

    pub const IS_INTERRUPTION: &str = "lk.is_interruption";
    pub const INTERRUPTION_PROBABILITY: &str = "lk.interruption.probability";
    pub const INTERRUPTION_TOTAL_DURATION: &str = "lk.interruption.total_duration";
    pub const INTERRUPTION_PREDICTION_DURATION: &str = "lk.interruption.prediction_duration";
    pub const INTERRUPTION_DETECTION_DELAY: &str = "lk.interruption.detection_delay";

    /// Not declared in `trace_types.py`: adding them would fail its byte-for-byte comparison with
    /// the reference.
    pub mod node {
        pub const METRIC: &str = "lk.metric";
        /// `root`, `live`, `stale` or `unknown`.
        pub const FRAME_RESOLUTION: &str = "lk.frame.resolution";
        /// The TTS reported no duplex path, so the adapter was selected. Not a failure.
        pub const TTS_NO_DUPLEX_PATH: &str = "lk.tts.no_duplex_path";
        pub const TTS_FRAMES: &str = "lk.tts.frames";
        pub const TTS_BYTES: &str = "lk.tts.bytes";
        pub const ERROR_RETRYABLE: &str = "lk.error.retryable";
    }
}

pub mod attributes;
pub mod report;

pub mod observation;
pub mod transport;
pub mod upload;

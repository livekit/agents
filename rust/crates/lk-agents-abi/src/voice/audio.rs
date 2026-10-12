//! `MultiSpeakerAdapter`'s `_speaker_filter_config()` dict. The detector lives in
//! `lk-agents-voice/src/audio.rs`.

use lk_agents_host_derive::Host;

/// `MultiSpeakerAdapter(...)`'s options as the host hands them to the driver, read flat.
// One filter per STT stream: the driver builds a fresh one each time the STT leg opens.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items)]
pub struct SpeakerFilterConfig {
    pub detect_primary_speaker: bool,
    pub suppress_background_speaker: bool,
    #[binding(flatten)]
    pub options: PrimarySpeakerOptions,
    #[binding(flatten)]
    pub formats: SpeakerFormats,
}

/// `PrimarySpeakerDetectionOptions`, declared for its field list ([`SpeakerFilterConfig`] flattens it).
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(fields_only)]
pub struct PrimarySpeakerOptions {
    /// Frame size for RMS computation.
    pub frame_size_ms: u32,
    /// How long to keep RMS data.
    pub rms_buffer_duration: f64,
    /// Minimum RMS samples needed for a speech event.
    pub min_rms_samples: usize,
    /// Smoothing factor for a speaker's RMS: `rms = rms * factor + new_rms * (1 - factor)`.
    pub rms_smoothing_factor: f64,
    /// A candidate's RMS must be louder than the current primary's by this multiplier.
    pub threshold_multiplier: f64,
    /// Time to decay from `threshold_multiplier` to 1.0 (equal levels).
    pub decay_to_equal_time: f64,
    /// Minimum threshold multiplier -- how much quieter a candidate may be and still take over.
    pub threshold_min_multiplier: f64,
}

impl Default for PrimarySpeakerOptions {
    fn default() -> Self {
        PrimarySpeakerOptions {
            frame_size_ms: 100,
            rms_buffer_duration: 120.0,
            min_rms_samples: 3,
            rms_smoothing_factor: 0.5,
            threshold_multiplier: 1.3,
            decay_to_equal_time: 60.0,
            threshold_min_multiplier: 0.5,
        }
    }
}

/// How a transcript should be rendered, per speaker class (keyed `primary_format`/`background_format`).
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(fields_only)]
pub struct SpeakerFormats {
    /// Supports `{text}` and `{speaker_id}`.
    #[host(rename = "primary_format")]
    pub primary: String,
    /// Supports `{text}` and `{speaker_id}`.
    #[host(rename = "background_format")]
    pub background: String,
}

impl Default for SpeakerFormats {
    fn default() -> Self {
        SpeakerFormats { primary: "{text}".to_string(), background: "{text}".to_string() }
    }
}

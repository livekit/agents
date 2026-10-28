//! The end-of-turn pipeline event (`voice/turn.py`). A `///` on a Python-targeted declaration is
//! the reference's docstring verbatim; notes for a Rust reader are `//`.

/// One end-of-turn prediction. Optional timing and backchannel data depend on the detector.
///
/// Read from the generated prediction record returned by a host detector's `_core_predict()`.
#[crate::host(record(module = "turn_events", extract))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EotPrediction {
    /// How likely the user has finished their turn, `[0, 1]`.
    pub end_of_turn_probability: f64,
    #[host(default = 0.0)]
    pub last_speaking_time: f64,
    /// Latest input audio creation time -> prediction receive time.
    #[host(default = None)]
    pub detection_delay: Option<f64>,
    /// Server-side model inference time.
    #[host(default = None)]
    pub inference_duration: Option<f64>,
    /// How appropriate it is to backchannel at this pause; `None` when the detector has none.
    #[host(default = None)]
    pub backchannel_probability: Option<f64>,
}

impl EotPrediction {
    pub fn new(end_of_turn_probability: f64, last_speaking_time: f64) -> Self {
        EotPrediction {
            end_of_turn_probability,
            last_speaking_time,
            detection_delay: None,
            inference_duration: None,
            backchannel_probability: None,
        }
    }
}

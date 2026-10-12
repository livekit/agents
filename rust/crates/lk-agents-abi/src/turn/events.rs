//! The end-of-turn pipeline event (`voice/turn.py`). A `///` on a Python-targeted declaration is
//! the reference's docstring verbatim; notes for a Rust reader are `//`.

use lk_agents_host_derive::Host;

/// One end-of-turn prediction (`TurnDetectionEvent`, `voice/turn.py:23`). The struct is the
/// `type` discriminator; the `Option`s are optional data a detector may not produce.
///
/// Read off a host detector's `_core_predict()`, a dict: the probability is required, the rest
/// optional.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(extract, from_items)]
pub struct EotPrediction {
    /// How likely the user has finished their turn, `[0, 1]`.
    pub end_of_turn_probability: f64,
    #[binding(default)]
    pub last_speaking_time: f64,
    /// Latest input audio creation time -> prediction receive time.
    #[binding(default)]
    pub detection_delay: Option<f64>,
    /// Server-side model inference time.
    #[binding(default)]
    pub inference_duration: Option<f64>,
    /// How appropriate it is to backchannel at this pause; `None` when the detector has none.
    #[binding(default)]
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

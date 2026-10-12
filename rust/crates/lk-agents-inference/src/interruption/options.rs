//! `InterruptionOptions` and the module constants.

use thiserror::Error;

/// The gateway's fixed input rate (`inference/interruption.py:50`).
pub const SAMPLE_RATE: u32 = 16_000;

/// 25 ms per model frame, two consecutive frames: the shortest thing the server calls a barge-in.
pub const MIN_INTERRUPTION_DURATION: f64 = 0.025 * 2.0;

/// Ceiling on the rolling window, prefix included; larger is rejected (the model was trained on
/// 3 s).
pub const MAX_AUDIO_DURATION: f64 = 3.0;

/// How often the window is shipped once an overlap is open.
pub const DETECTION_INTERVAL: f64 = 0.1;

/// Audio from before the overlap kept in the first request: the model hears the agent alone first.
pub const AUDIO_PREFIX_DURATION: f64 = 1.0;

/// How long the oldest unanswered request may sit before the stream fails the connection.
pub const REMOTE_INFERENCE_TIMEOUT: f64 = 0.7;

/// `_FRAMES_PER_SECOND`. 40 = 1 / 0.025.
pub const FRAMES_PER_SECOND: f64 = 40.0;

/// `math.ceil(min_interruption_duration * _FRAMES_PER_SECOND)`, saturating (never a silent 0).
#[lk_agents_abi::host(export(name = "interruption_min_frames"))]
pub fn min_frames_for(min_interruption_duration: f64) -> u32 {
    let frames = (min_interruption_duration * FRAMES_PER_SECOND).ceil();
    if frames.is_nan() { 0 } else { frames.clamp(0.0, u32::MAX as f64) as u32 }
}

/// What `AdaptiveInterruptionDetector.__init__` raises.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OptionsError {
    #[error("max_audio_duration must be less than or equal to 3.0 seconds")]
    MaxAudioDurationTooLarge,
    #[error(
        "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
    )]
    MissingApiKey,
    #[error(
        "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
    )]
    MissingApiSecret,
}

/// `InterruptionOptions` (`inference/interruption.py:68`). `threshold`'s `NotGivenOr` collapses to
/// `Option` safely: `is_given` is the only test applied to it.
#[derive(Debug, Clone, PartialEq)]
pub struct InterruptionOptions {
    pub sample_rate: u32,
    /// `None` = use whatever the server recommends. The server, not this code, applies it.
    pub threshold: Option<f64>,
    pub min_frames: u32,
    pub max_audio_duration: f64,
    pub audio_prefix_duration: f64,
    pub detection_interval: f64,
    pub inference_timeout: f64,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
}

impl InterruptionOptions {
    /// The reference's constructor arguments, already resolved by the caller through the shared
    /// URL and credential helpers. An empty resolved key or secret is rejected.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        threshold: Option<f64>,
        min_interruption_duration: f64,
        max_audio_duration: f64,
        audio_prefix_duration: f64,
        detection_interval: f64,
        inference_timeout: f64,
        base_url: String,
        api_key: String,
        api_secret: String,
    ) -> Result<Self, OptionsError> {
        // Ordered as `__init__`: the duration check runs before the credentials.
        if max_audio_duration > MAX_AUDIO_DURATION {
            return Err(OptionsError::MaxAudioDurationTooLarge);
        }
        if api_key.is_empty() {
            return Err(OptionsError::MissingApiKey);
        }
        if api_secret.is_empty() {
            return Err(OptionsError::MissingApiSecret);
        }
        Ok(InterruptionOptions {
            sample_rate: SAMPLE_RATE,
            threshold,
            min_frames: min_frames_for(min_interruption_duration),
            max_audio_duration,
            audio_prefix_duration,
            detection_interval,
            inference_timeout,
            base_url,
            api_key,
            api_secret,
        })
    }

    /// `update_options` (`inference/interruption.py:391`). `None` means "not given" and leaves the
    /// field alone.
    pub fn update(&mut self, threshold: Option<f64>, min_interruption_duration: Option<f64>) {
        if let Some(threshold) = threshold {
            self.threshold = Some(threshold);
        }
        if let Some(duration) = min_interruption_duration {
            self.min_frames = min_frames_for(duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(max_audio_duration: f64, api_key: &str, api_secret: &str) -> Result<InterruptionOptions, OptionsError> {
        InterruptionOptions::new(
            None,
            MIN_INTERRUPTION_DURATION,
            max_audio_duration,
            AUDIO_PREFIX_DURATION,
            DETECTION_INTERVAL,
            REMOTE_INFERENCE_TIMEOUT,
            "https://gateway".to_owned(),
            api_key.to_owned(),
            api_secret.to_owned(),
        )
    }

    // Catches: `round` or truncation where the reference has `ceil`, or `as u32` turning a negative
    // or NaN duration into a wrapped count.
    #[test]
    fn min_frames_rounds_up_to_the_next_whole_25ms_frame() {
        for (secs, want) in [
            (MIN_INTERRUPTION_DURATION, 2),
            (0.06, 3),
            (0.075, 3),
            (0.001, 1),
            (0.0, 0),
            (-1.0, 0),
            (f64::NAN, 0),
            (f64::INFINITY, u32::MAX),
        ] {
            assert_eq!(min_frames_for(secs), want, "{secs}");
        }
    }

    // Catches: accepting an empty resolved credential; the two messages name different env vars.
    #[test]
    fn missing_credentials_name_the_right_variable() {
        assert_eq!(opts(MAX_AUDIO_DURATION, "", ""), Err(OptionsError::MissingApiKey));
        assert_eq!(opts(MAX_AUDIO_DURATION, "k", ""), Err(OptionsError::MissingApiSecret));
        assert!(opts(MAX_AUDIO_DURATION, "k", "s").is_ok());
    }

    // Catches: the duration guard drifting to `>=`, or running after the credentials.
    #[test]
    fn max_audio_duration_rejects_above_three_seconds_before_looking_at_credentials() {
        assert_eq!(opts(3.001, "", ""), Err(OptionsError::MaxAudioDurationTooLarge));
        assert!(opts(3.0, "k", "s").is_ok());
    }

    // Catches: `update_options` clearing the threshold when only `min_interruption_duration` is
    // given.
    #[test]
    fn update_options_leaves_unspecified_fields_alone() {
        let mut o = opts(MAX_AUDIO_DURATION, "k", "s").unwrap();
        o.threshold = Some(0.8);
        o.update(None, Some(0.1));
        assert_eq!(o.threshold, Some(0.8));
        assert_eq!(o.min_frames, 4);
        o.update(Some(0.5), None);
        assert_eq!((o.threshold, o.min_frames), (Some(0.5), 4));
    }
}

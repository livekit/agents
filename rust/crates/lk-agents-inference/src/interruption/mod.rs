use lk_agents_abi::errors::Invalid;
use lk_agents_abi::turn::interruption::{InterruptionPort, InterruptionSource};
use lk_agents_abi::{ApiConnectOptions, WallAnchor};
use lk_agents_net::env::ProcessEnv;
use std::sync::Arc;

mod audio;
mod event;
mod leg;
pub mod machine;
mod options;
pub mod proto;
pub mod transport;

pub use audio::{AudioError, AudioFrame, AudioWindow};
pub use event::{
    CacheEntry, OverlappingSpeechEvent, estimate_probability, interruption_metrics,
    overlapping_speech_from_cache_entry,
};
pub use leg::InterruptionLeg;
pub use machine::{
    InputEvent, InputOutcome, InterruptionEvent, InterruptionMachine, OverlapEnd, OverlapStart,
    RequestIdentity, StreamError,
};
pub use options::{
    AUDIO_PREFIX_DURATION, DETECTION_INTERVAL, FRAMES_PER_SECOND, InterruptionOptions,
    MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION, OptionsError, REMOTE_INFERENCE_TIMEOUT,
    SAMPLE_RATE, min_frames_for,
};
pub use transport::{
    AdaptiveInterruptionDetector, ConnectionConfig, InterruptionStream, StreamClosed, StreamEvent,
};

/// `AdaptiveInterruptionDetector.model`.
pub const MODEL: &str = "adaptive interruption";

/// `AdaptiveInterruptionDetector.provider`.
pub const PROVIDER: &str = "livekit";

/// The detector the session reads, its streams timed against the host's clock.
pub struct NativeInterruptionDetector {
    detector: AdaptiveInterruptionDetector,
}

impl lk_agents_abi::turn::inference::InterruptionDetector for NativeInterruptionDetector {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        threshold: Option<f64>,
        min_interruption_duration: f64,
        max_audio_duration: f64,
        audio_prefix_duration: f64,
        detection_interval: f64,
        inference_timeout: f64,
        base_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
        conn_options: Option<ApiConnectOptions>,
        token_ttl_secs: u64,
    ) -> Result<Self, Invalid> {
        use lk_agents_abi::inference::InferenceCredentials as _;
        let credentials =
            crate::credentials::Credentials::inference_credentials(base_url, api_key, api_secret)?;
        let opts = InterruptionOptions::new(
            threshold,
            min_interruption_duration,
            max_audio_duration,
            audio_prefix_duration,
            detection_interval,
            inference_timeout,
            credentials.base_url,
            credentials.api_key,
            credentials.api_secret,
        )
        .map_err(|e| Invalid(e.to_string()))?;
        // The session's stream owns its retries.
        let conn_options = ApiConnectOptions { max_retry: 0, ..conn_options.unwrap_or_default() };
        let config =
            ConnectionConfig { conn_options, token_ttl_secs, ..ConnectionConfig::default() };
        let detector = AdaptiveInterruptionDetector::new(opts, config, env.clock.clone());
        Ok(NativeInterruptionDetector { detector })
    }

    fn update_options(&self, threshold: Option<f64>, min_interruption_duration: Option<f64>) {
        self.detector.update_options(threshold, min_interruption_duration);
    }
}

impl InterruptionSource for NativeInterruptionDetector {
    fn open(&self, anchor: WallAnchor, cx: &crate::telemetry::Cx) -> Arc<dyn InterruptionPort> {
        InterruptionLeg::open(&self.detector, anchor, cx)
    }
}

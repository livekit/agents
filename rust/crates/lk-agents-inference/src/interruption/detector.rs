//! `_core.InterruptionDetector`: the detector as the host constructs it, natively the session's
//! [`InterruptionSource`].

use std::sync::Arc;

use lk_agents_abi::inference::InvalidOption;
use lk_agents_abi::turn::interruption::{InterruptionPort, InterruptionSource};
use lk_agents_abi::{ApiConnectOptions, WallAnchor};
use lk_agents_net::credentials::resolve_credential;

use super::{AdaptiveInterruptionDetector, ConnectionConfig, InterruptionLeg, InterruptionOptions};
use crate::headers::default_inference_url_from_env;
use lk_agents_net::env::ProcessEnv;

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
    ) -> Result<Self, InvalidOption> {
        // `base_url if base_url else ...`: an empty argument falls through to the environment.
        let given = |v: Option<String>| v.filter(|v| !v.is_empty());
        let var = |name: &str| std::env::var(name).ok();
        let credential = |explicit, inference: &str, livekit: &str| {
            given(explicit).unwrap_or_else(|| {
                resolve_credential(None, var(inference).as_deref(), var(livekit).as_deref())
            })
        };
        let opts = InterruptionOptions::new(
            threshold,
            min_interruption_duration,
            max_audio_duration,
            audio_prefix_duration,
            detection_interval,
            inference_timeout,
            given(base_url).unwrap_or_else(default_inference_url_from_env),
            credential(api_key, "LIVEKIT_INFERENCE_API_KEY", "LIVEKIT_API_KEY"),
            credential(api_secret, "LIVEKIT_INFERENCE_API_SECRET", "LIVEKIT_API_SECRET"),
        )
        .map_err(|e| InvalidOption(e.to_string()))?;
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
    fn open(&self, anchor: WallAnchor) -> Arc<dyn InterruptionPort> {
        InterruptionLeg::open(&self.detector, anchor)
    }
}

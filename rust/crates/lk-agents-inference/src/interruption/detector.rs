//! `_core.InterruptionDetector`: the detector as the host constructs it, natively the session's
//! [`InterruptionSource`].

use std::sync::Arc;

use lk_agents_abi::inference::InvalidOption;
use lk_agents_abi::turn::interruption::{InterruptionPort, InterruptionSource};
use lk_agents_abi::{ApiConnectOptions, WallAnchor};

use super::{AdaptiveInterruptionDetector, ConnectionConfig, InterruptionLeg, InterruptionOptions};
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
        let required = |value: Option<String>, name: &str| {
            value
                .filter(|v| !v.is_empty())
                .ok_or_else(|| InvalidOption(format!("{name} is required")))
        };
        let opts = InterruptionOptions::new(
            threshold,
            min_interruption_duration,
            max_audio_duration,
            audio_prefix_duration,
            detection_interval,
            inference_timeout,
            required(base_url, "base_url")?,
            required(api_key, "api_key")?,
            required(api_secret, "api_secret")?,
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

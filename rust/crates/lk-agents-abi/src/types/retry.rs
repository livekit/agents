//! Per-provider attempt policy. Hosts run callbacks and enact these decisions.
use super::connect::ApiConnectOptions;
use crate::errors::{ApiError, ApiErrorKind};

/// Crosses to a host as its tag (`FromStr` / `as_str`).
/// The operation whose failures share an attempt allowance.
#[crate::host(string)]
#[crate::host(wire_strs(err = UnknownRequestKind, msg = UnknownRequestKind(s.to_owned())))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    Llm,
    SttBatch,
    SttStream,
    TtsBatch,
    TtsStream,
    Interruption,
    Avatar,
}

/// `Display` does not interpolate the value: the binding has always raised the bare message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownRequestKind(pub String);

impl std::fmt::Display for UnknownRequestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unknown provider request kind")
    }
}

impl std::error::Error for UnknownRequestKind {}

#[derive(Debug, PartialEq)]
#[crate::host(record(module = "_retry", name = "_core_RetryDecision", emit))]
pub enum RetryDecision {
    Stop,
    Raise,
    #[host(field = "message")]
    Exhausted(String),
    #[host(field = "delay")]
    Retry(f64),
}

impl RequestKind {
    /// The attempt owner gave up. Timeouts keep their class/phase and caller-owned cause;
    /// other exhausted recognition/completion allowances carry one shared diagnostic.
    pub fn terminal_failure(self, error: &ApiError, max_retry: u32, retries: u32) -> RetryDecision {
        if max_retry == 0 || !error.retryable || error.kind == ApiErrorKind::Timeout {
            return RetryDecision::Raise;
        }
        exhaustion_message(self, retries).map_or(RetryDecision::Raise, RetryDecision::Exhausted)
    }
}

fn exhaustion_message(kind: RequestKind, retries: u32) -> Option<String> {
    let action = match kind {
        RequestKind::Llm => "generate LLM completion",
        RequestKind::SttBatch | RequestKind::SttStream => "recognize speech",
        RequestKind::Interruption => "detect interruption",
        RequestKind::TtsBatch | RequestKind::TtsStream | RequestKind::Avatar => return None,
    };
    Some(format!("failed to {action} after {} attempts", u64::from(retries) + 1))
}

pub fn decide(
    kind: RequestKind,
    options: &ApiConnectOptions,
    attempt: u32,
    retryable: bool,
    status: Option<i64>,
    produced_output: bool,
) -> RetryDecision {
    use RequestKind::*;
    if matches!(kind, Llm | TtsBatch | TtsStream) && status == Some(499) {
        return RetryDecision::Stop;
    }
    // Visible output forbids a retry (F-10): a second attempt would repeat it. A streaming
    // recognition reconnect sends only new audio, so it never repeats a delivered transcript.
    if options.max_retry == 0 || !retryable || (produced_output && !matches!(kind, SttStream)) {
        return RetryDecision::Raise;
    }
    if attempt >= options.max_retry {
        return exhaustion_message(kind, attempt)
            .map_or(RetryDecision::Raise, RetryDecision::Exhausted);
    }
    RetryDecision::Retry(options.interval_for_retry(attempt))
}

#[crate::host(export(pid = false))]
pub fn validate_synthesis(
    text: &str,
    duration: f64,
    #[host(required)] segments: Option<(usize, usize)>,
) -> Option<String> {
    if text.trim_matches(|c| crate::utils::strings::PYTHON_WHITESPACE.contains(&c)).is_empty() {
        return None;
    }
    if duration <= 0.0 {
        return Some(format!("no audio frames were pushed for text: {text}"));
    }
    if let Some((expected, actual)) = segments {
        if expected != actual {
            return Some(format!(
                "number of segments mismatch: expected {expected}, but got {actual}"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gateway_and_interruption_preserve_their_error_contracts() {
        let opts =
            ApiConnectOptions { max_retry: 2, retry_interval: 1.0, ..ApiConnectOptions::default() };
        assert_eq!(decide(RequestKind::Avatar, &opts, 2, true, None, false), RetryDecision::Raise);
        assert_eq!(
            decide(RequestKind::Avatar, &opts, 0, false, Some(400), false),
            RetryDecision::Raise
        );
        assert_eq!(
            decide(RequestKind::Interruption, &opts, 2, true, None, false),
            RetryDecision::Exhausted("failed to detect interruption after 3 attempts".into())
        );
        assert_eq!(
            decide(RequestKind::Interruption, &opts, 0, true, Some(499), false),
            RetryDecision::Retry(0.1)
        );
    }
    #[test]
    fn modality_and_partial_output_control_retries() {
        let opts =
            ApiConnectOptions { max_retry: 2, retry_interval: 1.0, ..ApiConnectOptions::default() };
        assert_eq!(decide(RequestKind::Llm, &opts, 0, false, Some(499), true), RetryDecision::Stop);
        assert_eq!(
            decide(RequestKind::TtsStream, &opts, 0, true, None, true),
            RetryDecision::Raise
        );
        for kind in [RequestKind::Llm, RequestKind::TtsBatch, RequestKind::Avatar] {
            assert_eq!(decide(kind, &opts, 0, true, None, true), RetryDecision::Raise);
        }
        assert_eq!(
            decide(RequestKind::TtsBatch, &opts, 0, true, None, false),
            RetryDecision::Retry(0.1)
        );
        // `STT.recognize` raises a non-retryable `APIError` at once, as every other kind does.
        assert_eq!(
            decide(RequestKind::SttBatch, &opts, 1, false, None, false),
            RetryDecision::Raise
        );
        assert_eq!(
            decide(RequestKind::SttBatch, &opts, 1, true, None, false),
            RetryDecision::Retry(1.0)
        );
        assert_eq!(
            decide(RequestKind::SttStream, &opts, 1, false, None, false),
            RetryDecision::Raise
        );
        assert!(matches!(
            decide(RequestKind::Llm, &opts, 2, true, None, false),
            RetryDecision::Exhausted(_)
        ));
        assert!(validate_synthesis("hello", 0.0, None).is_some());
        assert!(validate_synthesis("hello", 1.0, Some((2, 1))).is_some());
    }
}

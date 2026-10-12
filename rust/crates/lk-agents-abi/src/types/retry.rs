//! Per-provider attempt policy. Hosts run callbacks and enact these decisions.
use super::connect::ApiConnectOptions;

/// Crosses to a host as its tag (`FromStr` / `as_str`).
/// The wire strings are what `_provider_retry.py` passes.
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
pub enum RetryDecision {
    Stop,
    Raise,
    Exhausted(String),
    Retry(f64),
}

impl RetryDecision {
    pub fn kind_str(&self) -> &'static str {
        match self {
            RetryDecision::Stop => "stop",
            RetryDecision::Raise => "raise",
            RetryDecision::Exhausted(_) => "exhausted",
            RetryDecision::Retry(_) => "retry",
        }
    }
}

/// The decision for one failed attempt, as the host's retry loops read it: `(action, delay,
/// exhausted message)` -- only `retry` carries a delay and only `exhausted` a message.
#[crate::host(export(name = "provider_retry", pid = false))]
pub fn decide_for_host(
    kind: RequestKind,
    maximum: u32,
    interval: f64,
    attempt: u32,
    retryable: bool,
    status: Option<i64>,
    produced_output: bool,
) -> Result<(&'static str, f64, Option<String>), super::connect::ApiConnectOptionsError> {
    let options = ApiConnectOptions::new(maximum, interval, 0.0)?;
    let decision = decide(kind, &options, attempt, retryable, status, produced_output);
    let (delay, message) = match &decision {
        RetryDecision::Retry(delay) => (*delay, None),
        RetryDecision::Exhausted(message) => (0.0, Some(message.clone())),
        RetryDecision::Stop | RetryDecision::Raise => (0.0, None),
    };
    Ok((decision.kind_str(), delay, message))
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
    if options.max_retry == 0
        || (!matches!(kind, SttBatch) && !retryable)
        || (matches!(kind, TtsStream) && produced_output)
    {
        return RetryDecision::Raise;
    }
    if attempt >= options.max_retry {
        return match kind {
            Llm => RetryDecision::Exhausted(format!(
                "failed to generate LLM completion after {} attempts",
                options.max_retry as u64 + 1
            )),
            SttBatch => RetryDecision::Exhausted(format!(
                "failed to recognize speech after {} attempts",
                options.max_retry as u64 + 1
            )),
            SttStream => RetryDecision::Exhausted(format!(
                "failed to recognize speech after {attempt} attempts"
            )),
            Interruption => RetryDecision::Exhausted(format!(
                "failed to detect interruption after {attempt} attempts"
            )),
            TtsBatch | TtsStream | Avatar => RetryDecision::Raise,
        };
    }
    RetryDecision::Retry(options.interval_for_retry(attempt))
}

#[crate::host(export(pid = false))]
pub fn validate_synthesis(
    text: &str,
    duration: f64,
    #[host(required)] segments: Option<(usize, usize)>,
) -> Option<String> {
    if text.trim_matches(crate::tokenize::is_py_space).is_empty() {
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
        let opts = ApiConnectOptions::new(2, 1.0, 10.0).unwrap();
        assert_eq!(decide(RequestKind::Avatar, &opts, 2, true, None, false), RetryDecision::Raise);
        assert_eq!(
            decide(RequestKind::Avatar, &opts, 0, false, Some(400), false),
            RetryDecision::Raise
        );
        assert_eq!(
            decide(RequestKind::Interruption, &opts, 2, true, None, false),
            RetryDecision::Exhausted("failed to detect interruption after 2 attempts".into())
        );
        assert_eq!(
            decide(RequestKind::Interruption, &opts, 0, true, Some(499), false),
            RetryDecision::Retry(0.1)
        );
    }
    #[test]
    fn modality_and_partial_output_control_retries() {
        let opts = ApiConnectOptions::new(2, 1.0, 10.0).unwrap();
        assert_eq!(decide(RequestKind::Llm, &opts, 0, false, Some(499), true), RetryDecision::Stop);
        assert_eq!(
            decide(RequestKind::TtsStream, &opts, 0, true, None, true),
            RetryDecision::Raise
        );
        assert_eq!(
            decide(RequestKind::TtsBatch, &opts, 0, true, None, true),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(
            decide(RequestKind::SttBatch, &opts, 1, false, None, false),
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

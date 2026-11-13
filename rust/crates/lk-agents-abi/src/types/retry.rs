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

/// The one log of a retry: `attempt` (from 1) of `subject` failed with `error`, and the next runs
/// in `delay`. A request's retry is announced through `lk_agents_core::request::Backoff`, as a
/// warning; a loop no request drives (a background renewal, a reconnecting session) calls this
/// directly and says why its level differs.
pub fn retrying(
    level: log::Level,
    subject: &str,
    attempt: u32,
    delay: std::time::Duration,
    error: &dyn std::fmt::Display,
) {
    log::log!(
        target: crate::LOG_TARGET,
        level,
        (crate::telemetry::spans::common::log_keys::RETRY_SUBJECT) = subject,
        (crate::telemetry::spans::common::log_keys::RETRY_ATTEMPT) = attempt,
        (crate::telemetry::spans::common::log_keys::RETRY_DELAY) = delay.as_secs_f64();
        "{subject}: attempt {attempt} failed, retrying in {:.2}s: {error}",
        delay.as_secs_f64()
    );
}

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
    if options.max_retry == 0 || !retryable || (matches!(kind, TtsStream) && produced_output)
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

/// Who [`Attempts`] reports a failed attempt to: `_emit_error(error, recoverable)`.
#[crate::host(proxy)]
pub trait AttemptReport: Send + Sync {
    fn failed(&self, error: crate::errors::ProviderError, recoverable: bool);
}

/// `_core.Attempts`: the retry policy of `RecognizeStream._main_task` and `STT.recognize`, whose
/// loops attempt and hand each failure to [`failed`](Self::failed). An `APIError` [`decide`]
/// retries is reported recoverable and waited out; any other failure is reported and raised. A
/// stream's attempt re-seeds `start_time`, and `start_time_offset` grows by the time between
/// attempts, so transcripts keep linear timestamps.
pub struct Attempts {
    kind: RequestKind,
    options: ApiConnectOptions,
    label: String,
    state: std::sync::Mutex<AttemptState>,
}

#[derive(Default)]
struct AttemptState {
    retries: u32,
    /// When the current attempt began; `None` before the first.
    began: Option<f64>,
    start_time: f64,
    start_time_offset: f64,
}

fn wall_secs(clock: &dyn crate::Clock) -> f64 {
    clock.wall().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or_default()
}

#[crate::host(object(cell = "arc"))]
impl Attempts {
    #[host(new(kw_only = "label"))]
    pub fn for_host(
        kind: RequestKind,
        maximum: u32,
        interval: f64,
        timeout: f64,
        label: String,
        #[host(ambient)] clock: std::sync::Arc<dyn crate::Clock>,
    ) -> Result<Self, super::connect::ApiConnectOptionsError> {
        let options = ApiConnectOptions::new(maximum, interval, timeout)?;
        let state = AttemptState { start_time: wall_secs(&*clock), ..AttemptState::default() };
        Ok(Attempts { kind, options, label, state: std::sync::Mutex::new(state) })
    }

    /// `RecognizeStream.start_time`: the attempt's wall-clock anchor; a plugin may move it.
    #[host(getter)]
    pub fn start_time(&self) -> f64 {
        self.state.lock().unwrap().start_time
    }

    #[host(setter)]
    pub fn set_start_time(&self, value: f64) {
        self.state.lock().unwrap().start_time = value;
    }

    /// `RecognizeStream.start_time_offset`.
    #[host(getter)]
    pub fn start_time_offset(&self) -> f64 {
        self.state.lock().unwrap().start_time_offset
    }

    #[host(setter)]
    pub fn set_start_time_offset(&self, value: f64) {
        self.state.lock().unwrap().start_time_offset = value;
    }

    /// A recognition succeeded: the retry budget restarts.
    #[host(method)]
    pub fn reset(&self) {
        self.state.lock().unwrap().retries = 0;
    }

    /// An attempt begins: a stream's clock moves to now.
    #[host(method)]
    pub fn begin(&self, #[host(ambient)] clock: std::sync::Arc<dyn crate::Clock>) {
        let now = wall_secs(&*clock);
        let mut st = self.state.lock().unwrap();
        if self.kind == RequestKind::SttStream {
            if let Some(last) = st.began {
                st.start_time_offset += now - last;
            }
            st.start_time = now;
        }
        st.began = Some(now);
    }

    /// The attempt failed with `error`: reported to `report`, then waited out when it is retried;
    /// else raised (the budget spent: `APIConnectionError` caused by it).
    #[host(method)]
    pub async fn failed(
        &self,
        error: crate::errors::ProviderError,
        report: std::sync::Arc<dyn AttemptReport>,
        #[host(ambient)] clock: std::sync::Arc<dyn crate::Clock>,
    ) -> Result<(), crate::errors::ProviderError> {
        let streamed = self.kind == RequestKind::SttStream;
        let retries = {
            let mut st = self.state.lock().unwrap();
            // An attempt that outlived the connect timeout had connected: the budget restarts.
            let began = st.began.unwrap_or(st.start_time);
            if streamed && wall_secs(&*clock) - began > self.options.timeout {
                st.retries = 0;
            }
            st.retries
        };
        let decision = match error.api_kind() {
            None => RetryDecision::Raise,
            Some(_) => decide(self.kind, &self.options, retries, error.retryable(), error.status_code(), false),
        };
        let delay = match decision {
            RetryDecision::Retry(delay) => delay,
            RetryDecision::Exhausted(message) => {
                report.failed(error.clone(), false);
                return Err(crate::errors::chained_connection_error(message, &error));
            }
            RetryDecision::Raise | RetryDecision::Stop => {
                report.failed(error.clone(), false);
                return Err(error);
            }
        };
        report.failed(error.clone(), true);
        log::warn!(
            target: "livekit.agents",
            stt = self.label.as_str(), attempt = retries + u32::from(!streamed), streamed = streamed;
            "failed to recognize speech: {error}, retrying in {delay}s"
        );
        clock.wait_until(clock.now().saturating_add(std::time::Duration::from_secs_f64(delay.max(0.0)))).await;
        self.state.lock().unwrap().retries += 1;
        Ok(())
    }
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

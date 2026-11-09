//! The retry machine every retried request shares: phase limits, retry allowance and backoff,
//! total deadline and cleanup, decided on supplied time.

use lk_agents_abi::errors::{ApiError, ApiErrorKind};
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{ApiConnectOptions, ApiConnectOptionsError};
use std::time::Duration;

mod effects;
mod machine;

pub use effects::{Retry, Step};
pub use machine::Request;

/// The resolved per-request policy. `None` disables a limit. There is no pool limit: the
/// gateway transport has no pool capacity, so a request never waits for a connection slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    pub connect_timeout: Option<Duration>,
    pub write_timeout: Option<Duration>,
    pub read_timeout: Option<Duration>,
    pub total_timeout: Option<Duration>,
    /// Attempts after the first.
    pub max_retry: u32,
    pub retry_interval: Duration,
}

impl Policy {
    /// The shared retry allowance and delay; the owner supplies whether replay is eligible.
    pub fn retry_delay(&self, retries: u32, retryable: bool) -> Option<Duration> {
        if !retryable || retries >= self.max_retry {
            return None;
        }
        Some(if retries == 0 {
            Duration::from_secs_f64(lk_agents_abi::types::connect::FIRST_RETRY_INTERVAL)
        } else {
            self.retry_interval
        })
    }

    /// Phase limits of validated options. Pool capacity belongs to the transport.
    pub fn from_options(options: &ApiConnectOptions) -> Self {
        let limit = |seconds: Option<f64>| seconds.map(Duration::from_secs_f64);
        Policy {
            connect_timeout: limit(options.connect_timeout),
            write_timeout: limit(options.write_timeout),
            read_timeout: limit(options.read_timeout),
            total_timeout: limit(options.total_timeout),
            max_retry: options.max_retry,
            retry_interval: Duration::from_secs_f64(options.retry_interval),
        }
    }
}

/// Why an attempt failed, as the transport saw it.
#[derive(Debug, Clone, PartialEq)]
pub enum AttemptFailure {
    Rejected { status: i64, message: String, request_id: Option<String>, retryable: Option<bool> },
    Connection(String),
    Malformed(String),
}

impl AttemptFailure {
    pub fn into_error(self) -> ApiError {
        match self {
            Self::Rejected { status, message, request_id, retryable } => {
                let error = ApiError::from_http(&message, status, request_id);
                match retryable {
                    Some(retryable) => error.with_retryable(retryable),
                    None => error,
                }
            }
            Self::Connection(message) => ApiError::connection(Some(message)),
            Self::Malformed(message) => {
                ApiError::connection(Some(format!("malformed response: {message}")))
            }
        }
    }
}

/// The attempt owner gave up. Timeouts retain their phase and caller-owned cause.
pub fn terminal_failure(
    kind: RequestKind,
    error: &ApiError,
    max_retry: u32,
    retries: u32,
) -> RetryDecision {
    if max_retry == 0 || !error.retryable || error.kind == ApiErrorKind::Timeout {
        return RetryDecision::Raise;
    }
    let action = match kind {
        RequestKind::Llm => "generate LLM completion",
        RequestKind::SttBatch | RequestKind::SttStream => "recognize speech",
        RequestKind::Interruption => "detect interruption",
        RequestKind::TtsBatch | RequestKind::TtsStream | RequestKind::Avatar => {
            return RetryDecision::Raise;
        }
    };
    RetryDecision::Exhausted(format!(
        "failed to {action} after {} attempts",
        u64::from(retries) + 1
    ))
}

/// The host's `APIConnectOptions` rules: its boundary check and its retry interval.
pub struct ConnectOptionsRules;

impl lk_agents_abi::types::connect::ConnectOptionsFunctions for ConnectOptionsRules {
    fn api_connect_options_check(
        max_retry: i64,
        retry_interval: f64,
        pool_timeout: Option<f64>,
        connect_timeout: Option<f64>,
        write_timeout: Option<f64>,
        read_timeout: Option<f64>,
        total_timeout: Option<f64>,
    ) -> Result<(), ApiConnectOptionsError> {
        let max_retry = u32::try_from(max_retry).map_err(|_| ApiConnectOptionsError {
            field: "max_retry",
            requirement: "must be an integer >= 0",
        })?;
        ApiConnectOptions {
            max_retry,
            retry_interval,
            pool_timeout,
            connect_timeout,
            write_timeout,
            read_timeout,
            total_timeout,
        }
        .validate()
    }

    fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
        ApiConnectOptions { retry_interval, ..ApiConnectOptions::default() }
            .interval_for_retry(num_retries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::types::connect::ConnectOptionsFunctions as _;

    #[test]
    fn a_negative_max_retry_is_refused_by_name() {
        let err =
            ConnectOptionsRules::api_connect_options_check(-1, 2.0, None, None, None, None, None)
                .unwrap_err();
        assert_eq!(err.to_string(), "conn_options.max_retry must be an integer >= 0");
    }
}

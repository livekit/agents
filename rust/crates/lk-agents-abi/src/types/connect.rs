//! `APIConnectOptions` (`types.py:113`). The Python type stays a frozen dataclass (the fallback
//! adapters `dataclasses.replace` it).

use std::time::Duration;

use lk_agents_host_derive::Host;
use thiserror::Error;

use crate::Ts;

/// A field outside its declared range, named by its path.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("conn_options.{field} {requirement}")]
pub struct ApiConnectOptionsError {
    pub field: &'static str,
    pub requirement: &'static str,
}

/// Per-operation connection policy: retries, their interval and the limit of each wait.
///
/// The five phase limits are seconds, finite and non-negative, `None` disabling one: `pool` waits
/// for transport capacity, `connect` for a usable connection (DNS and TLS included), `write`
/// without progress while sending, `read` without progress while a response is expected, `total`
/// the whole operation (admission, retries and fallback). Each phase defaults to 10 s; `total`
/// is off.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[host(host_type = "livekit.agents.types.APIConnectOptions")]
#[binding(emit, extract)]
pub struct ApiConnectOptions {
    /// Attempts after the first.
    pub max_retry: u32,
    /// Seconds between retries; the first retry waits 0.1 s.
    pub retry_interval: f64,
    pub pool_timeout: Option<f64>,
    pub connect_timeout: Option<f64>,
    pub write_timeout: Option<f64>,
    pub read_timeout: Option<f64>,
    pub total_timeout: Option<f64>,
}

const DEFAULT_PHASE_TIMEOUT: Option<f64> = Some(10.0);

pub const DEFAULT_API_CONNECT_OPTIONS: ApiConnectOptions = ApiConnectOptions {
    max_retry: 3,
    retry_interval: 2.0,
    pool_timeout: DEFAULT_PHASE_TIMEOUT,
    connect_timeout: DEFAULT_PHASE_TIMEOUT,
    write_timeout: DEFAULT_PHASE_TIMEOUT,
    read_timeout: DEFAULT_PHASE_TIMEOUT,
    total_timeout: None,
};

impl Default for ApiConnectOptions {
    fn default() -> Self {
        DEFAULT_API_CONNECT_OPTIONS
    }
}

impl ApiConnectOptions {
    /// Every duration finite and non-negative, before any work starts. A native duration is
    /// built only from a validated value.
    pub fn validate(&self) -> Result<(), ApiConnectOptionsError> {
        let seconds = |field, value: f64| {
            if value.is_finite() && value >= 0.0 && value <= MAX_SECONDS {
                Ok(())
            } else {
                Err(ApiConnectOptionsError {
                    field,
                    requirement: "must be a finite number of seconds >= 0",
                })
            }
        };
        seconds("retry_interval", self.retry_interval)?;
        for (field, value) in [
            ("pool_timeout", self.pool_timeout),
            ("connect_timeout", self.connect_timeout),
            ("write_timeout", self.write_timeout),
            ("read_timeout", self.read_timeout),
            ("total_timeout", self.total_timeout),
        ] {
            if let Some(value) = value {
                seconds(field, value)?;
            }
        }
        Ok(())
    }

    /// A failed stream attempt that lasted beyond the connect limit had connected, so its failure
    /// is not consecutive with the earlier ones and the retry budget restarts (#7275): a socket a
    /// provider closes every few idle minutes never exhausts it. Without a connect limit no
    /// duration proves the attempt connected.
    pub fn attempt_connected(&self, lasted_seconds: f64) -> bool {
        self.connect_timeout.is_some_and(|limit| lasted_seconds > limit)
    }

    /// When a connection opened at `start` runs out of `connect_timeout`.
    pub fn connect_deadline(&self, start: Ts) -> Option<Ts> {
        after(start, self.connect_timeout)
    }

    /// When a write pending since `since` runs out of `write_timeout`.
    pub fn write_deadline(&self, since: Ts) -> Option<Ts> {
        after(since, self.write_timeout)
    }

    /// When a response expected since `since` runs out of `read_timeout`.
    pub fn read_deadline(&self, since: Ts) -> Option<Ts> {
        after(since, self.read_timeout)
    }

    /// When an operation accepted at `accepted` runs out of `total_timeout`.
    pub fn total_deadline(&self, accepted: Ts) -> Option<Ts> {
        after(accepted, self.total_timeout)
    }

    /// `types.py:138`. The first retry waits 100 ms, not zero as the docstring says (#1337). Shared
    /// so native and host retry loops agree.
    pub fn interval_for_retry(&self, num_retries: u32) -> f64 {
        if num_retries == 0 { FIRST_RETRY_INTERVAL } else { self.retry_interval }
    }
}

/// `seconds` (validated) after `start`; `None` disables the limit.
fn after(start: Ts, seconds: Option<f64>) -> Option<Ts> {
    seconds.map(|seconds| start.saturating_add(Duration::from_secs_f64(seconds)))
}

/// The largest duration accepted: a century, far below what `Duration::from_secs_f64` overflows on.
const MAX_SECONDS: f64 = 100.0 * 365.25 * 86_400.0;

/// The record as the host built it, checked at the boundary; the error names the field.
#[crate::host(export(pid = false))]
pub fn api_connect_options_check(
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

pub const FIRST_RETRY_INTERVAL: f64 = 0.1;

/// [`ApiConnectOptions::attempt_connected`] for the host's stream retry loop.
#[crate::host(export(pid = false))]
pub fn api_attempt_connected(connect_timeout: Option<f64>, lasted_seconds: f64) -> bool {
    ApiConnectOptions { connect_timeout, ..DEFAULT_API_CONNECT_OPTIONS }
        .attempt_connected(lasted_seconds)
}

/// `APIConnectOptions._interval_for_retry` (`types.py:138`) for the host's retry loops; only
/// `retry_interval` affects the answer.
#[crate::host(export(pid = false))]
pub fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
    ApiConnectOptions { max_retry: 0, retry_interval, ..DEFAULT_API_CONNECT_OPTIONS }
        .interval_for_retry(num_retries)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a 0.0 first retry (from the docstring), or retry #1 not back on the interval.
    #[test]
    fn only_the_first_retry_skips_the_configured_interval() {
        let opts = ApiConnectOptions::default();
        assert_eq!(opts.interval_for_retry(0), 0.1);
        assert_eq!(opts.interval_for_retry(1), 2.0);
        assert_eq!(opts.interval_for_retry(7), 2.0);
        // The 100 ms floor is the first step only, not a `max`.
        let opts = ApiConnectOptions { retry_interval: 0.0, ..ApiConnectOptions::default() };
        assert_eq!(opts.interval_for_retry(0), 0.1);
        assert_eq!(opts.interval_for_retry(1), 0.0);
    }

    // Catches: a negative, NaN or infinite duration accepted (NaN slipped past `< 0` checks and
    // infinity panicked in `Duration::from_secs_f64`), zero refused, or an error that does not
    // name its field.
    #[test]
    fn invalid_durations_fail_with_their_field_and_zero_or_none_is_valid() {
        let zero =
            ApiConnectOptions { max_retry: 0, retry_interval: 0.0, ..ApiConnectOptions::default() };
        assert_eq!(zero.validate(), Ok(()));
        let ok = ApiConnectOptions {
            read_timeout: None,
            total_timeout: Some(0.0),
            ..DEFAULT_API_CONNECT_OPTIONS
        };
        assert_eq!(ok.validate(), Ok(()));
        for bad in [-1.0, f64::NAN, f64::INFINITY, 1e300] {
            let err = ApiConnectOptions { read_timeout: Some(bad), ..DEFAULT_API_CONNECT_OPTIONS }
                .validate();
            assert_eq!(err.unwrap_err().field, "read_timeout", "{bad}");
            let err =
                ApiConnectOptions { retry_interval: bad, ..DEFAULT_API_CONNECT_OPTIONS }.validate();
            assert_eq!(err.unwrap_err().field, "retry_interval", "{bad}");
        }
        let err = api_connect_options_check(-1, 2.0, None, None, None, None, None).unwrap_err();
        assert_eq!(err.to_string(), "conn_options.max_retry must be an integer >= 0");
    }
}

/// Select the immutable host policy without reconstructing it. The constructor and native
/// operation boundary validate its typed fields.
#[crate::host(export(name = "_core_resolve_api_connect_options", pid = false))]
pub fn resolve_api_connect_options(
    provider: crate::HostRef,
    call: Option<crate::HostRef>,
) -> crate::HostRef {
    call.unwrap_or(provider)
}

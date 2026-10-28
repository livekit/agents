//! `APIConnectOptions` (`types.py:113`). The Python type stays a frozen dataclass (the fallback
//! adapters `dataclasses.replace` it).

use lk_agents_host_derive::Host;
use thiserror::Error;

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
///
/// `timeout` is the single limit of the providers not yet reading the phases (B-9 removes it with
/// their migration); nothing that reads the phases reads it.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(emit, extract)]
pub struct ApiConnectOptions {
    /// Attempts after the first.
    pub max_retry: u32,
    /// Seconds between retries; the first retry waits 0.1 s.
    pub retry_interval: f64,
    /// Legacy single limit, in seconds, of providers not yet reading the phase limits.
    pub timeout: f64,
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
    timeout: 10.0,
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
    /// The legacy three-field form; the phase limits keep their defaults.
    pub fn new(max_retry: u32, retry_interval: f64, timeout: f64) -> Result<Self, ApiConnectOptionsError> {
        let options = ApiConnectOptions { max_retry, retry_interval, timeout, ..DEFAULT_API_CONNECT_OPTIONS };
        options.validate()?;
        Ok(options)
    }

    /// Every duration finite and non-negative, before any work starts. A native duration is
    /// built only from a validated value.
    pub fn validate(&self) -> Result<(), ApiConnectOptionsError> {
        let seconds = |field, value: f64| {
            if value.is_finite() && value >= 0.0 && value <= MAX_SECONDS {
                Ok(())
            } else {
                Err(ApiConnectOptionsError { field, requirement: "must be a finite number of seconds >= 0" })
            }
        };
        seconds("retry_interval", self.retry_interval)?;
        seconds("timeout", self.timeout)?;
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

    /// `types.py:138`. The first retry waits 100 ms, not zero as the docstring says (#1337). Shared
    /// so native and host retry loops agree.
    pub fn interval_for_retry(&self, num_retries: u32) -> f64 {
        if num_retries == 0 { FIRST_RETRY_INTERVAL } else { self.retry_interval }
    }
}

/// The largest duration accepted: a century, far below what `Duration::from_secs_f64` overflows on.
const MAX_SECONDS: f64 = 100.0 * 365.25 * 86_400.0;

/// The record as the host built it, checked at the boundary; the error names the field.
#[crate::host(export(pid = false))]
pub fn api_connect_options_check(
    max_retry: i64,
    retry_interval: f64,
    timeout: f64,
    pool_timeout: Option<f64>,
    connect_timeout: Option<f64>,
    write_timeout: Option<f64>,
    read_timeout: Option<f64>,
    total_timeout: Option<f64>,
) -> Result<(), ApiConnectOptionsError> {
    let max_retry = u32::try_from(max_retry)
        .map_err(|_| ApiConnectOptionsError { field: "max_retry", requirement: "must be an integer >= 0" })?;
    ApiConnectOptions {
        max_retry,
        retry_interval,
        timeout,
        pool_timeout,
        connect_timeout,
        write_timeout,
        read_timeout,
        total_timeout,
    }
    .validate()
}

pub const FIRST_RETRY_INTERVAL: f64 = 0.1;

/// `APIConnectOptions._interval_for_retry` (`types.py:138`) for the host's retry loops; only
/// `retry_interval` affects the answer.
#[crate::host(export(pid = false))]
pub fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
    ApiConnectOptions { max_retry: 0, retry_interval, ..DEFAULT_API_CONNECT_OPTIONS }.interval_for_retry(num_retries)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a 0.0 first retry (from the docstring), or retry #1 not back on the interval.
    #[test]
    fn only_the_first_retry_skips_the_configured_interval() {
        let opts = ApiConnectOptions::new(3, 2.0, 10.0).unwrap();
        assert_eq!(opts.interval_for_retry(0), 0.1);
        assert_eq!(opts.interval_for_retry(1), 2.0);
        assert_eq!(opts.interval_for_retry(7), 2.0);
        // The 100 ms floor is the first step only, not a `max`.
        let opts = ApiConnectOptions::new(1, 0.0, 5.0).unwrap();
        assert_eq!(opts.interval_for_retry(0), 0.1);
        assert_eq!(opts.interval_for_retry(1), 0.0);
    }


    // Catches: a negative, NaN or infinite duration accepted (NaN slipped past `< 0` checks and
    // infinity panicked in `Duration::from_secs_f64`), zero refused, or an error that does not
    // name its field.
    #[test]
    fn invalid_durations_fail_with_their_field_and_zero_or_none_is_valid() {
        assert!(ApiConnectOptions::new(0, 0.0, 0.0).is_ok());
        let ok = ApiConnectOptions { read_timeout: None, total_timeout: Some(0.0), ..DEFAULT_API_CONNECT_OPTIONS };
        assert_eq!(ok.validate(), Ok(()));
        for bad in [-1.0, f64::NAN, f64::INFINITY, 1e300] {
            let err = ApiConnectOptions { read_timeout: Some(bad), ..DEFAULT_API_CONNECT_OPTIONS }.validate();
            assert_eq!(err.unwrap_err().field, "read_timeout", "{bad}");
            assert_eq!(ApiConnectOptions::new(3, bad, 10.0).unwrap_err().field, "retry_interval", "{bad}");
        }
        let err = api_connect_options_check(-1, 2.0, 10.0, None, None, None, None, None).unwrap_err();
        assert_eq!(err.to_string(), "conn_options.max_retry must be an integer >= 0");
    }
}

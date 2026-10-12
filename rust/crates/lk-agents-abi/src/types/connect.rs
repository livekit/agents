//! `APIConnectOptions` (`types.py:113`). The Python type stays a frozen dataclass (the fallback
//! adapters `dataclasses.replace` it).

use lk_agents_host_derive::Host;
use thiserror::Error;

/// `types.py:127`, minus the guard `u32` makes unnecessary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ApiConnectOptionsError {
    #[error("retry_interval must be greater than or equal to 0")]
    NegativeRetryInterval,
    #[error("timeout must be greater than or equal to 0")]
    NegativeTimeout,
}

/// Per-call connection policy: how many retries, how long between them, how long to wait.
///
/// The dict for `APIConnectOptions(..)` is generated from these field names: a misspelt keyword
/// would silently take the reference's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Host)]
#[binding(emit)]
pub struct ApiConnectOptions {
    /// Maximum number of retries to connect to the API.
    pub max_retry: u32,
    /// Interval between retries to connect to the API in seconds.
    pub retry_interval: f64,
    /// Timeout for connecting to the API in seconds.
    pub timeout: f64,
}

/// `types.py:149`.
pub const DEFAULT_API_CONNECT_OPTIONS: ApiConnectOptions =
    ApiConnectOptions { max_retry: 3, retry_interval: 2.0, timeout: 10.0 };

impl Default for ApiConnectOptions {
    fn default() -> Self {
        DEFAULT_API_CONNECT_OPTIONS
    }
}

impl ApiConnectOptions {
    /// `max_retry < 0` is unrepresentable. NaN is accepted, as in Python.
    pub fn new(
        max_retry: u32,
        retry_interval: f64,
        timeout: f64,
    ) -> Result<Self, ApiConnectOptionsError> {
        if retry_interval < 0.0 {
            return Err(ApiConnectOptionsError::NegativeRetryInterval);
        }
        if timeout < 0.0 {
            return Err(ApiConnectOptionsError::NegativeTimeout);
        }
        Ok(ApiConnectOptions { max_retry, retry_interval, timeout })
    }

    /// `types.py:138`. The first retry waits 100 ms, not zero as the docstring says (#1337). Shared
    /// so native and host retry loops agree.
    pub fn interval_for_retry(&self, num_retries: u32) -> f64 {
        if num_retries == 0 { FIRST_RETRY_INTERVAL } else { self.retry_interval }
    }
}

pub const FIRST_RETRY_INTERVAL: f64 = 0.1;

/// `APIConnectOptions._interval_for_retry` (`types.py:138`) for the host's retry loops; only
/// `retry_interval` affects the answer.
#[crate::host(export(pid = false))]
pub fn api_retry_interval(retry_interval: f64, num_retries: u32) -> f64 {
    ApiConnectOptions { max_retry: 0, retry_interval, timeout: 0.0 }.interval_for_retry(num_retries)
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


    // Catches: dropping the guards, or checking `<= 0` instead of `< 0` (zero is legal and used).
    #[test]
    fn zero_is_valid_and_negatives_are_not() {
        assert!(ApiConnectOptions::new(0, 0.0, 0.0).is_ok());
        assert_eq!(
            ApiConnectOptions::new(3, -1.0, 10.0),
            Err(ApiConnectOptionsError::NegativeRetryInterval)
        );
        assert_eq!(
            ApiConnectOptions::new(3, 2.0, -1.0),
            Err(ApiConnectOptionsError::NegativeTimeout)
        );
    }
}

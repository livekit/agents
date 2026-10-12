//! The chain's calculators: pure functions that differ per modality.

use std::sync::Arc;

use lk_agents_abi::ApiConnectOptions;

/// Views built up front by the host (only while its loop runs), so the chain can bound a wrapped
/// provider's own retry loop; attempts and probes run concurrently with different budgets.
/// `None` (a native provider) serves the provider itself.
pub fn bound_views<T: ?Sized>(
    provider: &Arc<T>,
    policy: &ChainPolicy,
    bind: impl Fn(&ApiConnectOptions) -> Option<Arc<T>>,
) -> (Arc<T>, Arc<T>) {
    let caller = caller_default();
    let view = |conn: ApiConnectOptions| bind(&conn).unwrap_or_else(|| Arc::clone(provider));
    (view(conn_options_for_attempt(&caller, policy)), view(conn_options_for_recovery(&caller, policy)))
}

/// What the chain does when a backend fails after it already produced output:
/// LLM default re-raises ([`AfterOutput::Abort`], `llm/fallback_adapter.py:300-306`); LLM with
/// `retry_on_chunk_sent` and STT try the next entry ([`AfterOutput::Retry`]); TTS returns success
/// with truncated audio ([`AfterOutput::Stop`], `tts/fallback_adapter.py:293-297,479-483`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterOutput {
    Retry,
    Abort,
    Stop,
}

/// The `conn_options` fields the chain rewrites before handing a request to a child, plus
/// `retry_on_chunk_sent`. `max_retry` defaults per modality (LLM 0, STT 1, TTS 2); the hard 0s
/// are [`caller_default`] and [`conn_options_for_recovery`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainPolicy {
    pub max_retry_per_backend: u32,
    /// Replaces `conn_options.timeout`. `None` keeps the caller's, as TTS does
    /// (`tts/fallback_adapter.py:193`).
    pub attempt_timeout: Option<f64>,
    /// Replaces `conn_options.retry_interval`; `None` keeps the caller's.
    pub retry_interval: Option<f64>,
    pub after_output: AfterOutput,
}

impl ChainPolicy {
    /// `llm/fallback_adapter.py:45-49`: `attempt_timeout=5.0`, `max_retry_per_llm=0`,
    /// `retry_interval=0.5`, `retry_on_chunk_sent=False`.
    pub fn llm_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 0,
            attempt_timeout: Some(5.0),
            retry_interval: Some(0.5),
            after_output: AfterOutput::Abort,
        }
    }

    /// `stt/fallback_adapter.py:57-59`: `attempt_timeout=10.0`, `max_retry_per_stt=1`,
    /// `retry_interval=5`. No progress guard exists on either STT path.
    pub fn stt_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 1,
            attempt_timeout: Some(10.0),
            retry_interval: Some(5.0),
            after_output: AfterOutput::Retry,
        }
    }

    /// `tts/fallback_adapter.py:64`: `max_retry_per_tts=2` and nothing else -- the caller's
    /// timeout and retry interval are passed through untouched.
    pub fn tts_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 2,
            attempt_timeout: None,
            retry_interval: None,
            after_output: AfterOutput::Stop,
        }
    }

    /// LLM only.
    pub fn with_retry_on_chunk_sent(mut self, retry: bool) -> Self {
        self.after_output = if retry { AfterOutput::Retry } else { AfterOutput::Abort };
        self
    }
}

/// `DEFAULT_FALLBACK_API_CONNECT_OPTIONS` (`llm:17`, `stt:27`, `tts:31`): the adapter is the retry
/// loop. `retry_interval` drops to the dataclass default (2.0), as in the reference.
pub fn caller_default() -> ApiConnectOptions {
    ApiConnectOptions {
        max_retry: 0,
        retry_interval: ApiConnectOptions::default().retry_interval,
        timeout: ApiConnectOptions::default().timeout,
    }
}

/// What a child sees on a real attempt: `dataclasses.replace(conn_options, max_retry=…,
/// timeout=…, retry_interval=…)`, unconditionally (the N x M guard).
pub fn conn_options_for_attempt(
    caller: &ApiConnectOptions,
    policy: &ChainPolicy,
) -> ApiConnectOptions {
    ApiConnectOptions {
        max_retry: policy.max_retry_per_backend,
        timeout: policy.attempt_timeout.unwrap_or(caller.timeout),
        retry_interval: policy.retry_interval.unwrap_or(caller.retry_interval),
    }
}

/// What a child sees on a background recovery probe: `max_retry` is a hard 0 (`stt:445-449`,
/// `tts:524-529`).
pub fn conn_options_for_recovery(
    caller: &ApiConnectOptions,
    policy: &ChainPolicy,
) -> ApiConnectOptions {
    ApiConnectOptions {
        max_retry: 0,
        timeout: policy.attempt_timeout.unwrap_or(caller.timeout),
        retry_interval: policy.retry_interval.unwrap_or(caller.retry_interval),
    }
}

/// `tts/fallback_adapter.py:86-87,109-118`: the adapter's output rate (`requested`, else the
/// maximum of the entries) and, per entry, whether it needs resampling.
pub fn resampling_plan(entry_rates: &[u32], requested: Option<u32>) -> (u32, Vec<bool>) {
    let target = requested.unwrap_or_else(|| entry_rates.iter().copied().max().unwrap_or(0));
    (target, entry_rates.iter().map(|&r| r != target).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: choosing the minimum rate, or resampling an entry that already matches.
    #[test]
    fn the_target_rate_is_the_best_entry_and_only_the_others_resample() {
        assert_eq!(
            resampling_plan(&[24_000, 16_000, 24_000], None),
            (24_000, vec![false, true, false])
        );
        assert_eq!(resampling_plan(&[24_000, 16_000], Some(48_000)), (48_000, vec![true, true]));
        assert_eq!(resampling_plan(&[22_050], None), (22_050, vec![false]));
    }

    // Catches: `caller_default` carrying the workspace `max_retry=3`.
    #[test]
    fn the_adapters_own_default_conn_options_do_not_retry() {
        assert_eq!(caller_default().max_retry, 0);
        assert_eq!(caller_default().timeout, ApiConnectOptions::default().timeout);
    }
}

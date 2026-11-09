mod chain;
mod channel;
mod llm;
pub mod machine;
pub mod realtime;
mod stt;
mod traced;
mod tts;

pub use chain::{
    Attempt, AvailabilityChanged, Backend, ChainObserver, FallbackChain, IgnoreEvents, RecoveryMode,
};
use lk_agents_abi::ApiConnectOptions;
pub use llm::{FallbackLlm, LlmBackend};
pub use machine::{ChainPolicy, FallbackPolicy};
pub use stt::{
    FallbackStt, InputTee, SttBackend, SttReq, SttSession,
    merge_capabilities as merge_stt_capabilities,
};
pub use tts::{
    FallbackTts, Token, TokenLog, TtsBackend, TtsReq, merge_capabilities as merge_tts_capabilities,
};

/// `DEFAULT_FALLBACK_API_CONNECT_OPTIONS` (`llm:17`, `stt:27`, `tts:31`): the adapter is the retry
/// loop. `retry_interval` drops to the dataclass default (2.0), as in the reference.
pub fn caller_default() -> ApiConnectOptions {
    ApiConnectOptions { max_retry: 0, ..ApiConnectOptions::default() }
}

/// What a child sees on a real attempt: `dataclasses.replace(conn_options, max_retry=0,
/// timeout=attempt_timeout)`. The reference's `timeout` bounded each limit, so the attempt
/// timeout caps every one; the caller's total stays the operation's total, or a
/// listening STT stream would be cut once it outlived one attempt's timeout.
pub fn conn_options_for_attempt(
    caller: &ApiConnectOptions,
    policy: &ChainPolicy,
) -> ApiConnectOptions {
    let capped = |limit: Option<f64>| match (policy.attempt_timeout, limit) {
        (Some(attempt), Some(limit)) => Some(attempt.min(limit)),
        (attempt, limit) => attempt.or(limit),
    };
    ApiConnectOptions {
        max_retry: 0,
        pool_timeout: capped(caller.pool_timeout),
        connect_timeout: capped(caller.connect_timeout),
        write_timeout: capped(caller.write_timeout),
        read_timeout: capped(caller.read_timeout),
        retry_interval: policy.retry_interval.unwrap_or(caller.retry_interval),
        ..*caller
    }
}

/// `tts/fallback_adapter.py:86-87,109-118`: the adapter's output rate (`requested`, else the
/// maximum of the entries) and, per entry, whether it needs resampling.
pub fn resampling_plan(entry_rates: &[u32], requested: Option<u32>) -> (u32, Vec<bool>) {
    let target = requested.unwrap_or_else(|| entry_rates.iter().copied().max().unwrap_or(0));
    (target, entry_rates.iter().map(|&r| r != target).collect())
}

/// A backend's identity: its host object's, else the retained native allocation (alive for the
/// chain's lifetime, shared by clones), so rebinding options never adds a backend.
pub(crate) fn backend_resource<T: ?Sized>(
    provider: &T,
    host_instance: Option<usize>,
) -> lk_agents_abi::resources::ResourceId {
    let instance =
        host_instance.unwrap_or_else(|| std::ptr::from_ref(provider).cast::<()>() as usize);
    lk_agents_abi::resources::ResourceId(instance as u64)
}

#[cfg(test)]
mod resampling_tests {
    use super::resampling_plan;

    #[test]
    fn the_target_rate_is_the_best_entry_and_only_the_others_resample() {
        assert_eq!(
            resampling_plan(&[24_000, 16_000, 24_000], None),
            (24_000, vec![false, true, false])
        );
        assert_eq!(resampling_plan(&[24_000, 16_000], Some(48_000)), (48_000, vec![true, true]));
        assert_eq!(resampling_plan(&[22_050], None), (22_050, vec![false]));
    }
}

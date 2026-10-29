//! Provider fallback: one generic [`FallbackChain`] for LLM/STT/TTS, and the realtime adapter's
//! sessions, driven by [`machine`]. A fallback adapter implements the same traits as its entries.
//! Realtime differs: failure is a pushed event, recovery is a cooldown, a healthy primary is not
//! preempted, and availability is per session. The `metrics_collected` tee lives in
//! `lk-agents-py`, since these traits carry no event emitter.

mod chain;
mod channel;
mod llm;
pub mod machine;
pub mod realtime;
mod stt;
mod tts;

pub use chain::{
    Attempt, AvailabilityChanged, Backend, ChainObserver, ChainOutcome, FallbackChain,
    IgnoreEvents, NoBackends, Recovery, RecoverySlot,
};
use lk_agents_abi::ApiConnectOptions;
pub use llm::{FallbackLlm, LlmBackend};
pub use machine::{AvailabilityChange, ChainPolicy, ErrorPlan, FallbackPolicy};
pub use realtime::{CapabilityMismatch, merge_capabilities as merge_realtime_capabilities};
pub use stt::{
    FallbackStt, InputTee, SttBackend, SttChainError, SttReq, SttSession,
    merge_capabilities as merge_stt_capabilities,
};
pub use tts::{
    FallbackTts, Token, TokenLog, TtsBackend, TtsChainError, TtsReq,
    merge_capabilities as merge_tts_capabilities,
};

#[cfg(test)]
mod adapter_tests;
#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

/// `DEFAULT_FALLBACK_API_CONNECT_OPTIONS` (`llm:17`, `stt:27`, `tts:31`): the adapter is the retry
/// loop. `retry_interval` drops to the dataclass default (2.0), as in the reference.
pub fn caller_default() -> ApiConnectOptions {
    ApiConnectOptions { max_retry: 0, ..ApiConnectOptions::default() }
}

/// What a child sees on a real attempt: `dataclasses.replace(conn_options, max_retry=0,
/// timeout=attempt_timeout)`. The reference's `timeout` bounded each phase, so the attempt
/// timeout caps every phase limit; the caller's total stays the operation's total, or a
/// listening STT stream would be cut once it outlived one attempt's timeout.
pub fn conn_options_for_attempt(
    caller: &ApiConnectOptions,
    policy: &ChainPolicy,
) -> ApiConnectOptions {
    let phase = |limit: Option<f64>| match (policy.attempt_timeout, limit) {
        (Some(attempt), Some(limit)) => Some(attempt.min(limit)),
        (attempt, limit) => attempt.or(limit),
    };
    ApiConnectOptions {
        max_retry: 0,
        pool_timeout: phase(caller.pool_timeout),
        connect_timeout: phase(caller.connect_timeout),
        write_timeout: phase(caller.write_timeout),
        read_timeout: phase(caller.read_timeout),
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

//! The supplied-time request lifecycle used by host-authored providers.

use crate::SpeechEvent;
use crate::errors::{ApiError, ProviderError};
use crate::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use crate::types::retry::{RequestKind, RetryDecision};

/// What a host synthesis's response did, for its read deadline.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown synthesis observation {s:?}")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthesisObservation {
    /// Response data: the deadline restarts.
    Progress,
    /// Audio settles a protocol's first response; after the final input it is progress.
    Audio,
    /// No response is outstanding: the deadline disarms.
    Complete,
}

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[crate::host(class(name = "_core_ProviderAttempts"))]
pub trait ProviderAttempts: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        kind: RequestKind,
        conn_options: ApiConnectOptions,
        now_ns: u64,
    ) -> Result<Self, ApiConnectOptionsError>;
    #[host(call)]
    fn external_retry_owner(&mut self);
    #[host(call)]
    fn ready(&mut self, now_ns: u64) -> Result<bool, ApiError>;
    #[host(call)]
    fn failure(
        &mut self,
        error: ProviderError,
        produced_output: bool,
        now_ns: u64,
    ) -> Result<RetryDecision, ApiError>;
    #[host(call)]
    fn wake_at(&self) -> Option<u64>;
    #[host(call)]
    fn capacity_exceeded(&mut self, input: bool, now_ns: u64) -> Result<(), ApiError>;
    #[host(call)]
    fn expect_response(&mut self, now_ns: u64);
    #[host(call)]
    fn input_ended(&mut self, now_ns: u64);
    #[host(call)]
    fn observe(&mut self, event: SpeechEvent, now_ns: u64);
    #[host(call)]
    fn observe_synthesis(&mut self, observation: SynthesisObservation, now_ns: u64);
    #[host(call)]
    fn succeeded(&mut self, now_ns: u64) -> Result<(), ApiError>;
    #[host(getter)]
    fn cleanup_timeout(&self) -> f64;
    #[host(call)]
    fn cancel(&mut self, now_ns: u64);
}

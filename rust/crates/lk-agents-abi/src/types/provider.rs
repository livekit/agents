//! The supplied-time request lifecycle used by host-authored providers.

use crate::SpeechEvent;
use crate::errors::ApiError;
use crate::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use crate::types::retry::RequestKind;

/// Host callbacks execute attempts authorized by the shared inference request machine.
#[crate::host(class(name = "ProviderAttempts"))]
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
        retryable: bool,
        status: Option<i64>,
        produced_output: bool,
        now_ns: u64,
    ) -> Result<(String, f64, Option<String>), ApiError>;
    #[host(call)]
    fn wake_at(&self) -> Option<u64>;
    #[host(call)]
    fn expect_response(&mut self, now_ns: u64);
    #[host(call)]
    fn input_ended(&mut self, now_ns: u64);
    #[host(call)]
    fn observe(&mut self, event: SpeechEvent, now_ns: u64);
    #[host(call)]
    fn succeeded(&mut self, now_ns: u64) -> Result<(), ApiError>;
    #[host(getter)]
    fn cleanup_timeout(&self) -> f64;
    #[host(call)]
    fn cancel(&mut self, now_ns: u64);
}

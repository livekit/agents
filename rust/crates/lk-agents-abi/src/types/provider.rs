//! The supplied-time request lifecycle used by host-authored providers.

use std::sync::Arc;

use crate::errors::HostError;
use crate::errors::ProviderError;
use crate::invoke::{CallContext, Cancellation};
use crate::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use crate::types::retry::RequestKind;
use crate::{HostRef, SpeechEvent};

/// What a host provider's response did, for its read deadline.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown response observation {s:?}")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseObservation {
    /// Response data: the deadline restarts.
    Progress,
    /// Audio settles a protocol's first response; after the final input it is progress.
    Audio,
    /// No response is outstanding: the deadline disarms.
    Complete,
}

/// One host provider request, its attempts run by the native executor.
///
/// The host implements one attempt; the executor decides when an attempt may start, runs it,
/// cancels it when a deadline passes, and retries it under `conn_options`
/// (docs/models.md "Provider attempts").
#[crate::host(proxy(host = "livekit.agents._bindings.AttemptHost"))]
pub trait AttemptHost: Send + Sync {
    /// Run one attempt to completion: the provider's `_run`. Its value is the request's result
    /// (`recognize` returns its event; streams return `None`). `retry`: an earlier attempt
    /// failed, so a stream that replays its input rewinds it first. Cancelled when a deadline
    /// passes or the request is closed; the executor waits for its cleanup to settle.
    #[host(await, name = "_core_attempt")]
    fn attempt<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a CallContext,
        #[host(cancel)] cancel: Arc<dyn Cancellation>,
        retry: bool,
    ) -> crate::BoxFuture<'a, Result<Option<HostRef>, ProviderError>>;
    /// Whether output already reached the caller: a request that delivered output is not
    /// replayed (except a recognition, whose transcripts are idempotent per segment).
    #[host(call, name = "_core_produced")]
    fn produced(&self) -> Result<bool, HostError>;
    /// A failed attempt that will be retried, as the provider's recoverable `error` event. The
    /// host reports the error that ends the request itself, unless the session owns retries.
    #[host(notify, name = "_core_retrying")]
    fn retrying(&self, error: ProviderError);
}

/// The attempts of one host provider request (`livekit.agents.ProviderAttempts`).
///
/// `run()` drives them: it waits until an attempt may start, runs the host's attempt, races it
/// against the read and total deadlines, and on failure retries, raises or stops as the shared
/// attempt policy decides. Inside a session the session owns retries (`core = True`): `run()`
/// then runs exactly one attempt, still bounded by its deadlines.
///
/// The observation methods feed the read deadline while an attempt runs: a provider calls
/// `expect_response()` when it sent something that owes an answer, `progress()` when response
/// data arrived (heartbeats are not progress), `audio_received()` for synthesized audio and
/// `complete_response()` when nothing is owed any more.
#[crate::host(class(name = "ProviderAttempts"))]
pub trait ProviderAttempts: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        kind: RequestKind,
        conn_options: ApiConnectOptions,
        host: Arc<dyn AttemptHost>,
        #[host(context)] context: CallContext,
    ) -> Result<Self, ApiConnectOptionsError>;
    /// Whether the session owns retries; set before the request starts.
    #[host(getter)]
    fn core(&self) -> bool;
    #[host(setter, name = "core")]
    fn set_core(&self, core: bool);
    /// Run the request's attempts. Returns the successful attempt's value (`None` when the
    /// provider stopped the request, a 499); raises the attempt's own error when it is not
    /// retried, or `APIConnectionError` from the last error once the retries are exhausted.
    /// Each attempt runs under the caller's current span.
    #[host(await)]
    async fn run(&self, cx: crate::telemetry::Cx) -> Result<Option<HostRef>, ProviderError>;
    /// A request was sent that owes a response; `final`: it was the last input.
    #[host(call)]
    fn expect_response(&self, #[host(kw, default = false)] r#final: bool);
    /// The input ended: its final response is owed.
    #[host(call)]
    fn input_ended(&self);
    /// Response data arrived: the read deadline restarts.
    #[host(call)]
    fn progress(&self);
    /// Synthesized audio arrived: it settles the first response, and after the final input it
    /// is progress.
    #[host(call)]
    fn audio_received(&self);
    /// No response is outstanding: the read deadline disarms.
    #[host(call)]
    fn complete_response(&self);
    /// A recognition event, for the stream's read deadline and transcript completion.
    #[host(call)]
    fn observe(&self, event: SpeechEvent);
    /// Whether retries were exhausted: `run()` then raised `APIConnectionError` from the last
    /// attempt's error, which the request's ending `error` event carries.
    #[host(getter)]
    fn exhausted(&self) -> bool;
    /// After `run()` was cancelled and the host cancelled its running attempt: wait until that
    /// attempt finished its cleanup, at most `cleanup_timeout` (a stalled cleanup is logged and
    /// keeps running). `_bindings.run_attempts` does both.
    #[host(await)]
    async fn settled(&self);
    /// How long a cancelled attempt's cleanup may take before it is reported as stalled.
    #[host(getter)]
    fn cleanup_timeout(&self) -> f64;
}

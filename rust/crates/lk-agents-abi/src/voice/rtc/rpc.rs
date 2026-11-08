//! RPC as the job's local participant performs and handles it: the SDK's call records, the
//! `next` continuations of an `rtc.RpcInterceptor`, and the native interceptor that traces them.

use crate::errors::HostError;
use crate::worker::ipc_host::JobExecution;
use std::sync::Arc;

/// `rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD`: no handler is registered for the method.
pub const RPC_UNSUPPORTED_METHOD: u32 = 1400;

/// What a failed RPC answered: the callee's or the SDK's `rtc.RpcError` code and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: u32,
    pub message: String,
}

/// `rtc.RpcCallInfo`: an outgoing call, as `perform_rpc` takes it.
#[crate::host(record(
    module = "rtc_rpc",
    name = "RpcCallInfo",
    host_type = "livekit.rtc.RpcCallInfo",
    boundary,
    extract,
    emit
))]
#[derive(Debug, Clone, PartialEq)]
pub struct RpcCallInfo {
    pub destination_identity: String,
    pub method: String,
    pub payload: String,
    /// Seconds to wait for the response; `None` for the SDK's default.
    pub response_timeout: Option<f64>,
    pub max_round_trip_latency: Option<f64>,
}

/// `rtc.RpcInvocationData`: an incoming call, as its handler receives it.
#[crate::host(record(
    module = "rtc_rpc",
    name = "RpcInvocationData",
    host_type = "livekit.rtc.RpcInvocationData",
    boundary,
    extract,
    emit
))]
#[derive(Debug, Clone, PartialEq)]
pub struct RpcInvocationData {
    pub request_id: String,
    pub caller_identity: String,
    pub payload: String,
    /// Seconds the caller waits for the response.
    pub response_timeout: f64,
    pub method: String,
    /// The SDK's `rtc.RpcError.ErrorCode` once it cancelled the handler chain.
    pub cancel_reason: Option<u32>,
}

/// `next` of `RpcInterceptor.intercept_outgoing`: the rest of the chain, which performs the call.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait OutgoingRpc: Send + Sync {
    #[host(await, name = "__call__")]
    fn call(
        &self,
        cx: &crate::telemetry::Cx,
        call: RpcCallInfo,
    ) -> crate::BoxFuture<'_, Result<String, HostError>>;
}

/// `next` of `RpcInterceptor.intercept_incoming`: the rest of the chain, which runs the handler.
#[crate::host(proxy(host = "collections.abc.Callable"))]
pub trait IncomingRpc: Send + Sync {
    #[host(await, name = "__call__")]
    fn call(
        &self,
        cx: &crate::telemetry::Cx,
        invocation: RpcInvocationData,
    ) -> crate::BoxFuture<'_, Result<Option<String>, HostError>>;
}

/// The job's `rtc.RpcInterceptor`, added to its local participant once the room connected: an
/// `rpc_call` span around each call the participant performs, under the caller's span, and an
/// `rpc_handler` span around each call it handles, under the job's primary session. What the
/// chain returns or raises reaches the caller unchanged.
#[crate::host(class(name = "_core_RpcTracing", implements = "livekit.rtc.RpcInterceptor"))]
pub trait RpcTracing: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(job: Arc<dyn JobExecution>) -> Self;
    #[host(await)]
    async fn intercept_outgoing(
        &self,
        call: RpcCallInfo,
        next: Arc<dyn OutgoingRpc>,
        cx: crate::telemetry::Cx,
    ) -> Result<String, HostError>;
    #[host(await)]
    async fn intercept_incoming(
        &self,
        invocation: RpcInvocationData,
        next: Arc<dyn IncomingRpc>,
        cx: crate::telemetry::Cx,
    ) -> Result<Option<String>, HostError>;
}

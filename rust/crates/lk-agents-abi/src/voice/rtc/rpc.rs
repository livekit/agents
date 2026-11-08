//! RPC as the job's local participant performs and handles it: the SDK's call records, the
//! `next` continuations of an `rtc.RpcInterceptor`, and the native interceptor that traces them.

use crate::errors::HostError;
use crate::worker::ipc_host::JobExecution;
use std::any::Any;
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

/// `rtc.RpcInvocationData`: an incoming call, as its handler receives it. Read as the SDK's own
/// object, which the SDK marks with its cancel reason before it cancels the handler chain.
#[crate::host(proxy(host = "livekit.rtc.RpcInvocationData"))]
pub trait RpcInvocation: Any + Send + Sync {
    #[host(read_once, getter = "request_id")]
    fn request_id(&self) -> String;
    #[host(read_once, getter = "caller_identity")]
    fn caller_identity(&self) -> String;
    #[host(read_once, getter = "payload")]
    fn payload(&self) -> String;
    /// Seconds the caller waits for the response.
    #[host(read_once, getter = "response_timeout")]
    fn response_timeout(&self) -> f64;
    #[host(read_once, getter = "method")]
    fn method(&self) -> String;
    /// The SDK's `rtc.RpcError.ErrorCode` once it cancelled the handler chain.
    #[host(await, getter = "cancel_reason")]
    fn cancel_reason(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<u32>, HostError>>;
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
        invocation: Arc<dyn RpcInvocation>,
    ) -> crate::BoxFuture<'_, Result<Option<String>, HostError>>;
}

/// The job's `rtc.RpcInterceptor`, added to its local participant once the room connected: an
/// `rpc_call` span around each call the participant performs, under the caller's span, and an
/// `rpc_handler` span around each call it handles, under the job's primary session, else the job,
/// with the SDK's cancel reason when it cancelled the handler. What the chain returns or raises
/// reaches the caller unchanged.
#[crate::host(class(name = "_core_RpcTracing", implements = "livekit.rtc.RpcInterceptor"))]
pub trait RpcTracing: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(new)]
    fn new(env: &Self::Env, job: Arc<dyn JobExecution>) -> Self;
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
        invocation: Arc<dyn RpcInvocation>,
        next: Arc<dyn IncomingRpc>,
        cx: crate::telemetry::Cx,
    ) -> Result<Option<String>, HostError>;
}

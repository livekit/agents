//! The job's RPC tracing (`_core_RpcTracing`): an `rtc.RpcInterceptor` recording the RPCs the
//! job's local participant performs and handles.

use crate::telemetry;
use lk_agents_abi::errors::{HostError, HostErrorKind};
use lk_agents_abi::voice::rtc::rpc::{
    self, IncomingRpc, OutgoingRpc, RPC_UNSUPPORTED_METHOD, RpcCallInfo, RpcInvocationData,
};
use lk_agents_abi::worker::ipc_host::JobExecution;
use std::sync::Arc;

/// The characters of a payload or response a span keeps.
const CONTENT_CHARS: usize = 1024;

pub struct RpcTracing {
    job: Arc<dyn JobExecution>,
}

impl rpc::RpcTracing for RpcTracing {
    fn new(job: Arc<dyn JobExecution>) -> Self {
        Self { job }
    }

    async fn intercept_outgoing(
        &self,
        call: RpcCallInfo,
        next: Arc<dyn OutgoingRpc>,
        cx: telemetry::Cx,
    ) -> Result<String, HostError> {
        let span = telemetry::start(
            &cx,
            telemetry::RpcCall {
                method: call.method.clone(),
                destination_identity: call.destination_identity.clone(),
                payload_size: call.payload.len() as u64,
                payload: content(&call.payload),
                response_timeout: call.response_timeout.map(telemetry::Seconds),
            },
        );
        let result = next.call(span.cx(), call).await;
        end(span, result.as_deref(), false);
        result
    }

    /// Recorded under the job's primary session; untraced before it started.
    async fn intercept_incoming(
        &self,
        invocation: RpcInvocationData,
        next: Arc<dyn IncomingRpc>,
        _cx: telemetry::Cx,
    ) -> Result<Option<String>, HostError> {
        let Some(session) = self.job.session_trace() else {
            return next.call(&telemetry::Cx::NONE, invocation).await;
        };
        let span = telemetry::start(
            &session,
            telemetry::RpcHandler {
                method: invocation.method.clone(),
                request_id: invocation.request_id.clone(),
                caller_identity: invocation.caller_identity.clone(),
                response_timeout: telemetry::Seconds(invocation.response_timeout),
                payload_size: invocation.payload.len() as u64,
                payload: content(&invocation.payload),
            },
        );
        let result = next.call(span.cx(), invocation).await;
        end(span, result.as_ref().map(|r| r.as_deref().unwrap_or_default()), true);
        result
    }
}

/// Ends `span` with the chain's outcome: the response, or the error with its RPC code. A
/// cancellation ends it cancelled. A `handler` span records whether the method was registered.
fn end<D>(span: telemetry::ScopedSpan<D>, result: Result<&str, &HostError>, handler: bool)
where
    D: telemetry::SpanDecl<End = telemetry::RpcEnd>,
    telemetry::RpcEnd: telemetry::SpanEnd<D>,
{
    let error = match result {
        Ok(response) => {
            let end = telemetry::RpcEnd {
                error_code: None,
                response_size: Some(response.len() as u64),
                response: content(response),
                handler_registered: handler.then_some(true),
            };
            return telemetry::end(span, end);
        }
        Err(error) if error.kind == HostErrorKind::Cancelled => return drop(span),
        Err(error) => error,
    };
    let code = match error.kind {
        HostErrorKind::Rpc { code } => Some(code),
        _ => None,
    };
    let end = telemetry::RpcEnd {
        error_code: code.map(i64::from),
        response_size: None,
        response: None,
        handler_registered: handler.then_some(code != Some(RPC_UNSUPPORTED_METHOD)),
    };
    telemetry::end(span, telemetry::Failed { end, error });
}

fn content(text: &str) -> Option<telemetry::Content<String>> {
    (!text.is_empty()).then(|| telemetry::Content(text.chars().take(CONTENT_CHARS).collect()))
}

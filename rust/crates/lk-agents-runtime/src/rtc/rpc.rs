use crate::telemetry;
use lk_agents_abi::errors::{HostError, HostErrorKind};
use lk_agents_abi::voice::rtc::rpc::{
    self, IncomingRpc, OutgoingRpc, RPC_UNSUPPORTED_METHOD, RpcCallInfo, RpcInvocation,
};
use lk_agents_abi::worker::ipc_host::JobExecution;
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_net::env::{ProcessEnv, ProcessError};
use std::sync::Arc;

/// The characters of a payload or response a span keeps.
const CONTENT_CHARS: usize = 1024;

type Spawn = fn() -> Result<Arc<dyn Spawner>, ProcessError>;

pub struct RpcTracing {
    job: Arc<dyn JobExecution>,
    /// Where a handler span dropped with its chain finishes.
    spawner: Spawn,
}

impl rpc::RpcTracing for RpcTracing {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, job: Arc<dyn JobExecution>) -> Self {
        Self { job, spawner: env.spawner }
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

    /// Recorded under the job's primary session, else the job's span.
    async fn intercept_incoming(
        &self,
        invocation: Arc<dyn RpcInvocation>,
        next: Arc<dyn IncomingRpc>,
        cx: telemetry::Cx,
    ) -> Result<Option<String>, HostError> {
        let Some(parent) = self.job.rpc_trace() else {
            return next.call(&cx, invocation).await;
        };
        let payload = invocation.payload();
        let span = telemetry::start(
            &parent,
            telemetry::RpcHandler {
                method: invocation.method(),
                request_id: invocation.request_id(),
                caller_identity: invocation.caller_identity(),
                response_timeout: telemetry::Seconds(invocation.response_timeout()),
                payload_size: payload.len() as u64,
                payload: content(&payload),
            },
        );
        let cx = span.cx().clone();
        let mut handling = Handling {
            span: Some(span),
            invocation: Arc::clone(&invocation),
            spawner: self.spawner,
        };
        let result = next.call(&cx, invocation).await;
        if let Some(span) = handling.span.take() {
            match &result {
                Err(error) if error.kind == HostErrorKind::Cancelled => {
                    cancelled(span, &*handling.invocation).await;
                }
                result => {
                    end(span, result.as_ref().map(|r| r.as_deref().unwrap_or_default()), true)
                }
            }
        }
        result
    }
}

/// An `rpc_handler` span until its chain settles. Dropped with the chain the SDK cancelled, it
/// finishes on its own task: see [`cancelled`].
struct Handling {
    span: Option<telemetry::ScopedSpan<telemetry::RpcHandler>>,
    invocation: Arc<dyn RpcInvocation>,
    spawner: Spawn,
}

impl Drop for Handling {
    fn drop(&mut self) {
        let Some(span) = self.span.take() else { return };
        match (self.spawner)() {
            Ok(spawner) => {
                let invocation = Arc::clone(&self.invocation);
                spawner.spawn_detached(async move { cancelled(span, &*invocation).await });
            }
            Err(error) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "rpc_handler span: {error}")
            }
        }
    }
}

/// Ends a cancelled handler's `span` with the code the SDK marked the invocation with before it
/// cancelled the chain (a response timeout, the caller gone), else as cancelled.
async fn cancelled(
    span: telemetry::ScopedSpan<telemetry::RpcHandler>,
    invocation: &dyn RpcInvocation,
) {
    let cx = span.cx().clone();
    match invocation.cancel_reason(&cx).await {
        Ok(Some(code)) => {
            let error = HostError {
                kind: HostErrorKind::Rpc { code },
                ..HostError::new("the handler was cancelled")
            };
            end(span, Err(&error), true);
        }
        Ok(None) => drop(span),
        Err(error) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "rpc_handler cancel reason: {error}");
            drop(span);
        }
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

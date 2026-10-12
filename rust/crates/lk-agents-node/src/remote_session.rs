//! `RemoteSession`'s client, running in Rust (`driver::ClientRuntime`) over a **JS transport**:
//! the Node twin of `lk-agents-py`'s `RemoteSessionClient`, and the first consumer of the
//! generated proxies (`generated/proxies.rs`).
//!
//! The transport, its receiver and the event sink are JS objects, each handed over as its
//! dispatcher (`crate::bridge`); [`JsTransport`] and [`JsSink`] make them the
//! `Arc<dyn SessionTransport>` / `Arc<dyn ClientSink>` the runtime takes. Written here until the
//! napi emitter writes bare proxies (Python's are generated from the same traits). Messages,
//! requests and responses cross as encoded protobuf, as they do in Python.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{RealClock, TokioSpawner};
use lk_agents_voice::remote_session::driver::{ClientRuntime, ReadyError, RequestError};
use lk_agents_voice::remote_session::proto;
use napi::Env;
use napi::bindgen_prelude::{Buffer, PromiseRaw};
use napi_derive::napi;
use prost::Message as _;

use crate::bridge::{DispatchFn, Ignored, JsArg, NodeBridge, host_err};
use crate::host_error::JsError;
use lk_agents_abi::errors::HostError;
use lk_agents_voice::remote_session::{ClientSink, SessionTransport};

/// A JS transport and its async iterator, as the runtime's [`SessionTransport`].
struct JsTransport {
    transport: NodeBridge,
    receiver: NodeBridge,
}

#[async_trait::async_trait]
impl SessionTransport for JsTransport {
    async fn start(&self) -> Result<(), HostError> {
        self.transport.call::<Ignored>("start", Vec::new()).await.map(|_| ()).map_err(host_err)
    }

    async fn send_message(&self, msg: &proto::AgentSessionMessage) -> Result<(), HostError> {
        let arg = JsArg::new(Buffer::from(msg.encode_to_vec()));
        self.transport.call::<Ignored>("send_message", vec![arg]).await.map(|_| ()).map_err(host_err)
    }

    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage) {
        let _ = self.transport.soon("send_message_threadsafe", vec![JsArg::new(Buffer::from(msg.encode_to_vec()))]);
    }

    async fn recv(&self) -> Result<Option<proto::AgentSessionMessage>, HostError> {
        let Some(data) = self.receiver.anext::<Buffer>().await.map_err(host_err::<HostError>)? else { return Ok(None) };
        proto::AgentSessionMessage::decode(&data[..]).map(Some).map_err(|e| HostError::new(format!("not an AgentSessionMessage: {e}")))
    }

    async fn close(&self) {
        let _ = self.transport.call::<Ignored>("close", Vec::new()).await;
    }
}

/// A JS event sink, as the runtime's [`ClientSink`]: `emit(name, event)`.
struct JsSink(NodeBridge);

impl ClientSink for JsSink {
    fn event(&self, name: &str, event: &proto::AgentSessionEvent) {
        let args = vec![JsArg::new(name.to_owned()), JsArg::new(Buffer::from(event.encode_to_vec()))];
        let _ = self.0.soon("emit", args);
    }
}

fn transport_err(e: lk_agents_abi::errors::HostError) -> JsError {
    JsError::new("RuntimeError", e.message)
}

fn duration(secs: f64, label: &str) -> Result<Duration, JsError> {
    Duration::try_from_secs_f64(secs).map_err(|e| JsError::new("ValueError", format!("{label}: {e}")))
}

#[napi(js_name = "RemoteSessionClient")]
pub struct RemoteSessionClient {
    core: Arc<ClientRuntime>,
}

#[napi]
impl RemoteSessionClient {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let spawner = napi::bindgen_prelude::within_runtime_if_available(TokioSpawner::current);
        RemoteSessionClient { core: Arc::new(ClientRuntime::new(Arc::new(RealClock::new()), Arc::new(spawner))) }
    }

    #[napi(getter)]
    pub fn started(&self) -> bool {
        self.core.started()
    }

    /// Start the transport and the receive loop; arriving events go to `sink`'s `emit(name, event)`.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn start<'env>(
        &self,
        env: &'env Env,
        #[napi(ts_arg_type = "(op: 'soon' | 'call' | 'anext', name: string, ...args: any[]) => unknown")] transport: DispatchFn,
        #[napi(ts_arg_type = "(op: 'soon' | 'call' | 'anext', name: string, ...args: any[]) => unknown")] receiver: DispatchFn,
        #[napi(ts_arg_type = "(op: 'soon' | 'call' | 'anext', name: string, ...args: any[]) => unknown")] sink: DispatchFn,
    ) -> napi::Result<PromiseRaw<'env, ()>> {
        let core = Arc::clone(&self.core);
        let transport = Arc::new(JsTransport { transport: NodeBridge::new(transport), receiver: NodeBridge::new(receiver) });
        let sink = Arc::new(JsSink(NodeBridge::new(sink)));
        env.spawn_future_with_callback(
            async move { Ok(core.start(transport, sink).await.map_err(transport_err)) },
            |env, out| out.map_err(|e| e.into_napi(env)),
        )
    }

    /// Send one encoded `SessionRequest` (its `request_id` is minted here) and resolve with the
    /// encoded `SessionResponse`. Rejects with `TimeoutError`, `RuntimeError` (a response carrying
    /// an error, a transport failure, not started) or `CancelledError` (the client closed first) --
    /// Python's kinds.
    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn request<'env>(&self, env: &'env Env, request: Buffer, timeout: f64) -> napi::Result<PromiseRaw<'env, Buffer>> {
        let (request, timeout) = crate::host_error::run(env, || {
            let request = proto::SessionRequest::decode(&request[..])
                .map_err(|e| JsError::new("ValueError", format!("not a SessionRequest: {e}")))?
                .request
                .ok_or_else(|| JsError::new("ValueError", "the SessionRequest sets no request"))?;
            Ok((request, duration(timeout, "timeout")?))
        })?;
        let core = Arc::clone(&self.core);
        env.spawn_future_with_callback(
            async move {
                Ok(core.request(request, timeout).await.map_err(|e| match e {
                    RequestError::Transport(e) => transport_err(e),
                    RequestError::Timeout => JsError::new("TimeoutError", "request timed out"),
                    RequestError::Failed(m) => JsError::new("RuntimeError", m),
                    RequestError::Cancelled => JsError::new("CancelledError", "the remote session closed"),
                    RequestError::NotStarted => JsError::new("RuntimeError", "remote session is not started"),
                    RequestError::Malformed(m) => JsError::new("ValueError", m),
                }))
            },
            |env, out| match out {
                Ok(resp) => Ok(Buffer::from(resp.encode_to_vec())),
                Err(e) => Err(e.into_napi(env)),
            },
        )
    }

    /// Ping until answered; rejects with the latched transport error in preference to
    /// `TimeoutError("wait_for_ready timed out")`.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn wait_for_ready<'env>(&self, env: &'env Env, timeout: f64, retry_interval: f64) -> napi::Result<PromiseRaw<'env, ()>> {
        let (timeout, retry) =
            crate::host_error::run(env, || Ok((duration(timeout, "timeout")?, duration(retry_interval, "retry_interval")?)))?;
        let core = Arc::clone(&self.core);
        env.spawn_future_with_callback(
            async move {
                Ok(core.wait_for_ready(timeout, retry).await.map_err(|e| match e {
                    ReadyError::Timeout => JsError::new("TimeoutError", "wait_for_ready timed out"),
                    ReadyError::Transport(e) => transport_err(e),
                    ReadyError::Cancelled => JsError::new("CancelledError", "the remote session closed"),
                }))
            },
            |env, out| out.map_err(|e| e.into_napi(env)),
        )
    }

    #[napi]
    pub async fn aclose(&self) -> napi::Result<()> {
        self.core.aclose().await;
        Ok(())
    }
}

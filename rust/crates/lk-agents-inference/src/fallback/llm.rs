use lk_agents_abi::llm::fallback::{HostChatContext, HostTools};
use lk_agents_abi::{ChatContext, Extra};
use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::LlmChunk;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_abi::{ChatRequest, Closed, Llm, LlmStream, StreamOutput, ToolChoice, Ts};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::fallback::ChainPolicy;
use crate::fallback::chain::{
    Attempt, Backend, ChainCall, ChainObserver, FallbackChain, RecoveryMode,
};
use crate::fallback::channel::{ServeAttempt, drain_probe, forward};
use crate::fallback::traced::{self, Traced};
use crate::telemetry;
use lk_agents_net::env::ProcessEnv;

// `chat()` returns at once and the chain runs as a task feeding an `mpsc`, so its failure
// surfaces on the first `next()`.

pub struct LlmBackend {
    inner: Arc<dyn Llm>,
    clock: Arc<dyn Clock>,
}

impl LlmBackend {
    /// The provider this entry wraps, so `FallbackLlm::model()` can answer from the serving entry.
    pub fn inner(&self) -> &Arc<dyn Llm> {
        &self.inner
    }

    pub fn new(inner: Arc<dyn Llm>, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(LlmBackend { inner, clock })
    }

    /// One request of this entry: an `llm_request` under the adapter's span, a recovery probe's
    /// marked as such.
    async fn request(
        &self,
        req: &LlmAttempt,
        conn: &lk_agents_abi::ApiConnectOptions,
        recovery: bool,
    ) -> Result<LlmStream, ProviderError> {
        // `dataclasses.replace(self._conn_options, ...)`, on a clone of the request.
        let mut attempt = req.req.clone();
        attempt.conn_options = *conn;
        let llm = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        let mut trace = if recovery {
            telemetry::LlmTrace::recovery(&req.cx, llm.as_ref(), &attempt)
        } else {
            telemetry::LlmTrace::start(&req.cx, llm.as_ref(), &attempt)
        };
        let run = trace.attempt();
        match llm.chat(trace.cx(), &attempt).await {
            Ok(source) => Ok(Box::new(Traced { source, trace: Some(traced::Llm { trace, run }) })),
            Err(error) => {
                trace.end_attempt(run, Some(&error));
                trace.fail(&error);
                Err(error)
            }
        }
    }
}

/// What the chain replays to each entry: the whole request, because recovery replays it
/// (`llm:174-200`), and the adapter span its `llm_request` spans open under.
#[derive(Clone)]
pub struct LlmAttempt {
    pub req: ChatRequest,
    pub cx: telemetry::Cx,
}

#[async_trait]
impl Backend for LlmBackend {
    type Req = LlmAttempt;
    type Session = LlmStream;

    fn resource_id(&self) -> lk_agents_abi::resources::ResourceId {
        super::backend_resource(self.inner.as_ref(), self.inner.instance_id())
    }

    fn label(&self) -> String {
        self.inner.label()
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.inner.failed(error, recoverable);
    }

    async fn open(
        &self,
        req: &Self::Req,
        conn: &lk_agents_abi::ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        self.request(req, conn, false).await
    }

    async fn recover(
        &self,
        req: &Self::Req,
        conn: &lk_agents_abi::ApiConnectOptions,
        _slot: RecoveryMode,
        stop: CancellationToken,
    ) -> Result<bool, ProviderError> {
        // Drain and discard: reaching the end without an error is LLM's recovery signal.
        drain_probe(self.request(req, conn, true).await?, self.clock.as_ref(), stop).await
    }
}

/// A host-constructed chain runs on the executor and clock its env lends.
pub struct FallbackLlm {
    chain: Arc<FallbackChain<LlmBackend>>,
    /// The name logs give it: its host class's when its host built it.
    label: String,
}

impl FallbackLlm {
    /// The adapter's spans record under the context each `chat` call runs in.
    pub fn new(
        llms: Vec<Arc<dyn Llm>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, Invalid> {
        let entries =
            llms.into_iter().map(|llm| LlmBackend::new(llm, Arc::clone(&clock))).collect();
        let chain = FallbackChain::new(entries, policy, clock, spawner, observer)?;
        Ok(FallbackLlm { chain: Arc::new(chain), label: NativeLabel::LlmFallback.as_str().into() })
    }

    /// The adapter's span for one request, under the caller's context; its entries' requests
    /// open under it.
    fn adapter_span(
        &self,
        cx: &telemetry::Cx,
    ) -> telemetry::ScopedSpan<telemetry::LlmFallbackAdapter> {
        let id = self.active().id();
        telemetry::start(
            cx,
            telemetry::LlmFallbackAdapter {
                attempt: telemetry::Attempt {
                    provider: telemetry::provider_name(&id.provider),
                    model: id.model,
                    retry_count: 0,
                },
                stream: true,
                output_type: telemetry::OutputType::Text,
            },
        )
    }

    pub fn chain(&self) -> &Arc<FallbackChain<LlmBackend>> {
        &self.chain
    }

    /// The `LLM` that most recently served, for metrics labelling. `_active_instance`.
    pub fn active(&self) -> Arc<dyn Llm> {
        Arc::clone(&self.chain.active().inner)
    }

    /// Cancel every recovery probe in flight, idempotently. Dropping does not (§4.2 #7).
    pub fn aclose(&self) {
        self.chain.aclose();
    }
}

/// `llm.FallbackAdapter`'s engine as the host builds it. The public class stays Python (users
/// construct, subclass and call it standalone); its `chat()` opens the chain over its own objects.
impl lk_agents_abi::llm::fallback::FallbackLlm for FallbackLlm {
    type Chunks = FallbackLlmChunks;
    type Env = ProcessEnv;

    /// `llms` in order; the keywords are `FallbackAdapter.__init__`'s. Each entry is bound to
    /// `max_retry = 0`, distinct from the chain's `max_retry_per_llm`: the chain is the retry loop,
    /// and a plugin's own would multiply it (N×M attempts).
    fn new(
        env: &ProcessEnv,
        label: String,
        observer: Arc<dyn ChainObserver>,
        llms: Vec<Arc<dyn Llm>>,
        attempt_timeout: f64,
        max_retry_per_llm: u32,
        retry_interval: f64,
        sticky: bool,
    ) -> Result<Self, Invalid> {
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_llm,
            attempt_timeout: Some(attempt_timeout),
            retry_interval: Some(retry_interval),
            sticky,
        };
        let spawner = (env.spawner)().map_err(|e| Invalid(e.to_string()))?;
        let chain = FallbackLlm::new(llms, policy, env.clock.clone(), spawner, observer)?;
        Ok(FallbackLlm { label, ..chain })
    }

    /// Which entry most recently served, as a position in the host's list (the reference's
    /// `_active_instance`; 0 before any traffic).
    fn active_index(&self) -> usize {
        self.chain.active_index()
    }

    /// Every recovery probe in flight closes its stream, awaited (bounded).
    async fn aclose(&self) {
        self.chain.close().await;
    }

    /// A completion over the caller's own `ChatContext` and tools (the host objects every entry
    /// is handed, §1.11); the chain picks an entry on the first `recv()`.
    async fn chat(
        &self,
        cx: telemetry::Cx,
        chat_ctx: HostChatContext,
        tools: Option<HostTools>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<serde_json::Value>,
        extra_kwargs: Option<Extra>,
        conn_options: Option<lk_agents_abi::ApiConnectOptions>,
    ) -> Result<Self::Chunks, ProviderError> {
        let (tools, tools_host) = match tools {
            Some(tools) => (tools.specs, Some(tools.host)),
            None => (Vec::new(), None),
        };
        let req = ChatRequest {
            chat_ctx: ChatContext::new(chat_ctx.items),
            chat_ctx_host: Some(chat_ctx.host),
            tools,
            tools_host,
            conn_options: conn_options.unwrap_or_else(crate::fallback::caller_default),
            parallel_tool_calls,
            tool_choice: tool_choice.as_ref().and_then(ToolChoice::from_host_json),
            extra: extra_kwargs,
            ..ChatRequest::default()
        };
        Llm::chat(self, &cx, &req).await.map(FallbackLlmChunks)
    }
}

/// The chain's native source, consumed through the generated shared stream bridge.
pub struct FallbackLlmChunks(LlmStream);

impl StreamOutput<LlmChunk> for FallbackLlmChunks {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        self.0.next()
    }

    fn cancel_safe(&self) -> bool {
        self.0.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.0.close(deadline)
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for FallbackLlmChunks {}

impl Llm for FallbackLlm {
    /// The chain retries and falls back on its own.
    fn retries_itself(&self) -> bool {
        true
    }

    /// The adapter's `error` event.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.chain.report(error, recoverable);
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn request_span(&self) -> telemetry::RequestSpan {
        telemetry::RequestSpan::FallbackAdapter
    }

    /// The active member's identity: metrics name the instance that served (L-25).
    fn id(&self) -> lk_agents_abi::ProviderId {
        self.active().id()
    }

    fn prewarm(&self, cx: &telemetry::Cx) {
        for entry in self.chain.entries() {
            entry.inner().prewarm(cx);
        }
    }

    /// The active entry's model, not the trait's `"unknown"`: `_evaluate_with_llm` reads it to
    /// exclude `temperature` on gpt-5.
    fn model(&self) -> String {
        self.chain.active().inner().model()
    }

    fn chat<'a>(
        &'a self,
        cx: &telemetry::Cx,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let span = self.adapter_span(cx);
        Box::pin(async move {
            let req = LlmAttempt { req: req.clone(), cx: span.cx().clone() };
            let call = ChainCall {
                conn_options: req.req.conn_options,
                accepted_at: self.chain.clock().now(),
                slot: RecoveryMode::OneShot,
            };
            let settled = move |chain: &FallbackChain<LlmBackend>,
                                result: Result<(), &ProviderError>| {
                match result {
                    Ok(()) => telemetry::end(span, Some(served(chain))),
                    Err(error) => telemetry::end(span, error),
                }
            };
            let source = self.chain.serve_stream(req, call, Arc::new(LlmLeg), settled, ());
            Ok(Box::new(source) as LlmStream)
        })
    }
}

/// An entry's attempt: its chunks forwarded until the stream ends, then the stream closed.
struct LlmLeg;

impl ServeAttempt<LlmBackend, LlmChunk> for LlmLeg {
    fn serve<'a>(
        self: Arc<Self>,
        chain: Arc<FallbackChain<LlmBackend>>,
        index: usize,
        mut stream: LlmStream,
        tx: &'a mut mpsc::Sender<Result<LlmChunk, ProviderError>>,
        stop: CancellationToken,
    ) -> BoxFuture<'a, Attempt<()>> {
        Box::pin(async move {
            // Only `has_response()` chunks count as output (`llm:281-285`); usage-only chunks do
            // not block a retry.
            let outcome =
                forward(&chain, index, stream.as_mut(), &stop, tx, LlmChunk::has_response).await;
            stream.close(chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
            outcome
        })
    }
}

/// The entry that served: its label, position and model, and its provider.
fn served(chain: &FallbackChain<LlmBackend>) -> telemetry::LlmFallbackChoice {
    let index = chain.active_index();
    let llm = chain.active().inner();
    let id = llm.id();
    telemetry::LlmFallbackChoice {
        label: llm.label(),
        index: u32::try_from(index).unwrap_or(u32::MAX),
        response_model: id.model,
        provider: telemetry::provider_name(&id.provider),
    }
}

#[cfg(test)]
#[path = "llm_tests.rs"]
mod tests;

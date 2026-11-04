//! `llm/fallback_adapter.py`, as a [`FallbackChain`] plus glue. The adapter is itself an [`Llm`].
//! `chat()` returns at once and the chain runs as a task feeding an `mpsc`
//! (`FallbackLLMStream._run`), so its failure surfaces on the first `next()`.

use lk_agents_abi::{Extra, HostRef, ToolSpec};
use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::LlmChunk;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_abi::{ChatRequest, Closed, Llm, LlmStream, StreamOutput, ToolChoice, Ts};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT};
use tokio::sync::mpsc;

use crate::fallback::ChainPolicy;
use crate::fallback::chain::{
    Attempt, Backend, ChainCall, ChainObserver, ChainOutcome, FallbackChain, NoBackends, Recovery,
    RecoveryMode,
};
use crate::fallback::channel::{ChainTask, ChannelSource, drain_probe, forward};
use crate::telemetry;
use lk_agents_net::env::ProcessEnv;

pub struct LlmBackend {
    inner: Arc<dyn Llm>,
    clock: Arc<dyn Clock>,
    /// Records each request the adapter makes of this entry, under the adapter's span.
    tracer: telemetry::Tracer,
}

impl LlmBackend {
    /// The provider this entry wraps, so `FallbackLlm::model()` can answer from the serving entry.
    pub fn inner(&self) -> &Arc<dyn Llm> {
        &self.inner
    }

    pub fn new(inner: Arc<dyn Llm>, clock: Arc<dyn Clock>, tracer: telemetry::Tracer) -> Arc<Self> {
        Arc::new(LlmBackend { inner, clock, tracer })
    }

    /// One request of this entry: an `llm_request` under the adapter's span when the caller gave
    /// one, a recovery probe's marked as such.
    async fn request(
        &self,
        req: &ChatRequest,
        conn: &lk_agents_abi::ApiConnectOptions,
        recovery: bool,
    ) -> Result<LlmStream, ProviderError> {
        // `dataclasses.replace(self._conn_options, ...)`, on a clone of the request.
        let mut attempt = req.clone();
        attempt.conn_options = *conn;
        let llm = self.inner.with_conn_options(conn).unwrap_or_else(|| self.inner.clone());
        let Some(parent) = attempt.trace.take() else { return llm.chat(&attempt).await };
        let cx = self.tracer.resume(&parent);
        let mut trace = if recovery {
            telemetry::LlmTrace::recovery(&cx, llm.as_ref(), &mut attempt)
        } else {
            telemetry::LlmTrace::start(&cx, llm.as_ref(), &mut attempt)
        };
        let run = trace.attempt();
        match llm.chat(&attempt).await {
            Ok(stream) => Ok(Box::new(TracedStream { stream, trace: Some((trace, run)) })),
            Err(error) => {
                trace.end_attempt(run, Some(&error));
                trace.fail(&error);
                Err(error)
            }
        }
    }
}

/// An entry's stream, ending its `llm_request` with the stream: at its end, on its error, or as
/// cancelled when closed before either.
struct TracedStream {
    stream: LlmStream,
    trace: Option<(telemetry::LlmTrace, Option<telemetry::ScopedSpan<telemetry::LlmRequestRun>>)>,
}

impl StreamOutput<LlmChunk> for TracedStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            let next = self.stream.next().await;
            match &next {
                Ok(Some(chunk)) => {
                    if let Some((trace, _)) = &mut self.trace {
                        trace.observe(chunk);
                    }
                }
                Ok(None) => {
                    if let Some((mut trace, run)) = self.trace.take() {
                        trace.end_attempt(run, None);
                        trace.end(false, None);
                    }
                }
                Err(error) => {
                    if let Some((mut trace, run)) = self.trace.take() {
                        trace.end_attempt(run, Some(error));
                        trace.fail(error);
                    }
                }
            }
            next
        })
    }

    fn cancel_safe(&self) -> bool {
        self.stream.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        self.stream.close(deadline)
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for TracedStream {}

#[async_trait]
impl Backend for LlmBackend {
    /// The whole request, because recovery replays it (`llm:174-200`).
    type Req = ChatRequest;
    type Session = LlmStream;

    fn resource_id(&self) -> lk_agents_abi::resources::ResourceId {
        super::backend_resource(self.inner.as_ref(), self.inner.instance_id())
    }

    fn label(&self) -> String {
        self.inner.label()
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
        stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        // Drain and discard: reaching the end without an error is LLM's recovery signal.
        drain_probe(self.request(req, conn, true).await?, self.clock.as_ref(), stop).await
    }
}

/// A host-constructed chain runs on the executor and clock its env lends.
pub struct FallbackLlm {
    chain: Arc<FallbackChain<LlmBackend>>,
    tracer: telemetry::Tracer,
}

impl FallbackLlm {
    /// `tracer` records the adapter's spans when a caller hands it a context
    /// ([`ChatRequest::trace`]).
    pub fn new(
        llms: Vec<Arc<dyn Llm>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
        tracer: telemetry::Tracer,
    ) -> Result<Self, NoBackends> {
        let entries = llms
            .into_iter()
            .map(|llm| LlmBackend::new(llm, Arc::clone(&clock), tracer.clone()))
            .collect();
        let chain = FallbackChain::new(entries, policy, clock, spawner, observer)?;
        Ok(FallbackLlm { chain: Arc::new(chain), tracer })
    }

    /// The adapter's span for one request, under the caller's context; its entries' requests
    /// open under it.
    fn adapter_span(
        &self,
        req: &mut ChatRequest,
    ) -> Option<telemetry::ScopedSpan<telemetry::LlmFallbackAdapter>> {
        let parent = req.trace.take()?;
        let id = self.active().id();
        let span = telemetry::start(
            &self.tracer.resume(&parent),
            telemetry::LlmFallbackAdapter {
                attempt: telemetry::Attempt {
                    provider: telemetry::provider_name(&id.provider),
                    model: id.model,
                    retry_count: 0,
                },
                stream: true,
                output_type: "text".to_owned(),
            },
        );
        req.trace = Some(span.cx().trace_context());
        Some(span)
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
        llms: Vec<Arc<dyn Llm>>,
        attempt_timeout: f64,
        max_retry_per_llm: u32,
        retry_interval: f64,
        sticky: bool,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_llm,
            attempt_timeout: Some(attempt_timeout),
            retry_interval: Some(retry_interval),
            sticky,
        };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        FallbackLlm::new(
            llms,
            policy,
            env.clock.clone(),
            (env.spawner)().map_err(InvalidPayload)?,
            observer,
            env.tracer.clone(),
        )
        .map_err(|error| InvalidPayload(error.to_string()))
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
    #[allow(clippy::too_many_arguments, reason = "Preserve the public host operation signature")]
    async fn chat(
        &self,
        chat_ctx_host: HostRef,
        tools: Vec<ToolSpec>,
        tools_host: Option<HostRef>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<serde_json::Value>,
        extra_kwargs: Option<Extra>,
        conn_options: Option<lk_agents_abi::ApiConnectOptions>,
    ) -> Result<Self::Chunks, ProviderError> {
        let req = ChatRequest {
            chat_ctx_host: Some(chat_ctx_host),
            tools,
            tools_host,
            conn_options: conn_options.unwrap_or_else(crate::fallback::caller_default),
            parallel_tool_calls,
            tool_choice: tool_choice.as_ref().and_then(ToolChoice::from_host_json),
            extra: extra_kwargs,
            ..ChatRequest::default()
        };
        Llm::chat(self, &req).await.map(FallbackLlmChunks)
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
    fn label(&self) -> String {
        "llm.FallbackAdapter".into()
    }

    /// The active member's identity: metrics name the instance that served (L-25).
    fn id(&self) -> lk_agents_abi::ProviderId {
        self.active().id()
    }

    fn prewarm(&self) {
        for entry in self.chain.entries() {
            entry.inner().prewarm();
        }
    }

    /// The active entry's model, not the trait's `"unknown"`: `_evaluate_with_llm` reads it to
    /// exclude `temperature` on gpt-5.
    fn model(&self) -> String {
        self.chain.active().inner().model()
    }

    fn chat<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            // Bounded, so a slow reader backpressures the provider; the text leg may not drop (§1.2d).
            let (tx, rx) = mpsc::channel::<Result<LlmChunk, ProviderError>>(32);

            let chain = Arc::clone(&self.chain);
            let drain_chain = Arc::clone(&self.chain);
            let mut req = req.clone();
            let span = self.adapter_span(&mut req);
            let call = ChainCall {
                conn_options: req.conn_options,
                accepted_at: self.chain.clock().now(),
                slot: RecoveryMode::OneShot,
            };

            let (stop_rx, done_tx, chain_task) = ChainTask::pair(chain.clock());
            let handle = self.chain.spawn_stream(async move {
                let mut tx = tx;
                let result = chain
                    .run_cancellable(
                        &req,
                        call,
                        &mut tx,
                        stop_rx,
                        move |index, _backend, mut stream, tx, mut stop| {
                            let chain = Arc::clone(&drain_chain);
                            Box::pin(async move {
                                // Only `has_response()` chunks count as output (`llm:281-285`);
                                // usage-only chunks do not block a retry.
                                let outcome = forward(
                                    &chain,
                                    index,
                                    stream.as_mut(),
                                    &mut stop,
                                    tx,
                                    LlmChunk::has_response,
                                )
                                .await;
                                stream
                                    .close(
                                        chain.clock().now().saturating_add(HOST_TEARDOWN_TIMEOUT),
                                    )
                                    .await;
                                outcome
                            }) as BoxFuture<'_, Attempt<()>>
                        },
                    )
                    .await;

                match result {
                    Ok(ChainOutcome::Served(())) => {
                        if let Some(span) = span {
                            telemetry::end_with(span, Some(served(&chain)));
                        }
                    }
                    Err(e) => {
                        if let Some(span) = &span {
                            telemetry::fail(&span, &e);
                        }
                        let _ = tx.send(Err(e)).await;
                    }
                }
                let _ = done_tx.send(());
            });

            Ok(Box::new(ChannelSource { rx, task: Some(handle), chain: chain_task }) as LlmStream)
        })
    }
}

/// The entry that served: its label, position and model, and its provider.
fn served(chain: &FallbackChain<LlmBackend>) -> telemetry::LlmFallbackChoice {
    let index = chain.active_index();
    let llm = chain.active().inner();
    let id = llm.id();
    telemetry::LlmFallbackChoice {
        choice: telemetry::FallbackChoice {
            label: llm.label(),
            index: u32::try_from(index).unwrap_or(u32::MAX),
            response_model: id.model,
        },
        provider: telemetry::provider_name(&id.provider),
    }
}

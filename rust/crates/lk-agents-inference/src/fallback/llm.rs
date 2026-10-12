//! `llm/fallback_adapter.py`, as a [`FallbackChain`] plus glue. The adapter is itself an [`Llm`].
//! `chat()` returns at once and the chain runs as a task feeding an `mpsc`
//! (`FallbackLLMStream._run`), so its failure surfaces on the first `next()`.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::{ChatRequest, Extra, HostRef, Llm, LlmStream, ToolChoice, ToolSpec};
use lk_agents_abi::{BoxFuture, Spawner, SpawnerExt};
use lk_agents_abi::Clock;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::LlmChunk;
use tokio::sync::mpsc;

use crate::fallback::chain::{
    Attempt, Backend, ChainObserver, ChainOutcome, FallbackChain, NoBackends, Recovery,
    RecoverySlot,
};
use crate::fallback::policy::ChainPolicy;
use crate::fallback::channel::{ChainTask, ChannelSource, drain_probe, forward};

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
}

#[async_trait]
impl Backend for LlmBackend {
    /// The whole request, because recovery replays it (`llm:174-200`).
    type Req = ChatRequest;
    type Session = LlmStream;

    fn label(&self) -> String {
        self.inner.label()
    }

    async fn open(
        &self,
        req: &Self::Req,
        conn: &lk_agents_abi::ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        // `dataclasses.replace(self._conn_options, ...)`, on a clone of the request.
        let mut attempt = req.clone();
        attempt.conn_options = *conn;
        self.inner.chat(&attempt).await
    }

    async fn recover(
        &self,
        req: &Self::Req,
        conn: &lk_agents_abi::ApiConnectOptions,
        _slot: RecoverySlot,
        stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        // Drain and discard: reaching the end without an error is LLM's recovery signal.
        drain_probe(self.open(req, conn).await?, self.clock.as_ref(), stop).await
    }
}

pub struct FallbackLlm {
    chain: Arc<FallbackChain<LlmBackend>>,
    spawner: Arc<dyn Spawner>,
}

impl FallbackLlm {
    pub fn new(
        llms: Vec<Arc<dyn Llm>>,
        policy: ChainPolicy,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        observer: Arc<dyn ChainObserver>,
    ) -> Result<Self, NoBackends> {
        let entries =
            llms.into_iter().map(|llm| LlmBackend::new(llm, Arc::clone(&clock))).collect();
        let chain = FallbackChain::new(entries, policy, clock, Arc::clone(&spawner), observer)?;
        Ok(FallbackLlm { chain: Arc::new(chain), spawner })
    }

    pub fn chain(&self) -> &Arc<FallbackChain<LlmBackend>> {
        &self.chain
    }

    /// The `LLM` that most recently served, for metrics labelling. `_active_instance`.
    pub fn active(&self) -> Arc<dyn Llm> {
        Arc::clone(&self.chain.active().inner)
    }
}

/// `llm.FallbackAdapter`'s engine as the host builds it. The public class stays Python (users
/// construct, subclass and call it standalone); its `chat()` calls [`chat_for_host`](Self::chat_for_host).
#[lk_agents_abi::host(object(cell = "arc", as_dyn(Llm = as_llm)))]
impl FallbackLlm {
    /// `llms` in order; the keywords are `FallbackAdapter.__init__`'s. Each entry is bound to
    /// `max_retry = 0`, distinct from the chain's `max_retry_per_llm`: the chain is the retry loop,
    /// and a plugin's own would multiply it (N×M attempts).
    #[host(new(
        kw_only = "attempt_timeout",
        defaults(attempt_timeout = 5.0, max_retry_per_llm = 0, retry_interval = 0.5, retry_on_chunk_sent = false)
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn for_host(
        llms: Vec<Arc<dyn Llm>>,
        attempt_timeout: f64,
        max_retry_per_llm: u32,
        retry_interval: f64,
        retry_on_chunk_sent: bool,
        on_availability: Option<Arc<dyn ChainObserver>>,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Result<Self, NoBackends> {
        let seam = lk_agents_abi::ApiConnectOptions { max_retry: 0, ..Default::default() };
        let llms = llms.into_iter().map(|llm| llm.with_conn_options(&seam).unwrap_or(llm)).collect();
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_llm,
            attempt_timeout: Some(attempt_timeout),
            retry_interval: Some(retry_interval),
            ..ChainPolicy::llm_defaults()
        }
        .with_retry_on_chunk_sent(retry_on_chunk_sent);
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        FallbackLlm::new(llms, policy, clock, spawner, observer)
    }

    /// The chain as the provider the session holds (a native `_core.FallbackLlm` is not proxied).
    pub fn as_llm(self: &Arc<Self>) -> Arc<dyn Llm> {
        Arc::clone(self) as Arc<dyn Llm>
    }

    /// Which entry most recently served, as a position in the host's list (the reference's
    /// `_active_instance`; 0 before any traffic).
    #[host(getter)]
    pub fn active_index(&self) -> usize {
        self.chain.active_index()
    }

    /// Cancel every recovery probe in flight, idempotently. Dropping does not (§4.2 #7).
    pub fn aclose(&self) {
        self.chain.aclose();
    }

    /// `aclose()`: every recovery probe in flight closes its stream, awaited (bounded).
    #[host(method(name = "aclose"))]
    pub async fn aclose_for_host(&self) {
        self.chain.close().await;
    }

    /// Open a completion over the caller's own `ChatContext` and tools (the host objects every
    /// entry is handed, §1.11); the chain picks an entry on the first `recv()`.
    #[host(method(
        name = "chat",
        stream = lk_agents_abi::InferenceLlmStream,
        kw_only = "tools",
        defaults(tools = Vec::new())
    ))]
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_for_host(
        &self,
        chat_ctx_host: HostRef,
        tools: Vec<ToolSpec>,
        tools_host: Option<HostRef>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<serde_json::Value>,
        extra_kwargs: Option<Extra>,
    ) -> Result<LlmStream, ProviderError> {
        let req = ChatRequest {
            chat_ctx_host: Some(chat_ctx_host),
            tools,
            tools_host,
            conn_options: crate::fallback::caller_default(),
            parallel_tool_calls,
            tool_choice: tool_choice.as_ref().and_then(ToolChoice::from_host_json),
            extra: extra_kwargs,
            ..ChatRequest::default()
        };
        self.chat(&req).await
    }
}

#[async_trait]
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

    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        // Bounded, so a slow reader backpressures the provider; the text leg may not drop (§1.2d).
        let (tx, rx) = mpsc::channel::<Result<LlmChunk, ProviderError>>(32);

        let chain = Arc::clone(&self.chain);
        let drain_chain = Arc::clone(&self.chain);
        let req = req.clone();
        let caller_conn = req.conn_options;

        let (stop_rx, done_tx, chain_task) = ChainTask::pair(chain.clock());
        let handle = self.spawner.spawn_unboxed(async move {
            let mut tx = tx;
            let result = chain
                .run(
                    &req,
                    &caller_conn,
                    RecoverySlot::OneShot,
                    &mut tx,
                    move |index, _backend, mut stream, tx| {
                        let chain = Arc::clone(&drain_chain);
                        let mut stop = stop_rx.clone();
                        Box::pin(async move {
                            // Only `has_response()` chunks count as output (`llm:281-285`);
                            // usage-only chunks do not block a retry.
                            let outcome =
                                forward(&chain, index, stream.as_mut(), &mut stop, tx, LlmChunk::has_response)
                                    .await;
                            stream.close(chain.teardown_deadline()).await;
                            outcome
                        }) as BoxFuture<'_, Attempt<()>>
                    },
                )
                .await;

            match result {
                Ok(ChainOutcome::Served(())) | Ok(ChainOutcome::Truncated) => {}
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
            let _ = done_tx.send(());
        });

        Ok(Box::new(ChannelSource { rx, task: Some(handle), chain: chain_task }))
    }
}


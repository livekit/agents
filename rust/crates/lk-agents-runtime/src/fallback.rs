//! Host construction and streaming for the native LLM fallback chain.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::ChainObserver;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::{ChatRequest, Closed, Llm, LlmChunk, LlmStream, StreamSource, Ts};

/// Runtime-owned host facade. The inference engine remains the sole fallback-policy owner.
pub struct FallbackLlm(lk_agents_inference::fallback::FallbackLlm);

impl lk_agents_abi::llm::fallback::FallbackLlm for FallbackLlm {
    type Chunks = FallbackLlmChunks;

    fn new(
        llms: Vec<Arc<dyn Llm>>,
        attempt_timeout: f64,
        max_retry_per_llm: u32,
        retry_interval: f64,
        sticky: bool,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        #[cfg(feature = "python")]
        let spawner =
            <Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient();
        #[cfg(not(feature = "python"))]
        let spawner: Arc<dyn lk_agents_abi::Spawner> = Arc::new(lk_agents_abi::TokioSpawner::new(
            crate::process::handle().map_err(|error| InvalidPayload(error.to_string()))?,
        ));
        lk_agents_inference::fallback::FallbackLlm::for_host(
            llms,
            attempt_timeout,
            max_retry_per_llm,
            retry_interval,
            sticky,
            on_availability,
            crate::process::host_clock(),
            spawner,
        )
        .map(Self)
        .map_err(|error| InvalidPayload(error.to_string()))
    }

    fn active_index(&self) -> usize {
        self.0.active_index()
    }

    async fn aclose(&self) {
        self.0.aclose_for_host().await;
    }

    async fn chat(
        &self,
        chat_ctx_host: lk_agents_abi::HostRef,
        tools: Vec<lk_agents_abi::ToolSpec>,
        tools_host: Option<lk_agents_abi::HostRef>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<serde_json::Value>,
        extra_kwargs: Option<lk_agents_abi::Extra>,
        conn_options: Option<lk_agents_abi::ApiConnectOptions>,
    ) -> Result<Self::Chunks, ProviderError> {
        self.0
            .chat_for_host(
                chat_ctx_host,
                tools,
                tools_host,
                parallel_tool_calls,
                tool_choice,
                extra_kwargs,
                conn_options,
            )
            .await
            .map(FallbackLlmChunks)
    }
}

#[async_trait::async_trait]
impl Llm for FallbackLlm {
    fn label(&self) -> String {
        self.0.label()
    }

    fn model(&self) -> String {
        self.0.model()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.0.id()
    }

    fn prewarm(&self) {
        self.0.prewarm();
    }

    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        self.0.chat(req).await
    }
}

/// The chain's native source, consumed through the generated shared stream bridge.
pub struct FallbackLlmChunks(LlmStream);

#[async_trait::async_trait]
impl StreamSource<LlmChunk> for FallbackLlmChunks {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        self.0.next().await
    }

    fn cancel_safe(&self) -> bool {
        self.0.cancel_safe()
    }

    async fn close(self: Box<Self>, deadline: Ts) -> Closed {
        self.0.close(deadline).await
    }
}

#[cfg(all(test, not(feature = "python")))]
mod host_tests;

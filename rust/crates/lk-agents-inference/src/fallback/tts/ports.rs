//! Declared fallback resource surface over the injected speech factory.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::{self as port, ChainObserver};
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::{ApiConnectOptions, Tts, TtsCapabilities};

use crate::tts::stream::TtsStream;
use lk_agents_net::env::ProcessEnv;

use super::{FallbackTts, TokenLog};

impl port::FallbackTts for FallbackTts {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        ttss: Vec<Arc<dyn Tts>>,
        max_retry_per_tts: u32,
        sample_rate: Option<u32>,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        FallbackTts::for_host(
            ttss,
            max_retry_per_tts,
            sample_rate,
            on_availability,
            env.clock.clone(),
            (env.spawner)().map_err(InvalidPayload)?,
            env.speech_text,
        )
        .map_err(|error| InvalidPayload(error.to_string()))
    }
    fn active_index(&self) -> usize {
        self.active_index()
    }
    fn capabilities(&self) -> TtsCapabilities {
        self.merged_capabilities()
    }
    fn sample_rate(&self) -> u32 {
        self.output_sample_rate()
    }
    fn num_channels(&self) -> u16 {
        self.output_num_channels()
    }
    async fn aclose(&self) {
        self.aclose_for_host().await;
    }
    async fn synthesize(
        &self,
        text: String,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self::Stream, ProviderError> {
        self.synthesize_for_host(text, conn_options)
            .await
            .map(|source| TtsStream::new(None, source))
    }
    async fn stream(
        &self,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self::Stream, ProviderError> {
        self.stream_for_host(conn_options)
            .await
            .map(|(sink, source)| TtsStream::new(Some(sink), source))
    }
}

impl port::TtsTokenLog for TokenLog {
    fn new() -> Self {
        Self::default()
    }
    fn push(&self, text: String) {
        self.push_text(text);
    }
    fn flush(&self) {
        self.push_flush();
    }
    fn end(&self) {
        self.end_input();
    }
    fn snapshot(&self) -> (Vec<Option<String>>, bool) {
        self.replay_tokens()
    }
}

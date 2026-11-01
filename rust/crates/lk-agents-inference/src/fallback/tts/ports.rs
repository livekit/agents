//! Declared fallback resource surface over the injected speech factory.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::fallback::{self as port, ChainObserver};
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::{ApiConnectOptions, Tts, TtsCapabilities};

use crate::tts::stream::TtsStream;
use lk_agents_net::env::ProcessEnv;

use super::{FallbackTts, Token, TokenLog};
use crate::fallback::ChainPolicy;

impl port::FallbackTts for FallbackTts {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        ttss: Vec<Arc<dyn Tts>>,
        max_retry_per_tts: u32,
        attempt_timeout: Option<f64>,
        sample_rate: Option<u32>,
        on_availability: Option<Arc<dyn ChainObserver>>,
    ) -> Result<Self, InvalidPayload> {
        // `sample_rate = None` takes the maximum of the entries' rates and upsamples the rest.
        let policy = ChainPolicy {
            max_retry_per_backend: max_retry_per_tts,
            attempt_timeout,
            ..ChainPolicy::tts_defaults()
        };
        let observer = on_availability.unwrap_or_else(|| Arc::new(crate::fallback::IgnoreEvents));
        let spawner = (env.spawner)().map_err(InvalidPayload)?;
        FallbackTts::new(ttss, sample_rate, policy, env.clock.clone(), spawner, observer)
            .map_err(|error| InvalidPayload(error.to_string()))
    }
    fn active_index(&self) -> usize {
        self.chain.active_index()
    }
    fn capabilities(&self) -> TtsCapabilities {
        self.capabilities
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn num_channels(&self) -> u16 {
        self.num_channels
    }
    /// Every recovery probe in flight closes its stream, awaited (bounded).
    async fn aclose(&self) {
        self.chain.close().await;
    }
    async fn synthesize(
        &self,
        text: String,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self::Stream, ProviderError> {
        Tts::synthesize(&self.configured(conn_options), &text)
            .await
            .map(|source| TtsStream::new(None, source, self.chain.clock()))
    }
    async fn stream(
        &self,
        conn_options: Option<ApiConnectOptions>,
        expressive: bool,
    ) -> Result<Self::Stream, ProviderError> {
        Tts::stream(&self.configured(conn_options), expressive).await.map(|stream| {
            let (sink, source) = stream.split();
            TtsStream::new(Some(sink), source, self.chain.clock())
        })
    }
}

impl port::TtsTokenLog for TokenLog {
    fn new() -> Self {
        Self::default()
    }
    fn push(&self, text: String) {
        TokenLog::push(self, Token::Text(text));
    }
    fn flush(&self) {
        TokenLog::push(self, Token::Flush);
    }
    fn end(&self) {
        TokenLog::end(self);
    }
    /// Everything pushed so far (`None` for a flush) and whether the input ended.
    fn snapshot(&self) -> (Vec<Option<String>>, bool) {
        let (tokens, ended) = TokenLog::snapshot(self);
        let tokens = tokens
            .into_iter()
            .map(|token| match token {
                Token::Text(text) => Some(text),
                Token::Flush => None,
            })
            .collect();
        (tokens, ended)
    }
}

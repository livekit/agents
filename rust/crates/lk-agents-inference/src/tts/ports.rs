//! Native synthesis constructor services selected by the voice/text runtime.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::inference::fallback::TtsFallbackModel;
use lk_agents_abi::inference::tts_client::InferenceTts;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::tts::text::TtsTextProcessorFactory;
use lk_agents_abi::{ApiConnectOptions, Clock, Tts};

use super::stream::TtsStream;
use super::{NativeTts, TtsOptions};

/// What a host-constructed synthesis runs on: the process's services and its speech text.
pub trait TtsHost: TtsTextProcessorFactory + lk_agents_net::env::Env {}

impl<E: TtsTextProcessorFactory + lk_agents_net::env::Env> TtsHost for E {}

impl<P: TtsHost> InferenceTts for NativeTts<P> {
    type Stream = TtsStream;

    fn new(
        model: String,
        base_url: String,
        api_key: String,
        api_secret: String,
        conn_options: ApiConnectOptions,
        voice: Option<String>,
        language: Option<String>,
        encoding: String,
        sample_rate: u32,
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        fallback: Option<Vec<TtsFallbackModel>>,
    ) -> Result<Self, InvalidPayload> {
        let opts = TtsOptions {
            model,
            base_url,
            api_key,
            api_secret,
            conn_options,
            voice,
            language,
            encoding,
            sample_rate,
            extra_kwargs: extra_kwargs.unwrap_or_default(),
            fallback: fallback.unwrap_or_default(),
        };
        Ok(NativeTts::new(
            opts,
            Arc::new(lk_agents_net::websocket::TungsteniteTransport),
            P::spawner().map_err(InvalidPayload)?,
        )
        .with_clock(P::clock()))
    }

    fn set_expressive(&self, expressive: bool) {
        self.set_expressive(expressive);
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }
    fn aligned_transcript(&self) -> bool {
        self.aligned_transcript()
    }
    fn bind_clock(&self, clock: Arc<dyn Clock>) {
        self.bind_clock(clock);
    }
    fn prewarm(&self, headers: Option<Vec<(String, String)>>) {
        self.prewarm_identified(headers.map(|h| h.into_iter().collect()));
    }
    fn identified(&self, headers: Vec<(String, String)>, reporter: Option<Arc<dyn Tts>>) -> Self {
        self.identified(headers.into_iter().collect(), reporter)
    }
    fn update_options(
        &self,
        model: Option<String>,
        voice: Option<String>,
        language: Option<String>,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.update_options(
            model.as_deref(),
            voice.as_deref(),
            language.as_deref(),
            extra.as_ref(),
        );
    }
    async fn stream(
        &self,
        headers: Option<Vec<(String, String)>>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self::Stream, ProviderError> {
        self.stream_for_host(headers, conn_options)
            .await
            .map(|(sink, source)| TtsStream::new(Some(sink), source))
    }
}

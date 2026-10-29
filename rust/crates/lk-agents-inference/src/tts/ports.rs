//! Native synthesis constructor services selected by the voice/text runtime.

use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::inference::fallback::TtsFallbackModel;
use lk_agents_abi::inference::tts_client::InferenceTts;
use lk_agents_abi::llm::inference::InvalidPayload;
use lk_agents_abi::{ApiConnectOptions, Tts};

use lk_agents_net::env::ProcessEnv;

use super::stream::TtsStream;
use super::{GatewayTts, TtsOptions};

impl InferenceTts for GatewayTts {
    type Stream = TtsStream;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
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
        Ok(GatewayTts::new(
            opts,
            Arc::new(lk_agents_net::websocket::TungsteniteTransport),
            (env.spawner)().map_err(InvalidPayload)?,
            env.clock.clone(),
            env.speech_text,
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }
    fn aligned_transcript(&self) -> bool {
        self.aligned_transcript()
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
    /// Open a duplex synthesis. The gateway has no one-shot endpoint, so `synthesize` is this plus
    /// one push and `end_input`. `headers` is `get_inference_headers()`, read by the host in the
    /// synthesis's job context; each sentence pushed goes out as it arrives.
    async fn stream(
        &self,
        headers: Option<Vec<(String, String)>>,
        conn_options: Option<ApiConnectOptions>,
        expressive: bool,
    ) -> Result<Self::Stream, ProviderError> {
        let (sink, source) = self
            .configured(conn_options)
            .stream_identified(headers.map(|pairs| pairs.into_iter().collect()), expressive)?;
        Ok(TtsStream::new(Some(Box::new(sink)), Box::new(source)))
    }
}

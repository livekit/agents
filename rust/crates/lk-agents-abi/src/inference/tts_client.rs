//! The native gateway synthesis resource. Runtime owns its connections and secrets.

use super::fallback::TtsFallbackModel;
use crate::errors::ProviderError;
use crate::{ApiConnectOptions, Tts};

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`.
#[crate::host(class(name = "InferenceTts", native = Tts))]
pub trait InferenceTts: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "TtsStream"))]
    type Stream;
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw)] base_url: String,
        api_key: String,
        api_secret: String,
        conn_options: ApiConnectOptions,
        #[host(default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = "pcm_s16le")] encoding: String,
        #[host(default = 24_000)] sample_rate: u32,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        #[host(default = None, shape_error(items, kind = "ValueError", message = "each fallback entry must be a dict"))]
        fallback: Option<Vec<TtsFallbackModel>>,
    ) -> Result<Self, crate::llm::inference::InvalidPayload>;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(getter)]
    fn aligned_transcript(&self) -> bool;
    #[host(call)]
    fn prewarm(&self, #[host(kw, default = None)] headers: Option<Vec<(String, String)>>);
    #[host(call)]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        #[host(default = None)] reporter: Option<std::sync::Arc<dyn Tts>>,
    ) -> Self;
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] voice: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    );
    #[host(await)]
    async fn stream(
        &self,
        #[host(kw, default = None)] headers: Option<Vec<(String, String)>>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = false)] expressive: bool,
    ) -> Result<Self::Stream, ProviderError>;
}

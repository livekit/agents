//! `inference.LLM`'s native client: the LiveKit Inference gateway, implemented by the runtime.

use std::sync::Arc;

use crate::errors::ApiError;
use crate::llm::llm::{Llm, ToolChoice, ToolSpec};
use crate::utils::stream::StreamSource;
use crate::{ApiConnectOptions, ChatItem, Extra, LlmChunk};

/// The gateway LLM behind `inference.LLM`. Its options are shared by every view
/// (`identified`); each view's requests carry its own job's identification.
#[crate::host(class(name = "InferenceLlm", native = Llm))]
pub trait InferenceLlm: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// One completion's chunks, pulled by the host.
    #[host(source(item = LlmChunk, class = "InferenceLlmChunks"))]
    type Chunks: StreamSource<LlmChunk> + 'static;

    /// A client of `model` on the gateway at `base_url`. Starts nothing.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw, default = None)] provider: Option<String>,
        base_url: String,
        api_key: String,
        api_secret: String,
        #[host(default = None)] inference_class: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
    ) -> Result<Self, ApiError>;

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options, its requests carrying that job. `reporter` is the host
    /// `llm.LLM` whose `error` events report the attempts this view retries.
    #[host(call)]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        #[host(default = None)] reporter: Option<Arc<dyn Llm>>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> String;

    /// The persistent request options, a copy.
    #[host(getter)]
    fn extra_kwargs(&self) -> Extra;

    /// Read by the next `chat()`. `extra_kwargs` replaces the persistent options.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
    );

    /// Opens a completion of `items`; no I/O until the first pull. `headers` identify the
    /// caller's job, read in its context.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        items: Vec<ChatItem>,
        #[host(kw)] tools: Vec<ToolSpec>,
        conn_options: ApiConnectOptions,
        #[host(default = None)] parallel_tool_calls: Option<bool>,
        #[host(default = None)] tool_choice: Option<ToolChoice>,
        #[host(default = None)] response_format: Option<serde_json::Value>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
        #[host(default = None)] headers: Option<Vec<(String, String)>>,
    ) -> Result<Self::Chunks, ApiError>;
}

/// A provider payload that is not its declared wire shape.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidPayload(pub String);

/// Accumulates the streamed choices of one OpenAI-compatible completion into chunks, filtering
/// the model's thinking tags: `inference.LLMStream` over a plugin's own client.
#[crate::host(class(name = "LlmChoiceParser"))]
pub trait ChoiceParser: Send + Sized + 'static {
    #[host(new)]
    fn new(model: String) -> Self;

    /// The chunk `choice` (one streamed choice of completion `id`) completes, if any.
    #[host(call)]
    fn parse(
        &mut self,
        id: String,
        choice: serde_json::Value,
    ) -> Result<Option<LlmChunk>, InvalidPayload>;
}

/// The completion wire rules `inference.LLMStream` applies around a plugin's own client.
#[crate::host(functions)]
pub trait CompletionWire {
    /// The usage chunk of a streamed completion chunk, when it reports usage.
    #[host(function)]
    fn llm_usage_chunk(chunk: serde_json::Value) -> Result<Option<LlmChunk>, InvalidPayload>;
}

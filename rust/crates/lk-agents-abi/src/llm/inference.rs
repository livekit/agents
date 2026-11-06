//! `inference.LLM`'s native client: the LiveKit Inference gateway, implemented by the runtime.

use std::sync::Arc;

use crate::errors::ApiError;
use crate::llm::types::{Llm, ToolChoice, ToolSpec};
use crate::utils::stream::StreamOutput;
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
    type Chunks: StreamOutput<LlmChunk> + 'static;

    /// A client of `model` at `base_url`. Starts nothing. With `api_secret`, `base_url` is the
    /// LiveKit Inference gateway; without, `api_key` is an OpenAI-compatible provider's own key,
    /// sent as the bearer with no LiveKit header. `label` is the host class it reports as.
    /// `strict_tool_schema`, `gzip_body` and `msgpack_body` suit a provider that refuses strict
    /// tool schemas or accepts compressed or msgpack bodies. `inference.LLM`'s mixin builds it on
    /// first use, after a fork, from the LLM's attributes.
    #[host(new)]
    #[host(facade(
        module = "provider_hosts",
        class = "InferenceLlmHost",
        target = "_native",
        cached,
        legs(chat),
        leg = "_core_native",
        annotation(extra_kwargs = "dict[str, object]")
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw, default = None)] provider: Option<String>,
        base_url: String,
        api_key: String,
        api_secret: Option<String>,
        label: String,
        #[host(default = None)] inference_class: Option<String>,
        #[host(kw, default = true)] strict_tool_schema: bool,
        #[host(kw, default = false)] gzip_body: bool,
        #[host(kw, default = false)] msgpack_body: bool,
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
    #[host(facade(module = "provider_hosts", class = "InferenceLlmHost", target = "_native"))]
    fn model(&self) -> String;

    /// The persistent request options, a copy.
    #[host(getter)]
    fn extra_kwargs(&self) -> Extra;

    /// Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs.
    #[host(call)]
    #[host(facade(
        module = "provider_hosts",
        class = "InferenceLlmHost",
        target = "_native",
        not_given,
        annotation(
            model = ".inference_llm.LLMModels | str",
            extra_kwargs = ".llm_provider_options.ChatCompletionOptions | dict[str, Any]"
        ),
        resolve(extra_kwargs = "builtins.dict")
    ))]
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

// Public data contracts. Python behavior remains in its adapters.

/// `ChatCompletionOptions.prompt_cache_retention`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheRetention {
    InMemory,
    #[host(rename = "24h")]
    Day,
}

/// `ChatCompletionOptions.service_tier`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

/// `ChatCompletionOptions.verbosity`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

#[crate::host(record(
    module = "llm_provider_options",
    path = "livekit.agents.inference.llm",
    typed_dict
))]
pub struct ChatCompletionOptions {
    #[host(annotation = "float | None")]
    pub frequency_penalty: crate::Maybe<Option<f64>>,
    #[host(annotation = "dict[str, int] | None")]
    pub logit_bias: crate::Maybe<Option<std::collections::BTreeMap<String, i64>>>,
    #[host(annotation = "bool | None")]
    pub logprobs: crate::Maybe<Option<bool>>,
    #[host(annotation = "int | None")]
    pub max_completion_tokens: crate::Maybe<Option<i64>>,
    #[host(annotation = "int | None")]
    pub max_tokens: crate::Maybe<Option<i64>>,
    #[host(annotation = "openai.types.shared_params.Metadata | None")]
    pub metadata: crate::Maybe<Option<serde_json::Value>>,
    pub modalities: crate::Maybe<Option<Vec<crate::Modality>>>,
    #[host(annotation = "int | None")]
    pub n: crate::Maybe<Option<i64>>,
    #[host(annotation = "bool")]
    pub parallel_tool_calls: crate::Maybe<bool>,
    #[host(annotation = "openai.types.chat.ChatCompletionPredictionContentParam | None")]
    pub prediction: crate::Maybe<Option<serde_json::Value>>,
    #[host(annotation = "float | None")]
    pub presence_penalty: crate::Maybe<Option<f64>>,
    #[host(annotation = "str")]
    pub prompt_cache_key: crate::Maybe<String>,
    pub prompt_cache_retention: crate::Maybe<Option<PromptCacheRetention>>,
    #[host(annotation = "openai.types.shared.reasoning_effort.ReasoningEffort | None")]
    pub reasoning_effort: crate::Maybe<Option<serde_json::Value>>,
    #[host(annotation = "str")]
    pub safety_identifier: crate::Maybe<String>,
    #[host(annotation = "int | None")]
    pub seed: crate::Maybe<Option<i64>>,
    pub service_tier: crate::Maybe<Option<ServiceTier>>,
    #[host(annotation = "str | None | list[str] | None")]
    pub stop: crate::Maybe<Option<serde_json::Value>>,
    #[host(annotation = "bool | None")]
    pub store: crate::Maybe<Option<bool>>,
    #[host(annotation = "float | None")]
    pub temperature: crate::Maybe<Option<f64>>,
    #[host(annotation = "int | None")]
    pub top_logprobs: crate::Maybe<Option<i64>>,
    #[host(annotation = "float | None")]
    pub top_p: crate::Maybe<Option<f64>>,
    #[host(annotation = "str")]
    pub user: crate::Maybe<String>,
    pub verbosity: crate::Maybe<Option<Verbosity>>,
    #[host(annotation = "openai.types.chat.completion_create_params.WebSearchOptions")]
    pub web_search_options: crate::Maybe<serde_json::Value>,
    #[host(annotation = "livekit.agents.llm.ToolChoice")]
    pub tool_choice: crate::Maybe<serde_json::Value>,
}

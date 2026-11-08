//! `inference.LLM`'s native client: the LiveKit Inference gateway, implemented by the runtime.

use std::sync::Arc;

use crate::errors::{ApiError, Invalid};
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

    /// A client of `model`. Starts nothing. On the LiveKit Inference gateway by default, its
    /// URL and key pair resolved from the given values, the environment and the server's
    /// settings; with `endpoint`, an OpenAI-compatible provider called with its own key and no
    /// LiveKit header.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        model: String,
        #[host(kw, default = None)] provider: Option<String>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] inference_class: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
        #[host(default = None)] endpoint: Option<ProviderEndpoint>,
    ) -> Result<Self, Invalid>;

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options, its requests carrying that job. `reporter` is the host
    /// `llm.LLM` whose `error` events report the attempts this view retries.
    #[host(call)]
    fn identified(
        &self,
        headers: crate::Headers,
        #[host(default = None)] reporter: Option<Arc<dyn Llm>>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> String;

    /// The persistent request options, a copy.
    #[host(getter)]
    fn extra_kwargs(&self) -> Extra;

    /// The scheduling class of its requests.
    #[host(getter)]
    fn inference_class(&self) -> Option<String>;

    /// Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs, and
    /// ``endpoint`` the provider endpoint.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
        #[host(default = None)] endpoint: Option<ProviderEndpoint>,
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
        #[host(default = None)] headers: Option<crate::Headers>,
    ) -> Result<Self::Chunks, ApiError>;
}

/// An OpenAI-compatible provider the client calls directly with the provider's own key, in place
/// of the LiveKit Inference gateway; no LiveKit header is sent.
#[crate::host(record(
    module = "llm_provider_options",
    path = "livekit.agents.inference.llm",
    dataclass,
    frozen,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderEndpoint {
    pub base_url: String,
    pub api_key: String,
    /// Tool schemas in OpenAI's strict form; some OpenAI-compatible providers refuse it.
    #[host(default = true)]
    pub strict_tool_schema: bool,
    /// Gzip-compressed request bodies, for a provider that accepts them.
    #[host(default = false)]
    pub gzip_body: bool,
    /// Msgpack request bodies, for a provider that accepts them.
    #[host(default = false)]
    pub msgpack_body: bool,
}

/// `inference.LLM`: the LiveKit Inference gateway's LLM, or an OpenAI-compatible provider's.
#[crate::host(facade(
    path = "livekit.agents.inference.llm",
    extended,
    name = "LLM",
    base = "llm.LLM",
    native = InferenceLlm,
    legs(chat),
    leg = "_core_native"
))]
pub trait InferenceLlmFacade {
    /// A LiveKit Inference LLM.
    ///
    /// Args:
    ///     model: The model, ``"provider/model"``.
    ///     provider: The provider the gateway routes to; the gateway picks by default.
    ///     base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
    ///     api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
    ///     api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
    ///     inference_class: The scheduling class of its requests.
    ///     extra_kwargs: Request options sent with every completion.
    ///     endpoint: A provider plugin's own endpoint in place of the gateway.
    #[host(
        new,
        not_given,
        annotation(
            model = ".inference_llm.LLMModels | str",
            inference_class = ".inference_models.InferenceClass",
            extra_kwargs = ".llm_provider_options.ChatCompletionOptions | dict[str, Any]"
        ),
        resolve(extra_kwargs = "builtins.dict")
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: String,
        #[host(kw, default = None)] provider: Option<String>,
        #[host(default = None)] base_url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] inference_class: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
        #[host(default = None)] endpoint: Option<ProviderEndpoint>,
    ) -> Self;

    #[host(getter)]
    fn model(&self) -> String;

    /// Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs.
    #[host(
        not_given,
        annotation(
            model = ".inference_llm.LLMModels | str",
            extra_kwargs = ".llm_provider_options.ChatCompletionOptions | dict[str, Any]"
        ),
        resolve(extra_kwargs = "builtins.dict")
    )]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] extra_kwargs: Option<Extra>,
    );
}

/// Accumulates the streamed choices of one OpenAI-compatible completion into chunks, filtering
/// the model's thinking tags: `inference.LLMStream` over a plugin's own client.
#[crate::host(class(name = "LlmChoiceParser"))]
pub trait ChoiceParser: Send + Sized + 'static {
    #[host(new)]
    fn new(model: String) -> Self;

    /// The chunk `choice` (one streamed choice of completion `id`) completes, if any.
    #[host(call)]
    fn parse(&mut self, id: String, choice: serde_json::Value)
    -> Result<Option<LlmChunk>, Invalid>;
}

/// The completion wire rules `inference.LLMStream` applies around a plugin's own client.
#[crate::host(functions)]
pub trait CompletionWire {
    /// The usage chunk of a streamed completion chunk, when it reports usage.
    #[host(function)]
    fn llm_usage_chunk(chunk: serde_json::Value) -> Result<Option<LlmChunk>, Invalid>;
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

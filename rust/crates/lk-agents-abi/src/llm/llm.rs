#![cfg_attr(
    feature = "python",
    allow(
        clippy::disallowed_types,
        reason = "S0: generated host siblings need module scope; exact type sites are ratcheted."
    )
)]

//! The seam is `LLM.chat()` (`llm/llm.py:149`), not `_run()`: the plugin builds its own `LLMStream`,
//! so a host-backed provider keeps retry, metrics and channels in Python.

use crate::errors::ProviderError;
use crate::utils::stream::StreamSource;
use crate::{ApiConnectOptions, ChatContext, Extra, HostRef, LlmChunk};

/// Plain data: reflection and invocation stay Python. The host passes the dict of these keys.
// The omission-only defaults below move to the record declaration with the records family.
#[crate::host(handwritten(extract, host = "dict[str, object]", py))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    /// Defaults to the empty string only when absent; null is invalid.
    pub description: String,
    /// Explicit JSON, including JSON null; a missing key raises.
    pub json_schema: serde_json::Value,
    pub flags: ToolFlags,
    /// Provider-owned wire declaration; absent for a client function. `name` keeps its tool ID.
    pub provider_tool: Option<ProviderToolDeclaration>,
}

/// Native transport ownership of an external tool declaration. Unknown/custom host tools
/// have no native provider or payload; the canonical object remains in the host tool pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderToolDeclaration {
    pub provider: Option<String>,
    pub payload: serde_json::Value,
}

/// A struct, not an enum, so axes don't exclude each other.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolFlags {
    /// Must not fire on the turn its `beta/` task is entered (`dob.py`). Defaults on omission.
    pub ignore_on_enter: bool,
}

#[crate::host(handwritten(extract, emit, host = "livekit.agents.llm.tool_context.ToolChoice", py))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    /// `"none"`. Not `Option::None`, which here means `NOT_GIVEN`.
    Forbidden,
    Named(String),
}

/// Everything `LLM.chat()` takes, as one value `FallbackChain` can hold and replay for recovery
/// (`llm/fallback_adapter.py:174-200`).
///
/// `chat_ctx`/`tools` are Rust's plain data; `chat_ctx_host`/`tools_host` are side-table refs to
/// the canonical host objects a `Py*` provider hands straight to the plugin. Rust cannot rebuild
/// a live `Tool` callable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatRequest {
    pub chat_ctx: ChatContext,
    /// The host's canonical `llm.ChatContext`; a host LLM rebuilds one from `chat_ctx` when absent.
    pub chat_ctx_host: Option<HostRef>,
    pub tools: Vec<ToolSpec>,
    /// The host's live `list[Tool]`; required whenever `tools` is non-empty.
    pub tools_host: Option<HostRef>,
    /// Further pools of host tools the session added on top of the agent's (the IVR navigator's
    /// `send_dtmf_events`); narrowed with `tools_host` to `tools`, first name wins.
    pub tools_host_extra: Vec<HostRef>,
    pub conn_options: ApiConnectOptions,
    /// `NOT_GIVEN` is `None`; the provider's own default applies. R2b: this is not "false".
    pub parallel_tool_calls: Option<bool>,
    pub tool_choice: Option<ToolChoice>,
    /// `extra_kwargs`. JSON, not an opaque host payload -- `_provider_format` puts it on the wire.
    pub extra: Option<Extra>,
    /// LiveKit Inference's per-call priority class (`X-LiveKit-Inference-Priority`); `None` otherwise.
    pub inference_class: Option<String>,
}

impl ToolChoice {
    /// `Forbidden` is `"none"`: `None` means the caller said nothing; `"none"` forbids tools.
    pub fn from_literal(s: &str) -> Option<ToolChoice> {
        match s {
            "auto" => Some(ToolChoice::Auto),
            "required" => Some(ToolChoice::Required),
            "none" => Some(ToolChoice::Forbidden),
            _ => None,
        }
    }

    pub fn to_host_json(&self) -> serde_json::Value {
        match self {
            ToolChoice::Auto => "auto".into(),
            ToolChoice::Required => "required".into(),
            ToolChoice::Forbidden => "none".into(),
            ToolChoice::Named(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }

    pub fn from_host_json(v: &serde_json::Value) -> Option<ToolChoice> {
        match v {
            serde_json::Value::String(s) => ToolChoice::from_literal(s),
            serde_json::Value::Object(o) => o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(|name| ToolChoice::Named(name.to_owned())),
            _ => None,
        }
    }
}

/// The request as a host `LLM._core_chat` reads it: a `_core.ChatRequest` handle over a clone. The
/// host objects cross as their side-table keys (`_core.resolve_host_ref`), and the context as its
/// `ChatContext.to_dict()` form, read only when the request carries no host context of its own.
#[crate::host(class(name = "ChatRequest"))]
pub trait ChatRequestView: Send + Sync + Sized + 'static {
    /// The key of the host's canonical `llm.ChatContext`, when the request came from one.
    #[host(getter)]
    fn chat_ctx_host(&self) -> Option<u64>;
    /// `ChatContext.to_dict()` of the request's context, timestamps kept (images and audio are not
    /// rendered): the context of a request the core minted itself.
    #[host(getter)]
    fn chat_ctx(&self) -> Result<Extra, crate::llm::to_dict::ToDictError>;
    /// The names of the tools this turn offers, in order: the host pool is narrowed to these.
    #[host(getter)]
    fn tool_names(&self) -> Vec<String>;
    /// Canonical declarations of every function offered by this request, including tools
    /// executed by the core that have no callable object in the host pool.
    #[host(getter)]
    fn function_schemas(&self) -> Vec<crate::llm::provider_format::tools::RawFunctionDescription>;
    /// The key of the host's live `list[Tool]` pool.
    #[host(getter)]
    fn tools_host(&self) -> Option<u64>;
    /// The keys of the pools the session added on top of the agent's.
    #[host(getter)]
    fn tools_host_extra(&self) -> Vec<u64>;
    #[host(getter)]
    fn parallel_tool_calls(&self) -> Option<bool>;
    #[host(getter)]
    fn tool_choice(&self) -> Option<serde_json::Value>;
    #[host(getter)]
    fn extra(&self) -> Option<Extra>;
    #[host(getter)]
    fn inference_class(&self) -> Option<String>;
}

impl ChatRequestView for ChatRequest {
    fn chat_ctx_host(&self) -> Option<u64> {
        self.chat_ctx_host.as_ref().map(HostRef::id)
    }

    fn chat_ctx(&self) -> Result<Extra, crate::llm::to_dict::ToDictError> {
        use crate::llm::to_dict::{DumpCx, ModelDump, ToDictOptions};
        let opts = ToDictOptions { exclude_timestamp: false, ..Default::default() };
        let cx = DumpCx { opts: &opts, host: &Default::default() };
        let items = self.chat_ctx.items.model_dump(&cx, "ChatItem")?;
        Ok(Extra::from_iter([("items".to_owned(), items.unwrap_or_default())]))
    }

    fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    fn function_schemas(&self) -> Vec<crate::llm::provider_format::tools::RawFunctionDescription> {
        self.tools
            .iter()
            .filter(|tool| tool.provider_tool.is_none())
            .map(|tool| crate::llm::provider_format::tools::RawFunctionDescription {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                parameters: tool.json_schema.clone(),
            })
            .collect()
    }

    fn tools_host(&self) -> Option<u64> {
        self.tools_host.as_ref().map(HostRef::id)
    }

    fn tools_host_extra(&self) -> Vec<u64> {
        self.tools_host_extra.iter().map(HostRef::id).collect()
    }

    fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }

    fn tool_choice(&self) -> Option<serde_json::Value> {
        self.tool_choice.as_ref().map(ToolChoice::to_host_json)
    }

    fn extra(&self) -> Option<Extra> {
        self.extra.clone()
    }

    fn inference_class(&self) -> Option<String> {
        self.inference_class.clone()
    }
}

// A host `llm.LLM` is this trait through its generated proxy (`LlmProxy`), calling the `_core_*`
// methods of its hand-written base class (`llm.py`); the request crosses as a `_core.ChatRequest`
// handle.
/// A language model.
///
/// Plugins implement `chat()`, returning an `LLMStream` whose `_run` sends the provider's chunks.
#[crate::host(proxy(host = "livekit.agents.llm.LLM", base = "llm.py", generic))]
pub trait Llm: Send + Sync {
    /// The host object's identity, independent of proxy instances and bound connection options;
    /// `None` for a native provider.
    #[host(read_once, name = "_core_instance_id")]
    fn instance_id(&self) -> Option<usize> {
        None
    }

    /// The name logs and metrics give this instance: its class path.
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    // `LLM.model` (`llm/llm.py:120`): the model name, or `"unknown"`. Not derivable from
    // [`label`](Self::label) (`f"{module}.{classname}"`); an implementor that knows its model must
    // override this.
    #[host(read_once, name = "_core_model")]
    fn model(&self) -> String {
        "unknown".to_owned()
    }

    /// The model and provider labelling this instance's turn metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; `"unknown"` for missing names.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId {
            label: self.label(),
            model: self.model(),
            provider: "unknown".to_owned(),
        }
    }

    /// Pre-warm connection to the LLM service.
    ///
    /// Establishes DNS resolution and the TLS connection to the provider before the first
    /// inference request, reducing time-to-first-token on the initial reply. It is called
    /// automatically when a `VoiceSession` is constructed and when an agent activity starts, but
    /// can also be called directly.
    ///
    /// Non-blocking (fire-and-forget) and idempotent. Providers enable it by overriding
    /// `_prewarm_impl`.
    ///
    /// Args:
    /// - loop: Event loop to schedule the prewarm request on. Defaults to the running event loop.
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self) {}

    // A view of this provider whose calls carry `conn` (a host plugin's `conn_options`: the
    // request's own are the chain's). See [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    #[host(bind)]
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Llm>> {
        None
    }

    // The framework's report of a failed attempt of a call it retries: the provider's `error`
    // event, `recoverable` while the call is retried. A wrapper forwards it.
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: ProviderError, _recoverable: bool) {}

    /// Open a completion of `chat_ctx`, its attempts retried per `conn_options`.
    ///
    /// Args:
    /// - chat_ctx: The conversation to complete.
    /// - tools: The tools the model may call.
    /// - conn_options: The timeout and retries of the request.
    /// - parallel_tool_calls: Whether the model may call several tools at once.
    /// - tool_choice: Whether and which tool the model must call.
    /// - extra_kwargs: Provider-specific request options.
    // `Result`: a plugin's `chat()` can raise before any chunk, which `FallbackChain` treats as
    // retryable. `async` for the reason on [`Tts`](crate::Tts).
    #[host(await, name = "_core_chat")]
    fn chat<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> crate::BoxFuture<'a, Result<LlmStream, ProviderError>>;
}

/// The completion's drain half: the same [`StreamSource`] every other leg hands out.
pub type LlmStream = Box<dyn StreamSource<LlmChunk>>;

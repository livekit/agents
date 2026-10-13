//! The seam is `LLM.chat()` (`llm/llm.py:149`), not `_run()`: the plugin builds its own `LLMStream`,
//! so a host-backed provider keeps retry, metrics and channels in Python.

use crate::{ApiConnectOptions, ChatContext, Extra, HostRef, LlmChunk};
use async_trait::async_trait;
use crate::errors::ProviderError;
use crate::utils::stream::StreamSource;

/// Plain data: reflection and invocation stay Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    /// `None` reads as the empty string; an absent key is an error.
    pub description: String,
    /// A missing key raises (a null `parameters` is rejected by every provider); `None` reads as
    /// `{}`.
    pub json_schema: serde_json::Value,
    pub flags: ToolFlags,
}

/// A struct, not an enum, so axes don't exclude each other.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolFlags {
    /// Must not fire on the turn its `beta/` task is entered (`dob.py`). `None` reads as `False`.
    pub ignore_on_enter: bool,
}

/// Every key required.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolSpec {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{item, item_with, or_default, value_from_py};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "ToolSpecFromPy";
        let obj = &*obj;
        Ok(ToolSpec {
            name: item(obj, OWNER, "name")?,
            description: item_with(obj, OWNER, "description", |v| or_default(v))?,
            json_schema: item_with(obj, OWNER, "json_schema", |v| {
                if v.is_none() { Ok(serde_json::Value::Object(serde_json::Map::new())) } else { value_from_py(v, "json_schema") }
            })?,
            flags: ToolFlags { ignore_on_enter: item_with(obj, OWNER, "ignore_on_enter", |v| or_default(v))? },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    /// `"none"`. Not `Option::None`, which here means `NOT_GIVEN`.
    Forbidden,
    Named(String),
}

/// The JSON form: a literal string, or the two-level named-tool dict.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for ToolChoice {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::python::value_to_py(py, &self.to_host_json())
    }
}

/// `None` is not a choice: an `Option<ToolChoice>` reads it as "the provider decides".
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolChoice {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        use pyo3::types::PyAnyMethods as _;
        if obj.is_none() {
            return Err(PyValueError::new_err("tool_choice"));
        }
        if let Ok(name) = obj.extract::<String>() {
            return ToolChoice::from_literal(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown tool_choice {name:?}")));
        }
        // Strict, unlike `from_host_json`: a malformed dict raises the record's own error.
        let d = obj.cast::<pyo3::types::PyDict>().map_err(|_| PyValueError::new_err("tool_choice must be a string or a named-tool dict"))?;
        let named: crate::voice::calls::NamedToolChoice = d.extract()?;
        Ok(ToolChoice::Named(named.function.name))
    }
}

/// Everything `LLM.chat()` takes, as one value `FallbackChain` can hold and replay for recovery
/// (`llm/fallback_adapter.py:174-200`).
///
/// `chat_ctx`/`tools` are Rust's plain data; `chat_ctx_host`/`tools_host` are side-table refs to
/// the canonical host objects a `Py*` provider hands straight to the plugin. Rust cannot rebuild
/// a live `Tool` callable.
#[crate::class]
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
    fn from_literal(s: &str) -> Option<ToolChoice> {
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
            ToolChoice::Named(name) => serde_json::json!({"type": "function", "function": {"name": name}}),
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
#[crate::host(object)]
impl ChatRequest {
    /// The key of the host's canonical `llm.ChatContext`, when the request came from one.
    #[host(getter)]
    pub fn chat_ctx_host(&self) -> Option<u64> {
        self.chat_ctx_host.as_ref().map(HostRef::id)
    }

    /// `ChatContext.to_dict()` of [`chat_ctx`](Self::chat_ctx), timestamps kept (images and audio
    /// are not rendered): the context of a request the core minted itself.
    #[host(getter)]
    pub fn chat_ctx(&self) -> serde_json::Value {
        self.chat_ctx.to_host_dict()
    }

    /// The names of the tools this turn offers, in order: the host pool is narrowed to these.
    #[host(getter)]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    /// The key of the host's live `list[Tool]` pool.
    #[host(getter)]
    pub fn tools_host(&self) -> Option<u64> {
        self.tools_host.as_ref().map(HostRef::id)
    }

    /// The keys of the pools the session added on top of the agent's.
    #[host(getter)]
    pub fn tools_host_extra(&self) -> Vec<u64> {
        self.tools_host_extra.iter().map(HostRef::id).collect()
    }

    #[host(getter)]
    pub fn parallel_tool_calls(&self) -> Option<bool> {
        self.parallel_tool_calls
    }

    #[host(getter)]
    pub fn tool_choice(&self) -> Option<serde_json::Value> {
        self.tool_choice.as_ref().map(ToolChoice::to_host_json)
    }

    #[host(getter)]
    pub fn extra(&self) -> Option<Extra> {
        self.extra.clone()
    }

    #[host(getter)]
    pub fn inference_class(&self) -> Option<String> {
        self.inference_class.clone()
    }
}

/// A language model. A host `llm.LLM` is this trait through its bare `#[host(proxy)]` (`PyLlm`); the
/// request crosses as a `_core.ChatRequest` handle.
#[crate::host(proxy(host = "livekit.agents.llm.LLM", clock))]
#[async_trait]
pub trait Llm: Send + Sync {
    fn label(&self) -> String;

    /// `LLM.model` (`llm/llm.py:120`): the model name, or `"unknown"`.
    ///
    /// Not derivable from [`label`](Self::label) (`f"{module}.{classname}"`); an implementor that
    /// knows its model must override this.
    fn model(&self) -> String {
        "unknown".to_owned()
    }

    /// `label`/`model`/`provider` as metrics and errors stamp them; `"unknown"` for missing names.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId { label: self.label(), model: self.model(), provider: "unknown".to_owned() }
    }

    /// `LLM.prewarm()`: open the connection pool early; a provider with nothing to warm ignores it.
    fn prewarm(&self) {}

    /// A view of this provider whose calls carry `conn` (a host plugin's `conn_options`: the
    /// request's own are the chain's). See [`Stt::with_conn_options`](crate::Stt::with_conn_options).
    fn with_conn_options(&self, _conn: &ApiConnectOptions) -> Option<std::sync::Arc<dyn Llm>> {
        None
    }

    /// Open a completion.
    ///
    /// `Result`: a plugin's `chat()` can raise before any chunk, which `FallbackChain` treats as
    /// retryable. `async` for the reason on [`Tts`](crate::Tts).
    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError>;
}

/// The completion's drain half: the same [`StreamSource`] every other leg hands out. Handed to the
/// host as `InferenceLlmStream`.
#[crate::host(stream(name = "InferenceLlmStream", doc = "One in-flight completion.", output = LlmChunk))]
pub type LlmStream = Box<dyn StreamSource<LlmChunk>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A plugin-shaped provider: a module-path `label` and no model of its own.
    struct Plugin;

    #[async_trait]
    impl Llm for Plugin {
        fn label(&self) -> String {
            "livekit.plugins.openai.llm.LLM".to_owned()
        }
        async fn chat(&self, _req: &ChatRequest) -> Result<LlmStream, ProviderError> {
            unreachable!("this test never opens a completion")
        }
    }

    // Catches: defaulting `model()` to `label()` or `""` instead of `"unknown"` (the judge's
    // `"gpt-5" in llm.model` test would guess wrong).
    #[test]
    fn a_provider_that_does_not_know_its_model_says_unknown_not_its_label() {
        let p = Plugin;
        assert_eq!(p.model(), "unknown");
        assert_ne!(p.model(), p.label());
        assert!(!p.model().contains("gpt-5"), "and the judge's substring test is safe on it");
    }
}

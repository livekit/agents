//! `_core.InferenceLlm` and the `inference.llm` helpers. Only `inference.LLM.chat()` streams through
//! here: two plugins subclass `LLMStream` with their own client, and routing them through the
//! gateway would send their traffic to LiveKit. Chunks cross as `ChatChunk`-shaped dicts.

use std::sync::Arc;

use lk_agents_abi::python::{check_pid, extract_generated, provider_error_to_py, value_from_py, value_to_py};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatContext, ChatItem, ChatRequest, InferenceLlmStream, Llm, LlmStream, ToolChoice, ToolSpec};

use crate::headers::{BoundIdentity, Headers};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use super::{LlmClientConfig, LlmOptions, NativeLlm, parse, request};
use crate::python::{conn_options, json_from_py, register_class, register_fn, session_clock, transport};

/// `inference.LLM`'s Rust half.
///
/// `frozen`: `update_options` mutates through the `Mutex`, so pyo3 emits no borrow flag.
#[pyclass(frozen, name = "InferenceLlm", module = "livekit.agents._core")]
pub struct InferenceLlm {
    inner: Arc<GatewayLlm>,
}

register_class!(InferenceLlm);

// A session holds the client directly: `_core.InferenceLlm` is the core's `Llm`, not a proxy.
lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::HostNative {
        trait_id: || std::any::TypeId::of::<dyn Llm>(),
        read: |obj| {
            let llm = obj.extract::<PyRef<'_, InferenceLlm>>().ok()?;
            let llm: Arc<dyn Llm> = Arc::clone(&llm.inner) as Arc<dyn Llm>;
            Some(Box::new(llm) as Box<dyn std::any::Any>)
        },
    }
}

/// The gateway LLM: a client per request, since the reference overrides the timeout per request.
pub struct GatewayLlm {
    opts: std::sync::Mutex<LlmOptions>,
    images: Arc<lk_agents_abi::llm::provider_format::PreResolved>,
    http: Arc<dyn crate::transport::HttpTransport>,
    /// `RealClock` until a session binds its own.
    pub(crate) clock: Arc<lk_agents_abi::BoundClock>,
    identity: BoundIdentity,
}

impl InferenceLlm {
    pub(crate) fn build(_py: Python<'_>, cfg: LlmClientConfig) -> PyResult<Self> {
        let llm = Self::from_parts(cfg.into(), transport()?);
        // The ambient clock until a session binds its own: on a loop on virtual time, a request
        // times out in that loop's seconds.
        llm.inner.clock.bind(<Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient());
        Ok(llm)
    }

    pub(crate) fn from_parts(opts: LlmOptions, http: Arc<dyn crate::transport::HttpTransport>) -> Self {
        // An empty resolver: the host already serialized every image onto the item.
        let images = Arc::new(lk_agents_abi::llm::provider_format::PreResolved::from_pairs(Vec::<(
            String,
            lk_agents_abi::llm::provider_format::SerializedImage,
        )>::new()));
        let inner = GatewayLlm {
            opts: std::sync::Mutex::new(opts),
            images,
            http,
            clock: lk_agents_abi::BoundClock::new(),
            identity: BoundIdentity::default(),
        };
        InferenceLlm { inner: Arc::new(inner) }
    }

    pub(crate) fn client(&self, timeout: f64, headers: Option<Vec<(String, String)>>) -> NativeLlm {
        self.inner.client(timeout, headers.map(|pairs| pairs.into_iter().collect()))
    }
}

impl GatewayLlm {
    /// `headers` is `get_inference_headers(...)`, read by the host in the caller's job context.
    fn client(&self, timeout: f64, headers: Option<Headers>) -> NativeLlm {
        let opts = self.opts.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let client = NativeLlm::new(opts, self.http.clone(), self.images.clone())
            .with_clock(Arc::clone(&self.clock) as Arc<dyn lk_agents_abi::Clock>)
            .with_conn_options(conn_options(timeout, 0));
        match headers {
            Some(headers) => client.with_identification(headers),
            None => client,
        }
    }
}

#[async_trait::async_trait]
impl Llm for GatewayLlm {
    fn label(&self) -> String {
        "livekit.agents.inference.llm.LLM".into()
    }

    fn model(&self) -> String {
        self.opts.lock().unwrap_or_else(|p| p.into_inner()).model.clone()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId { label: self.label(), model: self.model(), provider: "livekit".into() }
    }

    /// The bound job identity, the request's scheduling class over the client's.
    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        let class = req
            .inference_class
            .clone()
            .or_else(|| self.opts.lock().unwrap_or_else(|p| p.into_inner()).inference_class.clone());
        let headers = self.identity.for_connection().map(|mut headers| {
            if let Some(class) = class {
                headers.set(crate::headers::HEADER_INFERENCE_PRIORITY, class);
            }
            headers
        });
        self.client(req.conn_options.timeout, headers).chat(req).await
    }
}

#[pymethods]
impl InferenceLlm {
    /// Bind the session's clock: every request from here times out in session time.
    fn bind_clock(&self, clock: &Bound<'_, PyAny>) -> PyResult<()> {
        let clock = session_clock(clock, "clock")?;
        check_pid()?;
        self.inner.clock.bind(clock);
        Ok(())
    }

    /// `get_inference_headers()` of the job the client now serves; every later request a session
    /// makes carries them, with a fresh session id.
    fn bind_identification(&self, headers: Vec<(String, String)>) -> PyResult<()> {
        check_pid()?;
        self.inner.identity.bind(headers.into_iter().collect());
        Ok(())
    }

    #[getter]
    fn model(&self) -> String {
        self.inner.opts.lock().unwrap_or_else(|p| p.into_inner()).model.clone()
    }

    /// The persistent kwargs, a copy (`update_options` replaces them).
    #[getter]
    fn extra_kwargs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let extra = self.inner.opts.lock().unwrap_or_else(|p| p.into_inner()).extra_kwargs.clone();
        value_to_py(py, &Value::Object(extra))
    }

    /// `LLM.update_options`: read by the next `chat()`. `extra_kwargs` replaces the dict.
    #[pyo3(signature = (*, model = None, extra_kwargs = None))]
    fn update_options(&self, model: Option<String>, extra_kwargs: Option<crate::extra::ExtraKwargs>) {
        let mut opts = self.inner.opts.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(model) = model {
            opts.model = model;
        }
        if let Some(extra) = extra_kwargs {
            opts.extra_kwargs = extra.0;
        }
    }

    /// Open a completion. Synchronous: the socket opens on the first `recv()`, as the reference's
    /// `chat()` does no I/O. `tools` carries the raw schemas.
    #[pyo3(signature = (
        items,
        *,
        tools = None,
        parallel_tool_calls = None,
        tool_choice = None,
        response_format = None,
        extra_kwargs = None,
        timeout = 10.0,
        headers = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        items: Vec<ChatItem>,
        tools: Option<&Bound<'_, PyList>>,
        parallel_tool_calls: Option<bool>,
        tool_choice: Option<&Bound<'_, PyAny>>,
        response_format: Option<&Bound<'_, PyAny>>,
        extra_kwargs: Option<&Bound<'_, PyAny>>,
        timeout: f64,
        headers: Option<Vec<(String, String)>>,
    ) -> PyResult<InferenceLlmStream> {
        check_pid()?;
        let ctx = ChatContext::new(items);
        let tool_specs = match tools {
            None => Vec::new(),
            Some(list) => list.iter().map(|item| extract_generated::<ToolSpec>(&item)).collect::<PyResult<Vec<_>>>()?,
        };
        let extra_kwargs: crate::extra::ExtraKwargs = match extra_kwargs {
            Some(v) if !v.is_none() => v.extract()?,
            _ => Default::default(),
        };
        let args = request::ChatArgs {
            extra_kwargs: Some(extra_kwargs.0),
            parallel_tool_calls,
            tool_choice: match tool_choice {
                Some(v) if !v.is_none() => Some(v.extract::<ToolChoice>()?),
                _ => None,
            },
            response_format: match json_from_py(response_format)? {
                Value::Null => None,
                other => Some(other),
            },
            inference_class: None,
        };
        // `max_retry` is not taken: the host's `LLMStream` owns the retry loop, and two multiply.
        let stream = self.client(timeout, headers).chat_with(&ctx, &tool_specs, args).map_err(provider_error_to_py)?;
        Ok(InferenceLlmStream::new(stream))
    }
}

/// `inference.llm`'s choice parser: accumulates a streamed completion's choices into chunks,
/// filtering the model's thinking tags.
#[pyclass(name = "LlmChoiceParser", module = "livekit.agents._core")]
pub struct ChoiceParser {
    accumulator: parse::ChoiceAccumulator,
    filter: parse::TagFilter,
}

register_class!(ChoiceParser);

#[pymethods]
impl ChoiceParser {
    #[new]
    fn new(model: &str) -> Self {
        Self { accumulator: Default::default(), filter: parse::TagFilter::for_model(model) }
    }

    fn parse<'py>(&mut self, py: Python<'py>, id: &str, choice: &Bound<'_, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let value = value_from_py(choice, "LLM choice")?;
        let choice = serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.accumulator
            .parse_choice(id, &choice, &mut self.filter)
            .map(|chunk| Ok(chunk.into_pyobject(py)?.into_any()))
            .transpose()
    }
}

#[pyfunction]
fn llm_normalize_tool_choice(extra: &Bound<'_, PyDict>, has_tools: bool) -> PyResult<()> {
    if !request::retain_tool_choice(has_tools) && extra.contains("tool_choice")? {
        extra.del_item("tool_choice")?;
    }
    Ok(())
}

#[pyfunction]
fn llm_usage_chunk<'py>(py: Python<'py>, chunk: &Bound<'_, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let value = value_from_py(chunk, "LLM usage")?;
    let chunk = serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    parse::usage_chunk(&chunk).map(|chunk| Ok(chunk.into_pyobject(py)?.into_any())).transpose()
}

register_fn!(llm_normalize_tool_choice, llm_usage_chunk);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::FakeHttp;
    use lk_agents_abi::errors::ApiErrorKind;
    use lk_agents_abi::{ChatRequest, VirtualClock};
    use std::future::Future;
    use std::time::Duration;

    fn opts() -> LlmOptions {
        LlmOptions {
            model: "openai/gpt-4o".into(),
            provider: None,
            base_url: "https://gw/v1".into(),
            api_key: "k".into(),
            api_secret: "s".into(),
            inference_class: None,
            extra_kwargs: Default::default(),
        }
    }

    fn poll_once<F: Future>(f: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        f.poll(&mut cx)
    }

    // Catches: the binding's `NativeLlm` staying on `RealClock`, `bind_clock` not reaching the
    // per-call client built after it, or an unbound LLM reacting to a clock it was never handed.
    #[tokio::test]
    async fn a_client_times_out_on_the_session_clock_once_bound_to_it() {
        use lk_agents_abi::Llm as _;
        let clock = Arc::new(VirtualClock::new());
        let unbound = InferenceLlm::from_parts(opts(), Arc::new(FakeHttp::hanging()));
        let mut idle = unbound.client(10.0, None).chat(&ChatRequest::default()).await.expect("opens");
        let mut idle = Box::pin(idle.next());
        let http = Arc::new(FakeHttp::hanging());
        let llm = InferenceLlm::from_parts(opts(), http.clone());
        llm.inner.clock.bind(Arc::clone(&clock) as Arc<dyn lk_agents_abi::Clock>);

        let mut stream = llm.client(10.0, None).chat(&ChatRequest::default()).await.expect("opens");
        let mut next = Box::pin(stream.next());
        assert!(poll_once(next.as_mut()).is_pending(), "the request is in flight");
        clock.advance(Duration::from_millis(9_900));
        assert!(poll_once(next.as_mut()).is_pending(), "9.9 s is not the 10 s bound");
        clock.advance(Duration::from_millis(100));
        let err = match poll_once(next.as_mut()) {
            std::task::Poll::Ready(Err(err)) => err,
            std::task::Poll::Ready(Ok(chunk)) => panic!("a chunk from a silent gateway: {chunk:?}"),
            std::task::Poll::Pending => panic!("the bound ran on the wall clock, not the session's"),
        };
        // `APITimeoutError` is what `provider_error_to_py` raises for this kind.
        assert_eq!(err.api_kind(), Some(ApiErrorKind::Timeout));
        assert_eq!(http.requests(), 1);
        assert!(poll_once(idle.as_mut()).is_pending(), "the unbound client's wall clock has not moved");
    }
}

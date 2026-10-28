//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `llm` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.
type FallbackLlmImpl = <crate::Selected as super::Implementations>::FallbackLlm;

/// The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks.
#[pyclass(frozen, name = "FallbackLlm", module = "livekit.agents._core")]
pub struct FallbackLlm(::std::sync::Arc<FallbackLlmImpl>);

#[pymethods]
impl FallbackLlm {
    #[new]
    #[pyo3(signature = (llms, *, attempt_timeout = 5.0, max_retry_per_llm = 0, retry_interval = 0.5, sticky = false, on_availability = None))]
    fn new(llms: Bound<'_, PyAny>, attempt_timeout: f64, max_retry_per_llm: u32, retry_interval: f64, sticky: bool, on_availability: Option<Bound<'_, PyAny>>) -> PyResult<FallbackLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let llms = llms.try_iter().and_then(|items| items.map(|item| { let item = item?; super::llm::LlmProxy::from_host(&item) }).collect::<PyResult<Vec<_>>>())?;
        let on_availability = on_availability.map(|v| super::llm::ChainObserverProxy::from_host(&v)).transpose()?;
        let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::new(llms, attempt_timeout, max_retry_per_llm, retry_interval, sticky, on_availability).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(FallbackLlm(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::active_index(&self.0);
        Ok(out)
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::aclose(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (chat_ctx_host, *, tools, tools_host = None, parallel_tool_calls = None, tool_choice = None, extra_kwargs = None, conn_options = None))]
    fn chat<'py>(&self, py: Python<'py>, chat_ctx_host: ::lk_agents_abi::HostRef, tools: ::std::vec::Vec<::lk_agents_abi::ToolSpec>, tools_host: ::std::option::Option<::lk_agents_abi::HostRef>, parallel_tool_calls: ::std::option::Option<bool>, tool_choice: Option<Bound<'_, PyAny>>, extra_kwargs: Option<Bound<'_, PyAny>>, conn_options: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let tool_choice = tool_choice.map(|v| ::lk_agents_abi::python::value_from_py(&v, "tool_choice")).transpose()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <FallbackLlmImpl as ::lk_agents_abi::llm::fallback::FallbackLlm>::chat(&this, chat_ctx_host, tools, tools_host, parallel_tool_calls, tool_choice, extra_kwargs, conn_options)
                .await.map_err(|e: ::lk_agents_abi::errors::ProviderError| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(FallbackLlmChunks(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))), py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Llm`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Llm>(),
        read: |obj| {
            let this = obj.cast::<FallbackLlm>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Llm> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

/// One stream of `::lk_agents_abi::LlmChunk` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "FallbackLlmChunks", module = "livekit.agents._core")]
pub struct FallbackLlmChunks(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::LlmChunk>);

#[pymethods]
impl FallbackLlmChunks {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.recv(py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.aclose(py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type InferenceLlmImpl = <crate::Selected as super::Implementations>::InferenceLlm;

/// The gateway LLM behind `inference.LLM`. Its options are shared by every view
/// (`identified`); each view's requests carry its own job's identification.
#[pyclass(frozen, name = "InferenceLlm", module = "livekit.agents._core")]
pub struct InferenceLlm(::std::sync::Arc<InferenceLlmImpl>);

#[pymethods]
impl InferenceLlm {
    /// A client of `model` on the gateway at `base_url`. Starts nothing.
    #[new]
    #[pyo3(signature = (model, *, provider = None, base_url, api_key, api_secret, inference_class = None, extra_kwargs = None))]
    fn new(model: ::std::string::String, provider: ::std::option::Option<::std::string::String>, base_url: ::std::string::String, api_key: ::std::string::String, api_secret: ::std::string::String, inference_class: ::std::option::Option<::std::string::String>, extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<InferenceLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::new(model, provider, base_url, api_key, api_secret, inference_class, extra_kwargs).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceLlm(::std::sync::Arc::new(out)))
    }

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options, its requests carrying that job. `reporter` is the host
    /// `llm.LLM` whose `error` events report the attempts this view retries.
    #[pyo3(signature = (headers, reporter = None))]
    fn identified(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, reporter: Option<Bound<'_, PyAny>>) -> PyResult<InferenceLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let reporter = reporter.map(|v| super::llm::LlmProxy::from_host(&v)).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::identified(&self.0, headers, reporter);
        Ok(InferenceLlm(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn model(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::model(&self.0);
        Ok(out)
    }

    /// The persistent request options, a copy.
    #[getter]
    fn extra_kwargs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::extra_kwargs(&self.0);
        Ok(::lk_agents_abi::python::extra_to_py(py, &out)?.into_any())
    }

    /// Read by the next `chat()`. `extra_kwargs` replaces the persistent options.
    #[pyo3(signature = (*, model = None, extra_kwargs = None))]
    fn update_options(&self, model: ::std::option::Option<::std::string::String>, extra_kwargs: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::update_options(&self.0, model, extra_kwargs);
        Ok(())
    }

    /// Opens a completion of `items`; no I/O until the first pull. `headers` identify the
    /// caller's job, read in its context.
    #[pyo3(signature = (items, *, tools, conn_options, parallel_tool_calls = None, tool_choice = None, response_format = None, extra_kwargs = None, headers = None))]
    fn chat(&self, items: ::std::vec::Vec<::lk_agents_abi::ChatItem>, tools: ::std::vec::Vec<::lk_agents_abi::ToolSpec>, conn_options: ::lk_agents_abi::ApiConnectOptions, parallel_tool_calls: ::std::option::Option<bool>, tool_choice: ::std::option::Option<::lk_agents_abi::ToolChoice>, response_format: Option<Bound<'_, PyAny>>, extra_kwargs: Option<Bound<'_, PyAny>>, headers: ::std::option::Option<::std::vec::Vec<(::std::string::String, ::std::string::String)>>) -> PyResult<InferenceLlmChunks> {
        ::lk_agents_abi::python::check_pid()?;
        let response_format = response_format.map(|v| ::lk_agents_abi::python::value_from_py(&v, "response_format")).transpose()?;
        let extra_kwargs = extra_kwargs.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "extra_kwargs")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("extra_kwargs", ": expected a dict"))) }).transpose()?;
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::chat(&self.0, items, tools, conn_options, parallel_tool_calls, tool_choice, response_format, extra_kwargs, headers).map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(InferenceLlmChunks(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out))))
    }
}

// Wherever the host passes a `::lk_agents_abi::Llm`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Llm>(),
        read: |obj| {
            let this = obj.cast::<InferenceLlm>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Llm> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

/// One stream of `::lk_agents_abi::LlmChunk` items: `recv()` resolves to the next, `None` at the end.
#[pyclass(frozen, name = "InferenceLlmChunks", module = "livekit.agents._core")]
pub struct InferenceLlmChunks(::lk_agents_runtime::source::NativeSource<::lk_agents_abi::LlmChunk>);

#[pymethods]
impl InferenceLlmChunks {
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.recv(py, |py, item| ::pyo3::IntoPyObjectExt::into_py_any(item, py))
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        self.0.aclose(py)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type LlmChoiceParserImpl = <crate::Selected as super::Implementations>::ChoiceParser;

/// Accumulates the streamed choices of one OpenAI-compatible completion into chunks, filtering
/// the model's thinking tags: `inference.LLMStream` over a plugin's own client.
#[pyclass(name = "LlmChoiceParser", module = "livekit.agents._core")]
pub struct LlmChoiceParser(LlmChoiceParserImpl);

#[pymethods]
impl LlmChoiceParser {
    #[new]
    #[pyo3(signature = (model))]
    fn new(model: ::std::string::String) -> PyResult<LlmChoiceParser> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <LlmChoiceParserImpl as ::lk_agents_abi::llm::inference::ChoiceParser>::new(model);
        Ok(LlmChoiceParser(out))
    }

    /// The chunk `choice` (one streamed choice of completion `id`) completes, if any.
    #[pyo3(signature = (id, choice))]
    fn parse(&mut self, id: ::std::string::String, choice: Bound<'_, PyAny>) -> PyResult<::std::option::Option<::lk_agents_abi::LlmChunk>> {
        ::lk_agents_abi::python::check_pid()?;
        let choice = ::lk_agents_abi::python::value_from_py(&choice, "choice")?;
        let out = <LlmChoiceParserImpl as ::lk_agents_abi::llm::inference::ChoiceParser>::parse(&mut self.0, id, choice).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/inference.rs.
type CompletionWireImpl = <crate::Selected as super::Implementations>::CompletionWire;

/// The usage chunk of a streamed completion chunk, when it reports usage.
#[pyfunction]
#[pyo3(signature = (chunk))]
pub fn llm_usage_chunk(chunk: Bound<'_, PyAny>) -> PyResult<::std::option::Option<::lk_agents_abi::LlmChunk>> {
    ::lk_agents_abi::python::check_pid()?;
    let chunk = ::lk_agents_abi::python::value_from_py(&chunk, "chunk")?;
    let out = <CompletionWireImpl as ::lk_agents_abi::llm::inference::CompletionWire>::llm_usage_chunk(chunk).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| <::lk_agents_abi::llm::inference::InvalidPayload as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
type McpSessionImpl = <crate::Selected as super::Implementations>::McpSession;

/// An SDK session whose owner cancels and joins independently of request borrowers.
#[pyclass(frozen, name = "McpSession", module = "livekit.agents._core")]
pub struct McpSession(::std::sync::Arc<McpSessionImpl>);

#[pymethods]
impl McpSession {
    #[new]
    #[pyo3(signature = (client_session_timeout_seconds))]
    fn new(client_session_timeout_seconds: f64) -> PyResult<McpSession> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::new(client_session_timeout_seconds).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(McpSession(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn connected(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connected(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (allowed))]
    fn set_allowed_tools(&self, allowed: ::std::vec::Vec<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::set_allowed_tools(&self.0, allowed);
        Ok(())
    }

    fn invalidate_cache(&self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::invalidate_cache(&self.0);
        Ok(())
    }

    #[pyo3(signature = (url, headers, transport, timeout, sse_read_timeout))]
    fn connect_http<'py>(&self, py: Python<'py>, url: ::std::string::String, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, transport: ::std::string::String, timeout: f64, sse_read_timeout: f64) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_http(&this, url, headers, transport, timeout, sse_read_timeout)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (command, args, env, cwd))]
    fn connect_stdio<'py>(&self, py: Python<'py>, command: ::std::string::String, args: ::std::vec::Vec<::std::string::String>, env: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, cwd: ::std::option::Option<::std::string::String>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_stdio(&this, command, args, env, cwd)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[pyo3(signature = (transport, host_context))]
    fn connect_host<'py>(&self, py: Python<'py>, transport: Bound<'_, PyAny>, host_context: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let transport = super::llm::McpTransportProxy::from_host(&transport)?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::connect_host(&this, transport, host_context)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    fn list_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::list_tools(&this, )
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(::lk_agents_abi::python::value_to_py(py, &out)?, py))
        })
    }

    #[pyo3(signature = (name, arguments, progress, host_context))]
    fn call_tool<'py>(&self, py: Python<'py>, name: ::std::string::String, arguments: Bound<'_, PyAny>, progress: Option<Bound<'_, PyAny>>, host_context: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let arguments = match ::lk_agents_abi::python::value_from_py(&arguments, "arguments")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("arguments", ": expected a dict"))) }?;
        let progress = progress.map(|v| super::llm::McpProgressProxy::from_host(&v)).transpose()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::call_tool(&this, name, arguments, progress, host_context)
                .await.map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(::lk_agents_abi::python::value_to_py(py, &out)?, py))
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::close(&this, )
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    #[getter]
    fn cleanup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <McpSessionImpl as ::lk_agents_abi::llm::mcp::McpSession>::cleanup(&self.0);
        Ok(crate::generated::records::mcp_session::mcp_cleanup_to_py(py, out)?)
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.
type McpToolsImpl = <crate::Selected as super::Implementations>::McpTools;

#[pyfunction]
#[pyo3(signature = (config))]
pub fn resolve_mcp_tool_options(config: ::std::option::Option<::lk_agents_abi::llm::mcp::McpToolOptionsPatch>) -> PyResult<::lk_agents_abi::llm::mcp::McpToolOptions> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::resolve_tool_options(config);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (name, description, input_schema, meta))]
pub fn mcp_raw_schema<'py>(py: Python<'py>, name: ::std::string::String, description: ::std::option::Option<::std::string::String>, input_schema: Bound<'_, PyAny>, meta: Option<Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let input_schema = match ::lk_agents_abi::python::value_from_py(&input_schema, "input_schema")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("input_schema", ": expected a dict"))) }?;
    let meta = meta.map(|v| match ::lk_agents_abi::python::value_from_py(&v, "meta")? { ::serde_json::Value::Object(map) => Ok(map), _ => Err(::pyo3::exceptions::PyTypeError::new_err(concat!("meta", ": expected a dict"))) }).transpose()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::raw_schema(name, description, input_schema, meta);
    Ok(::lk_agents_abi::python::value_to_py(py, &out)?)
}

#[pyfunction]
#[pyo3(signature = (reload, toolset_initialized, server_connected))]
pub fn mcp_plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::plan_setup(reload, toolset_initialized, server_connected);
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (url, transport_type))]
pub fn mcp_http_transport(url: ::std::string::String, transport_type: ::std::option::Option<::std::string::String>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::http_transport(url, transport_type).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tool_name, result))]
pub fn mcp_render_result(tool_name: ::std::string::String, result: Bound<'_, PyAny>) -> PyResult<::std::string::String> {
    ::lk_agents_abi::python::check_pid()?;
    let result = ::lk_agents_abi::python::value_from_py(&result, "result")?;
    let out = <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::render_result(tool_name, result).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (result))]
pub fn mcp_check_result(result: Bound<'_, PyAny>) -> PyResult<()> {
    ::lk_agents_abi::python::check_pid()?;
    let result = ::lk_agents_abi::python::value_from_py(&result, "result")?;
    <McpToolsImpl as ::lk_agents_abi::llm::mcp::McpTools>::check_result(result).map_err(|e: ::lk_agents_abi::llm::mcp::McpError| <::lk_agents_abi::llm::mcp::McpError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
type NativeRealtimeModelImpl = <crate::Selected as super::Implementations>::NativeRealtimeModel;

#[pyclass(frozen, name = "NativeRealtimeModel", module = "livekit.agents._core")]
pub struct NativeRealtimeModel(::std::sync::Arc<NativeRealtimeModelImpl>);

#[pymethods]
impl NativeRealtimeModel {
    #[new]
    #[pyo3(signature = (config))]
    fn new(config: Bound<'_, PyAny>) -> PyResult<NativeRealtimeModel> {
        ::lk_agents_abi::python::check_pid()?;
        let config = crate::generated::records::realtime_options::native_realtime_config_from_py(&config)?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::new(config).map_err(|e: ::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError| <::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(NativeRealtimeModel(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn provider(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::provider(&self.0);
        Ok(out)
    }

    #[getter]
    fn model(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::model(&self.0);
        Ok(out)
    }

    #[getter]
    fn label(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::label(&self.0);
        Ok(out)
    }

    fn capabilities(&self) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::capabilities(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (options))]
    fn update_options(&self, options: Bound<'_, PyAny>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let options = crate::generated::records::realtime_options::realtime_options_from_py(&options)?;
        <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::update_options(&self.0, options);
        Ok(())
    }

    #[pyo3(signature = (headers))]
    fn bind_identification(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <NativeRealtimeModelImpl as ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel>::bind_identification(&self.0, headers);
        Ok(())
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<NativeRealtimeModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/realtime_native.rs.
type RealtimeFallbackModelImpl = <crate::Selected as super::Implementations>::RealtimeFallbackModel;

/// The generated facade holds native resources; host provider objects use the shared bridge.
#[pyclass(frozen, name = "RealtimeFallbackModel", module = "livekit.agents._core")]
pub struct RealtimeFallbackModel(::std::sync::Arc<RealtimeFallbackModelImpl>);

#[pymethods]
impl RealtimeFallbackModel {
    #[new]
    #[pyo3(signature = (models, *, cooldown = 10.0, regenerate_on_swap = true, on_availability = None))]
    fn new(models: ::std::vec::Vec<::lk_agents_abi::HostRef>, cooldown: f64, regenerate_on_swap: bool, on_availability: ::std::option::Option<::lk_agents_abi::HostRef>) -> PyResult<RealtimeFallbackModel> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::new(models, cooldown, regenerate_on_swap, on_availability).map_err(|e: ::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError| <::lk_agents_abi::llm::realtime_native::RealtimeConfigurationError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(RealtimeFallbackModel(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn active_index(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::active_index(&self.0);
        Ok(out)
    }

    fn capabilities(&self) -> PyResult<::lk_agents_abi::llm::realtime::RealtimeCapabilities> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::capabilities(&self.0);
        Ok(out)
    }

    #[pyo3(signature = (*, switch_model = false))]
    fn restart<'py>(&self, py: Python<'py>, switch_model: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <RealtimeFallbackModelImpl as ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel>::restart(&this, switch_model)
                .await;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Wherever the host passes a `::lk_agents_abi::Realtime`, an instance of this class is itself.
::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::HostNative {
        trait_id: || ::std::any::TypeId::of::<dyn ::lk_agents_abi::Realtime>(),
        read: |obj| {
            let this = obj.cast::<RealtimeFallbackModel>().ok()?;
            let native: ::std::sync::Arc<dyn ::lk_agents_abi::Realtime> = this.get().0.clone();
            Some(::std::boxed::Box::new(native) as ::std::boxed::Box<dyn ::std::any::Any>)
        },
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/fallback.rs.

/// [`::lk_agents_abi::llm::fallback::ChainObserver`] over a host `livekit.agents._core_view.AvailabilityObserver`.
#[derive(Clone)]
pub struct ChainObserverProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
}

impl ChainObserverProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::fallback::ChainObserver>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::fallback::ChainObserver>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::fallback::ChainObserver>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(ChainObserverProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::fallback::ChainObserver>(|obj| ChainObserverProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::fallback::ChainObserver for ChainObserverProxy {
    fn availability_changed(&self, ev: ::lk_agents_abi::llm::fallback::AvailabilityChanged) {
        let host = ::std::sync::Arc::clone(&self.host);
        let ev_arg = ev;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "ChainObserver._core_availability_changed", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::llm_fallback::availability_changed_to_py(py, ev_arg)?];
            Ok((host.bind(py).getattr("_core_availability_changed")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn all_unavailable(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "ChainObserver._core_all_unavailable", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_all_unavailable")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/llm.rs.

/// [`::lk_agents_abi::Llm`] over a host `livekit.agents.llm.LLM`.
#[derive(Clone)]
pub struct LlmProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
    label: ::std::string::String,
    model: ::std::string::String,
    id: ::lk_agents_abi::ProviderId,
    /// Appended to every awaited call once bound.
    conn: ::std::option::Option<::lk_agents_abi::ApiConnectOptions>,
}

impl LlmProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::Llm>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::Llm>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::Llm>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let label = {
            let v = obj.call_method0("_core_label")?;
            v.extract::<::std::string::String>()?
        };
        let model = {
            let v = obj.call_method0("_core_model")?;
            v.extract::<::std::string::String>()?
        };
        let id = {
            let v = obj.call_method0("_core_id")?;
            v.extract::<::lk_agents_abi::ProviderId>()?
        };
        Ok(LlmProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?), label, model, id, conn: None })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::Llm>(|obj| LlmProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::Llm for LlmProxy {
    fn label(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.label)
    }

    fn model(&self) -> ::std::string::String {
        ::std::clone::Clone::clone(&self.model)
    }

    fn id(&self) -> ::lk_agents_abi::ProviderId {
        ::std::clone::Clone::clone(&self.id)
    }

    fn prewarm(&self) {
        let host = ::std::sync::Arc::clone(&self.host);
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Llm._core_prewarm", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 0] = [];
            Ok((host.bind(py).getattr("_core_prewarm")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    fn with_conn_options(&self, conn: &::lk_agents_abi::ApiConnectOptions) -> ::std::option::Option<::std::sync::Arc<dyn ::lk_agents_abi::Llm>> {
        Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))
    }

    fn failed(&self, error: ::lk_agents_abi::errors::ProviderError, recoverable: bool) {
        let host = ::std::sync::Arc::clone(&self.host);
        let error_arg = error;
        let error_arg = match ::pyo3::Python::attach(|py| -> ::pyo3::PyResult<::pyo3::Py<::pyo3::PyAny>> { Ok((::pyo3::IntoPyObjectExt::into_bound_py_any(error_arg, py)?).unbind()) }) {
            Ok(value) => value,
            Err(error) => {
                ::lk_agents_abi::python::log::warn!("Llm._core_failed failed: {error}");
                return;
            }
        };
        let recoverable_arg = recoverable;
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "Llm._core_failed", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [error_arg.into_bound(py), ::pyo3::IntoPyObjectExt::into_bound_py_any(recoverable_arg, py)?];
            Ok((host.bind(py).getattr("_core_failed")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }

    async fn chat(&self, req: &::lk_agents_abi::ChatRequest) -> ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::LlmChunk>>, ::lk_agents_abi::errors::ProviderError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let req_arg = ::std::clone::Clone::clone(req);
        let bound = ::std::clone::Clone::clone(&self.conn);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(req_arg, py)?];
                let mut args = args.to_vec();
                if let Some(bound) = bound {
                    args.push(py.import("livekit.agents.types")?.getattr("APIConnectOptions")?.call((), Some(&::pyo3::IntoPyObject::into_pyobject(bound, py)?.cast_into::<::pyo3::types::PyDict>()?))?);
                }
                Ok((host.bind(py).getattr("_core_chat")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_runtime::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::LlmChunk>()?), ::lk_agents_runtime::process::host_clock()).map(|s| ::std::boxed::Box::new(s) as ::std::boxed::Box<dyn ::lk_agents_abi::StreamSource<::lk_agents_abi::LlmChunk>>)).map_err(|e| <::lk_agents_abi::errors::ProviderError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

/// [`::lk_agents_abi::llm::mcp::McpTransport`] over a host `livekit.agents.llm.mcp._Streams`.
#[derive(Clone)]
pub struct McpTransportProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl McpTransportProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpTransport>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::mcp::McpTransport>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(McpTransportProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::mcp::McpTransport>(|obj| McpTransportProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::llm::mcp::McpTransport for McpTransportProxy {
    async fn start(&self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("McpTransport._core_start: the call captured no host context"))));
        };
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_start")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn send(&self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation, message: &::serde_json::Value) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let message_arg = ::std::clone::Clone::clone(message);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("McpTransport._core_send: the call captured no host context"))));
        };
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?, ::lk_agents_abi::python::value_to_py(py, &message_arg)?];
                Ok((host.bind(py).getattr("_core_send")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn receive(&self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::std::result::Result<::serde_json::Value, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("McpTransport._core_receive: the call captured no host context"))));
        };
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_receive")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_abi::python::value_from_py(&o, "McpTransport._core_receive")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }

    async fn close(&self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("McpTransport._core_close: the call captured no host context"))));
        };
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?];
                Ok((host.bind(py).getattr("_core_close")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/mcp.rs.

/// [`::lk_agents_abi::llm::mcp::McpProgress`] over a host `livekit.agents.llm.mcp._Progress`.
#[derive(Clone)]
pub struct McpProgressProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl McpProgressProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::mcp::McpProgress>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::mcp::McpProgress>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(McpProgressProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::mcp::McpProgress>(|obj| McpProgressProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::llm::mcp::McpProgress for McpProgressProxy {
    async fn update(&self, invocation: &::lk_agents_abi::llm::mcp::McpInvocation, message: &str) -> ::std::result::Result<(), ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let invocation_arg = ::std::clone::Clone::clone(invocation);
        let message_arg = message.to_owned();
        let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(invocation).cloned() else {
            return Err(::pyo3::Python::attach(|py| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &::pyo3::exceptions::PyRuntimeError::new_err("McpProgress._core_update: the call captured no host context"))));
        };
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 2] = [crate::generated::records::mcp_session::mcp_invocation_to_py(py, invocation_arg)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(message_arg, py)?];
                Ok((host.bind(py).getattr("_core_update")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = ({ let _ = o; ::pyo3::PyResult::Ok(()) }).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/llm/tool_executor.rs.

/// [`::lk_agents_abi::llm::tool_executor::HostToolRunner`] over a host `livekit.agents.voice.tool_executor.ToolRunner`.
#[derive(Clone)]
pub struct HostToolRunnerProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
    /// The `notify` calls, delivered in order.
    notifier: ::lk_agents_runtime::host::python::Notifier,
}

impl HostToolRunnerProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(HostToolRunnerProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())?, notifier: ::lk_agents_runtime::host::python::Notifier::new(::lk_agents_runtime::host::python::capture_context(obj.py())?) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::llm::tool_executor::HostToolRunner>(|obj| HostToolRunnerProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::llm::tool_executor::HostToolRunner for HostToolRunnerProxy {
    fn run(&self, call: &::lk_agents_abi::FunctionCall, speech: &::lk_agents_abi::CoreObject, granted: bool, task: bool) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::llm::tool_executor::ToolReport, ::lk_agents_abi::errors::HostError>> {
        let host = ::std::sync::Arc::clone(&self.host);
        let call_arg = ::std::clone::Clone::clone(call);
        let speech_arg = ::std::clone::Clone::clone(speech);
        let granted_arg = granted;
        let task_arg = task;
        let context = ::std::clone::Clone::clone(&self.context);
        ::std::boxed::Box::pin(async move {
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 4] = [::pyo3::IntoPyObjectExt::into_bound_py_any(call_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(speech_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(granted_arg, py)?, ::pyo3::IntoPyObjectExt::into_bound_py_any(task_arg, py)?];
                Ok((host.bind(py).getattr("_core_run")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::llm::tool_executor::ToolReport>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
        })
    }

    fn cancel(&self, call_id: &str) {
        let host = ::std::sync::Arc::clone(&self.host);
        let call_id_arg = call_id.to_owned();
        let context = ::std::clone::Clone::clone(&self.context);
        self.notifier.notify(context, "HostToolRunner._core_cancel", move |py| {
            let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(call_id_arg, py)?];
            Ok((host.bind(py).getattr("_core_cancel")?, ::pyo3::types::PyTuple::new(py, args)?))
        });
    }
}

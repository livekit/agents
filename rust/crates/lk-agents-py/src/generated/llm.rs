//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `llm` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

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
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::new(model, provider, base_url, api_key, api_secret, inference_class, extra_kwargs).map_err(::lk_agents_abi::errors::api_error_to_py)?;
        Ok(InferenceLlm(::std::sync::Arc::new(out)))
    }

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies
    /// holds it: the same options, its requests carrying that job. `reporter` is the host
    /// `llm.LLM` whose `error` events report the attempts this view retries.
    #[pyo3(signature = (headers, reporter = None))]
    fn identified(&self, headers: ::std::vec::Vec<(::std::string::String, ::std::string::String)>, reporter: Option<Bound<'_, PyAny>>) -> PyResult<InferenceLlm> {
        ::lk_agents_abi::python::check_pid()?;
        let reporter = reporter.map(|v| <dyn ::lk_agents_abi::Llm as ::lk_agents_abi::python::HostProxy>::from_host(&v, None)).transpose()?;
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
        let out = <InferenceLlmImpl as ::lk_agents_abi::llm::inference::InferenceLlm>::chat(&self.0, items, tools, conn_options, parallel_tool_calls, tool_choice, response_format, extra_kwargs, headers).map_err(::lk_agents_abi::errors::api_error_to_py)?;
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
        self.0.recv(py)
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
        let out = <LlmChoiceParserImpl as ::lk_agents_abi::llm::inference::ChoiceParser>::parse(&mut self.0, id, choice).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))?;
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
    let out = <CompletionWireImpl as ::lk_agents_abi::llm::inference::CompletionWire>::llm_usage_chunk(chunk).map_err(|e: ::lk_agents_abi::llm::inference::InvalidPayload| ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(&e)))?;
    Ok(out)
}

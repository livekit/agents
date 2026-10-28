//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `evals` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
type _core_JudgeRunnerImpl = <crate::Selected as super::Implementations>::JudgeRunner;

/// Runs judges on one model. A model the host implements is offered `tools`, the host's live
/// judge tools (`judge_tool_schemas`).
#[pyclass(frozen, name = "_core_JudgeRunner", module = "livekit.agents._core")]
pub struct _core_JudgeRunner(::std::sync::Arc<_core_JudgeRunnerImpl>);

#[pymethods]
impl _core_JudgeRunner {
    #[new]
    #[pyo3(signature = (llm, tools, host))]
    fn new(llm: Option<Bound<'_, PyAny>>, tools: ::lk_agents_abi::HostRef, host: Bound<'_, PyAny>) -> PyResult<_core_JudgeRunner> {
        ::lk_agents_abi::python::check_pid()?;
        let llm = llm.map(|v| super::llm::LlmProxy::from_host(&v)).transpose()?;
        let host = super::evals::EvaluationHostProxy::from_host(&host)?;
        let out = <_core_JudgeRunnerImpl as ::lk_agents_abi::evals::JudgeRunner>::new(llm, tools, host);
        Ok(_core_JudgeRunner(::std::sync::Arc::new(out)))
    }

    /// Runs `judges` concurrently on `chat_ctx`, judged against `reference` when given: their
    /// outcomes in judge order. An unknown built-in judge is refused before any runs.
    #[pyo3(signature = (judges, chat_ctx, reference))]
    fn evaluate<'py>(&self, py: Python<'py>, judges: ::std::vec::Vec<::lk_agents_abi::evals::JudgeSpec>, chat_ctx: ::std::vec::Vec<::lk_agents_abi::ChatItem>, reference: ::std::option::Option<::std::vec::Vec<::lk_agents_abi::ChatItem>>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <_core_JudgeRunnerImpl as ::lk_agents_abi::evals::JudgeRunner>::evaluate(&this, judges, chat_ctx, reference)
                .await.map_err(|e: ::lk_agents_abi::evals::InvalidJudgment| <::lk_agents_abi::evals::InvalidJudgment as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }

    /// Whether `message` fulfills `intent`. A provider failure raises the provider's error.
    #[pyo3(signature = (intent, message))]
    fn judge_intent<'py>(&self, py: Python<'py>, intent: ::std::string::String, message: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        ::lk_agents_abi::python::future_into_py(py, async move {
            let out = <_core_JudgeRunnerImpl as ::lk_agents_abi::evals::JudgeRunner>::judge_intent(&this, intent, message)
                .await.map_err(|e: ::lk_agents_abi::evals::IntentFailure| <::lk_agents_abi::evals::IntentFailure as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
type EvaluationsImpl = <crate::Selected as super::Implementations>::Evaluations;

/// The aggregates of `verdicts`; a verdict other than pass, fail or maybe is refused.
#[pyfunction]
#[pyo3(signature = (verdicts))]
pub fn _core_evaluation_score(verdicts: ::std::vec::Vec<::std::string::String>) -> PyResult<::lk_agents_abi::evals::EvaluationScore> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <EvaluationsImpl as ::lk_agents_abi::evals::Evaluations>::score(verdicts).map_err(|e: ::lk_agents_abi::evals::InvalidJudgment| <::lk_agents_abi::evals::InvalidJudgment as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

/// The judge tools' raw schemas, which the host offers a model it implements.
#[pyfunction]
pub fn _core_judge_tool_schemas<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <EvaluationsImpl as ::lk_agents_abi::evals::Evaluations>::tool_schemas();
    Ok(::lk_agents_abi::python::value_to_py(py, &out)?)
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.

/// [`::lk_agents_abi::evals::EvaluationHost`] over a host `livekit.agents.evals._Evaluation`.
#[derive(Clone)]
pub struct EvaluationHostProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
    /// The host's running loop and a copy of its context, captured when it was taken.
    context: ::lk_agents_abi::utils::host_ref::HostRef,
}

impl EvaluationHostProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::evals::EvaluationHost>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::evals::EvaluationHost>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj)?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::evals::EvaluationHost>> {
        Ok(::std::boxed::Box::new(Self::take(obj)?))
    }

    fn take(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(EvaluationHostProxy { host: ::std::sync::Arc::new(obj.clone().unbind()), context: ::lk_agents_runtime::host::python::capture_context(obj.py())? })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::evals::EvaluationHost>(|obj| EvaluationHostProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

#[::async_trait::async_trait]
impl ::lk_agents_abi::evals::EvaluationHost for EvaluationHostProxy {
    async fn evaluate(&self, evaluator: &::lk_agents_abi::HostRef) -> ::std::result::Result<::lk_agents_abi::evals::Judgment, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let evaluator_arg = ::std::clone::Clone::clone(evaluator);
        let context = ::std::clone::Clone::clone(&self.context);
        let context = &context;
        let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(evaluator_arg, py)?];
                Ok((host.bind(py).getattr("evaluate")?, ::pyo3::types::PyTuple::new(py, args)?))
            }).await;
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (o.extract::<::lk_agents_abi::evals::Judgment>()).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}

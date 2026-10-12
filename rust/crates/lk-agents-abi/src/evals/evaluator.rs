//! The reference's `Evaluator` is a structural Protocol (`evals/evaluation.py:17`): the host adapter
//! is duck-typed on `.name` and `evaluate`, never `isinstance`.

use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::Llm;
use lk_agents_abi::{ChatContext, Clock, HOST_TEARDOWN_TIMEOUT, HostRef, Ts};

use crate::evals::error::JudgeError;
use crate::evals::judgment::JudgmentResult;

/// `evaluate(*, chat_ctx, reference=None, llm=None)`.
#[derive(Clone)]
pub struct EvaluateRequest {
    pub chat_ctx: Arc<ChatContext>,
    pub reference: Option<Arc<ChatContext>>,
    /// Wins over the judge's own (`judge.py:222`).
    pub llm: Option<Arc<JudgeLlm>>,
}

#[async_trait]
pub trait Evaluator: Send + Sync {
    fn name(&self) -> String;

    async fn evaluate(&self, req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError>;
}

/// Plus what [`Llm`] cannot tell: `llm.model` (the gpt-5 temperature rule) and, host-backed, the
/// live `submit_verdict` tool `PyLlm` requires alongside `tools`.
pub struct JudgeLlm {
    llm: Arc<dyn Llm>,
    model: String,
    submit_verdict_host: Option<HostRef>,
    /// The wrapped provider's.
    clock: Arc<dyn Clock>,
    /// The batch class for a LiveKit Inference judge.
    inference_class: Option<String>,
    reasoning_effort: Option<String>,
}

/// The LLM a judge calls, plus the host's live `submit_verdict` tool list. Built by `judge_llm`.
#[crate::host(object(cell = "arc"))]
impl JudgeLlm {
    /// `model` is `LLM.model`, not the label; `submit_verdict_host` is the host's interned
    /// `list[Tool]` for a host-backed judge, `None` for a native one.
    pub fn new(
        llm: Arc<dyn Llm>,
        model: impl Into<String>,
        submit_verdict_host: Option<HostRef>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        JudgeLlm { llm, model: model.into(), submit_verdict_host, clock, inference_class: None, reasoning_effort: None }
    }

    pub fn inference_class(&self) -> Option<&str> {
        self.inference_class.as_deref()
    }

    pub fn reasoning_effort(&self) -> Option<&str> {
        self.reasoning_effort.as_deref()
    }

    /// [`HOST_TEARDOWN_TIMEOUT`] from now on the judge's clock.
    pub fn teardown_deadline(&self) -> Ts {
        self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)
    }

    pub fn llm(&self) -> &Arc<dyn Llm> {
        &self.llm
    }

    #[host(getter)]
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn submit_verdict_host(&self) -> Option<&HostRef> {
        self.submit_verdict_host.as_ref()
    }

    /// Whether the host's `submit_verdict` tool list rides along ([`Self::host_backed`]).
    #[host(getter(name = "host_backed"))]
    pub fn is_host_backed(&self) -> bool {
        self.submit_verdict_host.is_some()
    }

    /// `judge.py:136-139`: `temperature` is sent to everything except gpt-5 (substring test).
    #[host(getter)]
    pub fn supports_temperature(&self) -> bool {
        !self.model.contains("gpt-5")
    }
}

/// `judge.py:239`: the judge never sees two sets of instructions.
pub fn reference_transcript(reference: &ChatContext) -> String {
    let filtered = reference
        .copy(&lk_agents_abi::CopyOptions { exclude_instructions: true, ..Default::default() });
    crate::evals::transcript::format_chat_ctx(&filtered)
}

/// Build the judge LLM from a host `LLM` object.
///
/// `submit_verdict` is the host's `@function_tool` list, interned into the side table; without it
/// a host-backed judge fails on its first request. `model` defaults to the plugin's `LLM.model`.
#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(signature = (llm, model=None, submit_verdict=None, *, inference_class=None, reasoning_effort=None))]
pub fn judge_llm(
    py: pyo3::Python<'_>,
    llm: pyo3::Bound<'_, pyo3::PyAny>,
    model: Option<&str>,
    submit_verdict: Option<pyo3::Bound<'_, pyo3::PyAny>>,
    inference_class: Option<String>,
    reasoning_effort: Option<String>,
) -> pyo3::PyResult<PyJudgeLlm> {
    crate::python::check_pid()?;
    // The clock is shared so the shim and the judge agree on `close` deadlines.
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::RealClock::new());
    crate::python::bind_host_provider(py, &llm, &clock)?;
    let provider = <dyn Llm as crate::python::HostProxy>::from_host(&llm, Some(Arc::clone(&clock)))?;
    let shim = provider.with_conn_options(&crate::evals::JUDGE_CONN_OPTIONS).unwrap_or(provider);
    let model = model.map_or_else(|| shim.model(), str::to_owned);
    let host = submit_verdict.filter(|t| !pyo3::types::PyAnyMethods::is_none(t)).map(|t| crate::python::intern_host(t.unbind()));
    let mut inner = JudgeLlm::new(shim, model, host, clock);
    inner.inference_class = inference_class;
    inner.reasoning_effort = reasoning_effort;
    Ok(PyJudgeLlm { inner: Arc::new(inner) })
}

#[cfg(feature = "python")]
crate::register_host!(fn judge_llm);

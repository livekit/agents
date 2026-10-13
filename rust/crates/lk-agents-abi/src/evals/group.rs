//! `evals/evaluation.py`.

use std::sync::Arc;

#[cfg(feature = "python")]
use lk_agents_host_derive::Host;

use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::ChatContext;
use tokio::sync::oneshot;

use crate::evals::error::JudgeError;
use crate::evals::evaluator::{EvaluateRequest, Evaluator, JudgeLlm};
use crate::evals::judgment::JudgmentResult;

/// `EvaluationResult` (`evaluation.py:36`): the judgments keyed by judge name, plus aggregates.
// In Rust because `Tagger` consumes it. A `Vec` to keep dict insertion order; `insert` replaces a
// repeated name in place.
#[crate::class(mutable)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvaluationResult {
    judgments: Vec<(String, JudgmentResult)>,
}

#[crate::host(object(
    stub = "def insert(self, name: str, judgment: JudgmentResult) -> None: ...",
    stub = "def get(self, name: str) -> JudgmentResult | None: ...",
    stub = "def judgments(self) -> list[tuple[str, JudgmentResult]]: ...",
    stub = "def __len__(self) -> int: ...",
    stub = "def __contains__(self, name: str) -> bool: ...",
    stub_import = "from .._core import JudgmentResult"
))]
impl EvaluationResult {
    #[host(new)]
    pub fn new() -> Self {
        EvaluationResult::default()
    }

    /// Python `dict` assignment semantics: replace in place, else append.
    pub fn insert(&mut self, name: impl Into<String>, judgment: JudgmentResult) {
        let name = name.into();
        match self.judgments.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = judgment,
            None => self.judgments.push((name, judgment)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&JudgmentResult> {
        self.judgments.iter().find(|(n, _)| n == name).map(|(_, j)| j)
    }

    /// `judgments.items()`, in insertion order. This is what `Tagger._evaluation` iterates.
    pub fn judgments(&self) -> impl Iterator<Item = (&str, &JudgmentResult)> {
        self.judgments.iter().map(|(n, j)| (n.as_str(), j))
    }

    pub fn len(&self) -> usize {
        self.judgments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.judgments.is_empty()
    }

    /// Score from 0.0 to 1.0: pass = 1, maybe = 0.5, fail = 0.
    // `0.0` when empty while `all_passed` is vacuously true: both deliberate, as in the reference.
    #[host(method)]
    pub fn score(&self) -> f64 {
        if self.judgments.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .judgments
            .iter()
            .map(|(_, j)| {
                if j.passed() {
                    1.0
                } else if j.uncertain() {
                    0.5
                } else {
                    0.0
                }
            })
            .sum();
        total / self.judgments.len() as f64
    }

    /// All judgments passed. **`maybe` counts as not passed**, so this is the strictest gate.
    #[host(method)]
    pub fn all_passed(&self) -> bool {
        self.judgments.iter().all(|(_, j)| j.passed())
    }

    /// At least one passed. `false` on an empty set — `any()` is not vacuously true.
    #[host(method)]
    pub fn any_passed(&self) -> bool {
        self.judgments.iter().any(|(_, j)| j.passed())
    }

    /// Strictly more than half passed; `maybe` does not count.
    #[host(method)]
    pub fn majority_passed(&self) -> bool {
        if self.judgments.is_empty() {
            return true;
        }
        let passed = self.judgments.iter().filter(|(_, j)| j.passed()).count();
        passed as f64 > self.judgments.len() as f64 / 2.0
    }

    /// Nothing explicitly failed; `maybe` is allowed.
    #[host(method)]
    pub fn none_failed(&self) -> bool {
        !self.judgments.iter().any(|(_, j)| j.failed())
    }
}

impl FromIterator<(String, JudgmentResult)> for EvaluationResult {
    fn from_iter<I: IntoIterator<Item = (String, JudgmentResult)>>(iter: I) -> Self {
        let mut out = EvaluationResult::new();
        for (name, judgment) in iter {
            out.insert(name, judgment);
        }
        out
    }
}

#[derive(Debug)]
pub struct JudgeFailure {
    pub name: String,
    pub error: JudgeError,
}

/// One `JudgeFailure` as the dict `evals/evaluation.py` reads. Three keys are derived from
/// `error` in the binding's `drain_failures`.
#[cfg(feature = "python")]
#[derive(Host)]
#[binding(emit)]
pub struct JudgeFailureView {
    /// The evaluator's name.
    pub name: String,
    /// `error.to_string()` -- what a human reads.
    pub error: String,
    /// The taxonomy word `judge_error_kind` answers.
    pub kind: String,
    /// The original exception, taken out of the binding's error table; `None` if there was none.
    pub exception: Option<crate::errors::CapturedError>,
}

/// Failures are returned rather than logged (`evaluation.py:180`): a host error owns a side-table
/// entry only the binding can release.
#[derive(Debug)]
pub struct EvaluationRun {
    pub result: EvaluationResult,
    pub failures: Vec<JudgeFailure>,
}

/// `evaluation.py:81`. Takes no model string: the LLM's constructor owes `inference_class="low"`
/// and `min_reasoning_effort(model)`.
pub struct JudgeGroup {
    llm: Arc<JudgeLlm>,
    judges: Vec<Arc<dyn Evaluator>>,
    spawner: Arc<dyn Spawner>,
}

impl JudgeGroup {
    pub fn new(
        llm: Arc<JudgeLlm>,
        judges: Vec<Arc<dyn Evaluator>>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        JudgeGroup { llm, judges, spawner }
    }

    pub fn llm(&self) -> &Arc<JudgeLlm> {
        &self.llm
    }

    pub fn judges(&self) -> &[Arc<dyn Evaluator>] {
        &self.judges
    }

    /// Concurrently, collected in judge order. Auto-tagging is the caller's.
    pub async fn evaluate(
        &self,
        chat_ctx: Arc<ChatContext>,
        reference: Option<Arc<ChatContext>>,
    ) -> EvaluationRun {
        let req = EvaluateRequest { chat_ctx, reference, llm: Some(self.llm.clone()) };

        let mut pending = Vec::with_capacity(self.judges.len());
        for judge in &self.judges {
            // Read up front so a task that never reports still has a name.
            let name = judge.name();
            let (tx, rx) = oneshot::channel();
            let judge = judge.clone();
            let req = req.clone();
            let _abort = self.spawner.spawn_unboxed(async move {
                // `pending` holds every receiver until awaited.
                let _ = tx.send(judge.evaluate(&req).await);
            });
            pending.push((name, rx));
        }

        let mut result = EvaluationResult::new();
        let mut failures = Vec::new();
        for (name, rx) in pending {
            match rx.await {
                Ok(Ok(judgment)) => result.insert(name, judgment),
                Ok(Err(error)) => failures.push(JudgeFailure { name, error }),
                // Aborted or runtime shutting down: reported, not a missing judgment.
                Err(_) => failures.push(JudgeFailure {
                    name: name.clone(),
                    error: JudgeError::Evaluator {
                        name,
                        message: "the judge task did not run to completion".to_string(),
                    },
                }),
            }
        }

        EvaluationRun { result, failures }
    }
}

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::{EvaluationResult, JudgeFailure, JudgeGroup};
    use crate::evals::error::JudgeError;
    use crate::evals::judgment::PyJudgmentResult;
    use crate::evals::evaluator::PyJudgeLlm;
    use crate::evals::llm_judge::PyJudge;
    use crate::python::check_pid;

    #[pymethods]
    impl EvaluationResult {
        #[pyo3(name = "insert")]
        fn host_insert(&mut self, name: &str, judgment: &PyJudgmentResult) -> PyResult<()> {
            check_pid()?;
            self.insert(name, judgment.core());
            Ok(())
        }

        #[pyo3(name = "judgments")]
        fn host_judgments(&self) -> PyResult<Vec<(String, PyJudgmentResult)>> {
            check_pid()?;
            Ok(self.judgments().map(|(n, j)| (n.to_owned(), PyJudgmentResult::of(j))).collect())
        }
    }

    /// `JudgeGroup` (`evaluation.py:81`): fan one conversation out over a set of judges.
    #[pyclass(name = "JudgeGroup", frozen, module = "livekit.agents._core")]
    pub struct PyJudgeGroup {
        inner: Arc<JudgeGroup>,
    }

    #[pymethods]
    impl PyJudgeGroup {
        /// The group's `llm` overrides each judge's own.
        #[new]
        fn new(llm: &PyJudgeLlm, judges: Vec<PyRef<'_, PyJudge>>) -> PyResult<Self> {
            check_pid()?;
            let judges = judges.iter().map(|j| j.inner.clone()).collect();
            let spawner = <Arc<dyn crate::Spawner> as crate::python::Ambient>::ambient();
            Ok(PyJudgeGroup { inner: Arc::new(JudgeGroup::new(llm.inner.clone(), judges, spawner)) })
        }

        /// Run every judge against `chat_ctx` and return `(result, failures)`. Failures are dicts
        /// with `name`, `error`, `kind` and `exception`, and are absent from the score, not zeros in it.
        #[pyo3(signature = (chat_ctx, reference=None))]
        fn evaluate<'py>(
            &self,
            py: Python<'py>,
            chat_ctx: Vec<crate::ChatItem>,
            reference: Option<Vec<crate::ChatItem>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            check_pid()?;
            let group = self.inner.clone();
            let ctx = Arc::new(crate::ChatContext::new(chat_ctx));
            let reference = reference.map(|r| Arc::new(crate::ChatContext::new(r)));
            crate::python::future_into_py(py, async move {
                let run = group.evaluate(ctx, reference).await;
                // `Bound` is `!Send`; the future must be `Send`, so both halves leave as `Py<_>`.
                Python::attach(|py| {
                    let failures = drain_failures(py, run.failures)?;
                    let result = Py::new(py, run.result)?;
                    PyResult::Ok((result, failures))
                })
            })
        }
    }

    /// Handing the held exception to the host is its release.
    fn drain_failures(py: Python<'_>, failures: Vec<JudgeFailure>) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::with_capacity(failures.len());
        for f in failures {
            let exception = match &f.error {
                JudgeError::Provider(e) => e.host_ref(),
                _ => None,
            };
            let view = super::JudgeFailureView {
                name: f.name.clone(),
                error: f.error.to_string(),
                kind: f.error.kind().to_owned(),
                exception,
            };
            out.push(view.into_pyobject(py)?.unbind());
        }
        Ok(out)
    }

    crate::register_host!(class PyJudgeGroup);
}

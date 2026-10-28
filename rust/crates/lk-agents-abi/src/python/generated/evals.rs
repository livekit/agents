//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(record)]` declarations of `lk-agents-abi`.
//!
//! The `evals` records.

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
static JUDGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::evals::Judgment {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if !obj.is_instance(JUDGMENT.import(py, "livekit.agents._generated.evals", "Judgment")?)? {
            return Err(super::not_a(obj, "Judgment"));
        }
        Ok(Self {
            verdict: ::lk_agents_abi::python::attr_with(obj, "Judgment", "verdict", |v| v.extract::<::std::string::String>())?,
            reasoning: ::lk_agents_abi::python::attr_with(obj, "Judgment", "reasoning", |v| v.extract::<::std::string::String>())?,
            instructions: ::lk_agents_abi::python::attr_with(obj, "Judgment", "instructions", |v| v.extract::<::std::string::String>())?,
        })
    }
}

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::evals::Judgment {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { verdict, reasoning, instructions } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("verdict", ::pyo3::IntoPyObjectExt::into_bound_py_any(verdict, py)?)?;
        kwargs.set_item("reasoning", ::pyo3::IntoPyObjectExt::into_bound_py_any(reasoning, py)?)?;
        kwargs.set_item("instructions", ::pyo3::IntoPyObjectExt::into_bound_py_any(instructions, py)?)?;
        JUDGMENT.import(py, "livekit.agents._generated.evals", "Judgment")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
static JUDGE_SPEC_BUILT_IN: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static JUDGE_SPEC_HOST: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'a, 'py> FromPyObject<'a, 'py> for ::lk_agents_abi::evals::JudgeSpec {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let py = obj.py();
        if obj.is_instance(JUDGE_SPEC_BUILT_IN.import(py, "livekit.agents._generated.evals", "JudgeSpecBuiltIn")?)? {
            return Ok(Self::BuiltIn { name: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecBuiltIn", "name", |v| v.extract::<::std::string::String>())? });
        }
        if obj.is_instance(JUDGE_SPEC_HOST.import(py, "livekit.agents._generated.evals", "JudgeSpecHost")?)? {
            return Ok(Self::Host { name: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecHost", "name", |v| v.extract::<::std::string::String>())?, evaluator: ::lk_agents_abi::python::attr_with(obj, "JudgeSpecHost", "evaluator", |v| v.extract::<::lk_agents_abi::HostRef>().map_err(::std::convert::Into::<PyErr>::into))? });
        }
        Err(super::not_a(obj, "JudgeSpecBuiltIn | JudgeSpecHost"))
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
static JUDGE_OUTCOME_JUDGED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();
static JUDGE_OUTCOME_FAILED: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::evals::JudgeOutcome {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Judged { name, judgment } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
                kwargs.set_item("judgment", ::pyo3::IntoPyObjectExt::into_bound_py_any(judgment, py)?)?;
                JUDGE_OUTCOME_JUDGED.import(py, "livekit.agents._generated.evals", "JudgeOutcomeJudged")?.call((), Some(&kwargs))
            },
            Self::Failed { name, error } => {
                let kwargs = ::pyo3::types::PyDict::new(py);
                kwargs.set_item("name", ::pyo3::IntoPyObjectExt::into_bound_py_any(name, py)?)?;
                kwargs.set_item("error", ::pyo3::IntoPyObjectExt::into_bound_py_any(error, py)?)?;
                JUDGE_OUTCOME_FAILED.import(py, "livekit.agents._generated.evals", "JudgeOutcomeFailed")?.call((), Some(&kwargs))
            },
        }
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
static INTENT_JUDGMENT: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::evals::IntentJudgment {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { success, reason, usage } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("success", ::pyo3::IntoPyObjectExt::into_bound_py_any(success, py)?)?;
        kwargs.set_item("reason", ::pyo3::IntoPyObjectExt::into_bound_py_any(reason, py)?)?;
        kwargs.set_item("usage", match usage { None => py.None().into_bound(py), Some(x) => ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)? })?;
        INTENT_JUDGMENT.import(py, "livekit.agents._generated.evals", "IntentJudgment")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
static EVALUATION_SCORE: ::pyo3::sync::PyOnceLock<Py<::pyo3::types::PyType>> =
    ::pyo3::sync::PyOnceLock::new();

impl<'py> IntoPyObject<'py> for ::lk_agents_abi::evals::EvaluationScore {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Self { score, all_passed, any_passed, majority_passed, none_failed } = self;
        let kwargs = ::pyo3::types::PyDict::new(py);
        kwargs.set_item("score", ::pyo3::IntoPyObjectExt::into_bound_py_any(score, py)?)?;
        kwargs.set_item("all_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(all_passed, py)?)?;
        kwargs.set_item("any_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(any_passed, py)?)?;
        kwargs.set_item("majority_passed", ::pyo3::IntoPyObjectExt::into_bound_py_any(majority_passed, py)?)?;
        kwargs.set_item("none_failed", ::pyo3::IntoPyObjectExt::into_bound_py_any(none_failed, py)?)?;
        EVALUATION_SCORE.import(py, "livekit.agents._generated.evals", "EvaluationScore")?.call((), Some(&kwargs))
    }
}

// Declared in rust/crates/lk-agents-abi/src/evals.rs.
impl crate::python::HostProxy for dyn ::lk_agents_abi::evals::EvaluationHost {
    fn from_host(obj: &::pyo3::Bound<'_, ::pyo3::PyAny>, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::sync::Arc<Self>> {
        crate::python::declared_proxy::<dyn ::lk_agents_abi::evals::EvaluationHost>(obj)
    }

    fn view(handle: crate::python::HostHandle, _clock: ::std::option::Option<::std::sync::Arc<dyn crate::Clock>>) -> ::pyo3::PyResult<::std::boxed::Box<Self>> {
        let _ = handle;
        Err(::pyo3::exceptions::PyTypeError::new_err("`EvaluationHost` is a declared proxy: it is taken from the host object, not a view"))
    }
}

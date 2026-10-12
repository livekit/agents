//! `evals/judge.py:14`.

/// Three states: every aggregate treats `maybe` differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    Pass,
    Fail,
    /// `"maybe"` on the wire; the accessor is `uncertain`, as in the reference.
    Maybe,
}

impl Verdict {
    /// The `lk.judge.<name>` tag value and the model's argument.
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Maybe => "maybe",
        }
    }

    /// Strict (`"PASS"` is rejected, as pydantic's `Literal` does).
    pub fn from_literal(s: &str) -> Option<Verdict> {
        match s {
            "pass" => Some(Verdict::Pass),
            "fail" => Some(Verdict::Fail),
            "maybe" => Some(Verdict::Maybe),
            _ => None,
        }
    }
}

/// `judge.py:33`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgmentResult {
    pub verdict: Verdict,
    pub reasoning: String,
    /// Filled in by the judge after the LLM call (`judge.py:250`).
    pub instructions: String,
}

impl JudgmentResult {
    /// `instructions` defaults to `""`.
    pub fn new(verdict: Verdict, reasoning: impl Into<String>) -> Self {
        JudgmentResult { verdict, reasoning: reasoning.into(), instructions: String::new() }
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// **`maybe` is not a pass.**
    pub fn passed(&self) -> bool {
        self.verdict == Verdict::Pass
    }

    /// `maybe` is not a fail: this is not `!passed()`.
    pub fn failed(&self) -> bool {
        self.verdict == Verdict::Fail
    }

    pub fn uncertain(&self) -> bool {
        self.verdict == Verdict::Maybe
    }
}

#[cfg(feature = "python")]
pub use python::{PyJudgmentResult, PyVerdict};

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{JudgmentResult, Verdict};

    /// `Verdict` (`judge.py`): what one judge decided.
    #[pyclass(name = "Verdict", eq, eq_int, frozen, from_py_object, module = "livekit.agents._core")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PyVerdict {
        Pass,
        Fail,
        Maybe,
    }

    impl PyVerdict {
        pub fn core(self) -> Verdict {
            match self {
                PyVerdict::Pass => Verdict::Pass,
                PyVerdict::Fail => Verdict::Fail,
                PyVerdict::Maybe => Verdict::Maybe,
            }
        }

        pub fn of(v: Verdict) -> Self {
            match v {
                Verdict::Pass => PyVerdict::Pass,
                Verdict::Fail => PyVerdict::Fail,
                Verdict::Maybe => PyVerdict::Maybe,
            }
        }
    }

    #[pymethods]
    impl PyVerdict {
        /// `"pass" | "fail" | "maybe"`.
        fn as_str(&self) -> &'static str {
            self.core().as_str()
        }

        /// The inverse, strict: no case folding or trimming, so a judge ignoring the schema fails.
        #[staticmethod]
        fn from_literal(s: &str) -> Option<PyVerdict> {
            Verdict::from_literal(s).map(PyVerdict::of)
        }
    }

    /// One judgment: the verdict, the model's reasoning, and the instructions it was judging against.
    #[pyclass(name = "JudgmentResult", frozen, get_all, from_py_object, module = "livekit.agents._core")]
    #[derive(Clone)]
    pub struct PyJudgmentResult {
        pub verdict: PyVerdict,
        pub reasoning: String,
        pub instructions: String,
    }

    impl PyJudgmentResult {
        pub fn of(r: &JudgmentResult) -> Self {
            PyJudgmentResult {
                verdict: PyVerdict::of(r.verdict),
                reasoning: r.reasoning.clone(),
                instructions: r.instructions.clone(),
            }
        }

        pub fn core(&self) -> JudgmentResult {
            JudgmentResult::new(self.verdict.core(), self.reasoning.clone()).with_instructions(self.instructions.clone())
        }
    }

    #[pymethods]
    impl PyJudgmentResult {
        #[new]
        #[pyo3(signature = (verdict, reasoning, instructions=String::new()))]
        fn new(verdict: PyVerdict, reasoning: String, instructions: String) -> Self {
            PyJudgmentResult { verdict, reasoning, instructions }
        }
    }

    crate::register_host!(class PyVerdict);
    crate::register_host!(class PyJudgmentResult);
}

#[cfg(test)]
mod tests {
    use super::Verdict;

    // Catches: case-folding or trimming the verdict literal.
    #[test]
    fn the_verdict_literal_is_strict() {
        assert_eq!(Verdict::from_literal("pass"), Some(Verdict::Pass));
        assert_eq!(Verdict::from_literal("Pass"), None);
        assert_eq!(Verdict::from_literal(" pass "), None);
    }
}

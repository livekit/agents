//! `Maybe<T>`, instead of an unreadable `Option<Option<T>>`. Unlike the binding's `Maybe`, `None`
//! here is a value: `Agent(llm=None)` means "no LLM", not "unchanged".

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Maybe<T> {
    #[default]
    NotGiven,
    /// For `Maybe<Option<T>>` an inner `None` is a *given* `None`.
    Given(T),
}

impl<T> Maybe<T> {
    pub fn is_given(&self) -> bool {
        matches!(self, Maybe::Given(_))
    }

    /// Not `into_option`, which would reintroduce the ambiguity.
    pub fn given(self) -> Option<T> {
        match self {
            Maybe::NotGiven => None,
            Maybe::Given(v) => Some(v),
        }
    }

    pub fn unwrap_or(self, default: T) -> T {
        self.given().unwrap_or(default)
    }
}

impl<T> std::fmt::Debug for Maybe<Option<T>> {
    /// `Arc<dyn Trait>` is not `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Maybe::NotGiven => "not-given",
            Maybe::Given(Some(_)) => "given",
            Maybe::Given(None) => "given-none",
        })
    }
}

/// `NOT_GIVEN` is `NotGiven`; `None` is a *given* `None` when `T` reads one (`Maybe<Option<U>>`),
/// else `NotGiven`. An in-place parameter of this type defaults to `NotGiven`.
#[cfg(feature = "python")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for Maybe<T>
where
    T: pyo3::FromPyObject<'a, 'py>,
    T::Error: Into<pyo3::PyErr>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> Result<Self, pyo3::PyErr> {
        use pyo3::types::PyAnyMethods as _;
        if ob.is_instance(crate::python::not_given_type(ob.py())?)? {
            return Ok(Maybe::NotGiven);
        }
        if ob.is_none() {
            return Ok(T::extract(ob).map_or(Maybe::NotGiven, Maybe::Given));
        }
        T::extract(ob).map(Maybe::Given).map_err(Into::into)
    }
}

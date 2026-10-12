//! The host-given option bags every gateway client takes: `extra_kwargs` (a JSON object passed to
//! the provider as-is) and a server-side `fallback` chain.

use serde_json::{Map, Value};

/// A provider's `extra_kwargs`: the JSON object merged into its request unchanged. On the host it
/// is a `dict`; anything else is refused rather than dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtraKwargs(pub Map<String, Value>);

impl std::ops::Deref for ExtraKwargs {
    type Target = Map<String, Value>;
    fn deref(&self) -> &Map<String, Value> {
        &self.0
    }
}

impl From<Map<String, Value>> for ExtraKwargs {
    fn from(map: Map<String, Value>) -> Self {
        ExtraKwargs(map)
    }
}

/// A gateway client's server-side `fallback=` entries (`FallbackModel` dicts), in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fallbacks<T>(pub Vec<T>);

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    use super::*;

    /// A `dict` of JSON values (`None` is the parameter's own business: an `Option` reads it).
    impl<'a, 'py> FromPyObject<'a, 'py> for ExtraKwargs {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            match lk_agents_abi::python::value_from_py(&obj, "inference")? {
                Value::Null => Ok(ExtraKwargs::default()),
                Value::Object(map) => Ok(ExtraKwargs(map)),
                other => Err(PyValueError::new_err(format!("expected a dict of extra kwargs, got {other}"))),
            }
        }
    }

    /// A `list` of entry dicts, each read by the entry's own reading.
    impl<'a, 'py, T> FromPyObject<'a, 'py> for Fallbacks<T>
    where
        T: for<'b> FromPyObject<'b, 'py, Error = PyErr>,
    {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let list = obj.cast::<PyList>()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                // The cast stays for its message, which names the argument.
                let d = item.cast::<PyDict>().map_err(|_| PyValueError::new_err("each fallback entry must be a dict"))?;
                out.push(d.extract::<T>()?);
            }
            Ok(Fallbacks(out))
        }
    }
}

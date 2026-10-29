//! The storage helpers `turn.rs` crosses its turn records with: the pydantic models
//! `lk-agents-codegen` writes from the A2A crate's `#[derive(Host)]` records
//! (`livekit.agents._generated.storage_turn`).

use lk_agents_abi::python::value_from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::de::DeserializeOwned;

/// A storage failure as the `livekit.agents.storage` exception the ABI declares for it.
/// The `livekit.agents.storage` exception a storage failure raises as.
pub(crate) fn to_py_err(error: lk_agents_abi::storage::StorageError) -> PyErr {
    lk_agents_abi::python::HostException::to_py_err(&error)
}

/// A request into the crate's serde type: a pydantic model (its JSON dump) or a dict in the
/// model's shape; a malformed one is a `ValueError` naming the field.
pub(crate) fn parse_request<T: DeserializeOwned>(
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<T> {
    let dumped;
    let obj = if obj.hasattr("model_dump")? {
        let kwargs = PyDict::new(obj.py());
        kwargs.set_item("mode", "json")?;
        dumped = obj.call_method("model_dump", (), Some(&kwargs))?;
        &dumped
    } else {
        obj
    };
    let v = value_from_py(obj, what)?;
    serde_json::from_value(v).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

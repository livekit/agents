//! GENERATED FILE -- DO NOT EDIT.
use pyo3::prelude::*;
#[pyclass(eq, eq_int, from_py_object, module = "livekit.agents._core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Worker,
    Job,
}
pub fn role_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::lk_agents_abi::worker::ipc_host::ProcessRole> {
    Ok(match obj.extract::<Role>()? {
        Role::Worker => ::lk_agents_abi::worker::ipc_host::ProcessRole::Worker,
        Role::Job => ::lk_agents_abi::worker::ipc_host::ProcessRole::Job,
    })
}
pub fn role_to_py(
    py: Python<'_>,
    value: ::lk_agents_abi::worker::ipc_host::ProcessRole,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(Bound::new(
        py,
        match value {
            ::lk_agents_abi::worker::ipc_host::ProcessRole::Worker => Role::Worker,
            ::lk_agents_abi::worker::ipc_host::ProcessRole::Job => Role::Job,
        },
    )?
    .into_any())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Role>()?;
    Ok(())
}

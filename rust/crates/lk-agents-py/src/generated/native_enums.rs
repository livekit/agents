//! GENERATED FILE -- DO NOT EDIT.
use pyo3::prelude::*;
#[pyclass(eq, eq_int, from_py_object, module = "livekit.agents._core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    Unset,
    Ok,
    Error,
}
pub fn status_code_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::lk_agents_abi::telemetry::host::StatusCode> {
    Ok(match obj.extract::<StatusCode>()? {
        StatusCode::Unset => ::lk_agents_abi::telemetry::host::StatusCode::Unset,
        StatusCode::Ok => ::lk_agents_abi::telemetry::host::StatusCode::Ok,
        StatusCode::Error => ::lk_agents_abi::telemetry::host::StatusCode::Error,
    })
}
pub fn status_code_to_py(
    py: Python<'_>,
    value: ::lk_agents_abi::telemetry::host::StatusCode,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(Bound::new(
        py,
        match value {
            ::lk_agents_abi::telemetry::host::StatusCode::Unset => StatusCode::Unset,
            ::lk_agents_abi::telemetry::host::StatusCode::Ok => StatusCode::Ok,
            ::lk_agents_abi::telemetry::host::StatusCode::Error => StatusCode::Error,
        },
    )?
    .into_any())
}
#[pyclass(eq, eq_int, from_py_object, module = "livekit.agents._core")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Internal,
    Client,
    Server,
    Producer,
    Consumer,
}
pub fn span_kind_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<::lk_agents_abi::telemetry::decl::SpanKind> {
    Ok(match obj.extract::<SpanKind>()? {
        SpanKind::Internal => ::lk_agents_abi::telemetry::decl::SpanKind::Internal,
        SpanKind::Client => ::lk_agents_abi::telemetry::decl::SpanKind::Client,
        SpanKind::Server => ::lk_agents_abi::telemetry::decl::SpanKind::Server,
        SpanKind::Producer => ::lk_agents_abi::telemetry::decl::SpanKind::Producer,
        SpanKind::Consumer => ::lk_agents_abi::telemetry::decl::SpanKind::Consumer,
    })
}
pub fn span_kind_to_py(
    py: Python<'_>,
    value: ::lk_agents_abi::telemetry::decl::SpanKind,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(Bound::new(
        py,
        match value {
            ::lk_agents_abi::telemetry::decl::SpanKind::Internal => SpanKind::Internal,
            ::lk_agents_abi::telemetry::decl::SpanKind::Client => SpanKind::Client,
            ::lk_agents_abi::telemetry::decl::SpanKind::Server => SpanKind::Server,
            ::lk_agents_abi::telemetry::decl::SpanKind::Producer => SpanKind::Producer,
            ::lk_agents_abi::telemetry::decl::SpanKind::Consumer => SpanKind::Consumer,
        },
    )?
    .into_any())
}
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
    m.add_class::<StatusCode>()?;
    m.add_class::<SpanKind>()?;
    m.add_class::<Role>()?;
    Ok(())
}

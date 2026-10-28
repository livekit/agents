//! Host log-record conversion and development CLI entrypoints.
use crate::worker::cli::{LogFields, RESERVED};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

/// A record's attribute as JSON, as the reference's encoder writes it: a date by its
/// `isoformat()`, anything else not JSON by its `str()`.
fn lenient(obj: &Bound<'_, PyAny>) -> Value {
    if obj.is_none() {
        return Value::Null;
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Value::Bool(b.is_true());
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return i.into();
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        if let Some(n) = obj.extract::<f64>().ok().and_then(serde_json::Number::from_f64) {
            return Value::Number(n);
        }
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Value::String(s.to_string());
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return Value::Object(d.iter().map(|(k, v)| (key(&k), lenient(&v))).collect());
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        if let Ok(items) = obj.try_iter() {
            return Value::Array(items.flatten().map(|i| lenient(&i)).collect());
        }
    }
    if let Ok(iso) = crate::host::read(obj, "isoformat") {
        return Value::String(key(&iso));
    }
    obj.str().map(|s| Value::String(s.to_string())).unwrap_or(Value::Null)
}

fn key(obj: &Bound<'_, PyAny>) -> String {
    obj.str().map(|s| s.to_string()).unwrap_or_default()
}

/// Convert a host `logging.LogRecord` at the Python boundary.
fn fields(record: &Bound<'_, PyAny>) -> PyResult<LogFields> {
    let msg = record.getattr("msg")?;
    let mut extra = Map::new();
    for (k, v) in record.getattr("__dict__")?.cast::<PyDict>()?.iter() {
        let k = key(&k);
        if !RESERVED.contains(&k.as_str()) && !k.starts_with('_') {
            extra.insert(k, lenient(&v));
        }
    }
    Ok(LogFields {
        level: record.getattr("levelname")?.extract()?,
        name: record.getattr("name")?.extract()?,
        pid: record.getattr("process")?.extract()?,
        created: record.getattr("created")?.extract()?,
        msg: msg.cast::<PyDict>().ok().map(|d| match lenient(d.as_any()) {
            Value::Object(map) => map,
            _ => Map::new(),
        }),
        extra,
    })
}

pub struct WorkerCli;

#[async_trait::async_trait]
impl lk_agents_abi::worker::cli::WorkerCli for WorkerCli {
    fn parse_cli(
        argv: Vec<String>,
    ) -> Result<lk_agents_abi::worker::cli::CliArgs, lk_agents_abi::worker::cli::CliError> {
        crate::worker::cli::parse_cli(argv)
    }

    fn entrypoint_module(
        path: Option<String>,
    ) -> Result<(String, String), lk_agents_abi::worker::cli::CliError> {
        crate::worker::cli::entrypoint_module(path)
    }

    fn format_log_record(
        style: String,
        record: lk_agents_abi::HostRef,
        message: String,
        asctime: String,
        exception: Option<String>,
        stack: Option<String>,
        ensure_ascii: bool,
    ) -> Result<String, lk_agents_abi::errors::HostError> {
        Python::attach(|py| {
            let record = lk_agents_abi::python::resolve_key(py, record.id())?;
            Ok(crate::worker::cli::format_log_record(
                &style,
                fields(record.bind(py))?,
                message,
                asctime,
                exception,
                stack,
                ensure_ascii,
            ))
        })
        .map_err(|error| Python::attach(|py| lk_agents_abi::python::capture_host(py, &error)))
    }

    async fn dev_channel(
        cli_addr: String,
        agent_name: String,
        url: String,
        api_key: String,
        api_secret: String,
        server: std::sync::Arc<dyn lk_agents_abi::worker::cli::DevServer>,
    ) {
        crate::worker::cli::dev_channel(cli_addr, agent_name, url, api_key, api_secret, server)
            .await;
    }

    fn ipc_pair() -> Result<(i64, i64), lk_agents_abi::errors::HostError> {
        crate::worker::ipc::process::ipc_pair().map_err(|error| error.to_string().into())
    }
}

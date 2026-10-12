//! `_core.Worker` and `_core.EndpointStream`: the worker as Python drives it (plan 13.4).
//!
//! `Worker(options, host)` owns the `/agent` connection; `host.open(kind, args)` is the one
//! callback Rust makes, in the generator's shape: it returns an awaitable, scheduled on the event
//! loop that constructed the worker, and cancelled when Rust drops the operation. The kinds:
//!
//! | kind              | args                                   | result                      |
//! |-------------------|----------------------------------------|-----------------------------|
//! | `"job_request"`   | a `JobRequest` dict                    | a decision dict             |
//! | `"launch_job"`    | a `RunningJobInfo` dict                | the LiveKit job id          |
//! | `"terminate_job"` | `{"job_id"}`                           | ignored                     |
//! | `"load"`          | `{}`                                   | a float, or `None` for CPU  |
//! | `"endpoint"`      | the preamble fields and an `EndpointStream` | ignored; serve to completion |
//!
//! Python owns process supervision, signals and the ASGI bridge; Rust owns every socket.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::RealClock;
use lk_agents_abi::python::{
    CancelPythonTask, capture_host, check_pid, pyo3_async_runtimes, schedule_coroutine,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_async_runtimes::TaskLocals;

use crate::transport::{AgentEndpointKind, EndpointReader, EndpointWriter, HttpStreamResetCode};
use crate::{
    EndpointRequest, EnvOption, HEARTBEAT_INTERVAL, HostError, JobDecision, JobId, LOG,
    ServerType, TransportConfig, Worker, WorkerCommand, WorkerConfig, WorkerHandle, WorkerHost,
    WorkerNotification, WorkerOptions, WorkerTransport, protocol,
};

/// Seconds as a `Duration`; negative or non-finite is `None`.
fn secs(v: Option<f64>) -> Option<Duration> {
    match v {
        Some(s) if s.is_finite() && s >= 0.0 => Some(Duration::from_secs_f64(s)),
        _ => None,
    }
}

// --------------------------------------------------------------------- EndpointStream

type Half<T> = Arc<tokio::sync::Mutex<Option<T>>>;

/// One exchange stream after its preamble, as Python reads and writes it. The two directions are
/// independent: a `read` and a `write` may be in flight at once. Dropping it without `finish`
/// resets with code 0 (`HSR_ABORT`).
#[pyclass(name = "EndpointStream", module = "livekit.agents._core")]
pub struct PyEndpointStream {
    reader: Half<EndpointReader>,
    writer: Half<EndpointWriter>,
}

/// The largest single `read`; a body streams through many.
const MAX_READ: usize = 1 << 20;

fn reset_code(code: i32) -> PyResult<HttpStreamResetCode> {
    HttpStreamResetCode::try_from(code)
        .map_err(|_| PyValueError::new_err(format!("unknown HttpStreamResetCode {code}")))
}

#[pymethods]
impl PyEndpointStream {
    /// Up to `max_bytes` of the peer's bytes; `b""` once the peer half-closed (FIN). Raises when
    /// the peer reset the stream.
    #[pyo3(signature = (max_bytes = 65536))]
    fn read<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let reader = self.reader.clone();
        let max_bytes = max_bytes.clamp(1, MAX_READ);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = reader.lock().await;
            let Some(reader) = guard.as_mut() else {
                return Err(PyRuntimeError::new_err("endpoint stream is closed"));
            };
            let mut buf = vec![0u8; max_bytes];
            match reader.read(&mut buf).await {
                Ok(Some(n)) => {
                    buf.truncate(n);
                    Ok(buf)
                }
                Ok(None) => Ok(Vec::new()),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            }
        })
        .map(|awaitable| awaitable.into_any())
    }

    /// Writes all of `data`, waiting on QUIC flow control.
    fn write<'py>(&self, py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let writer = self.writer.clone();
        let data = data.to_vec();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = writer.lock().await;
            let Some(writer) = guard.as_mut() else {
                return Err(PyRuntimeError::new_err("endpoint stream is closed"));
            };
            writer.write_all(&data).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// FIN: the response is complete. Idempotent; a stream already reset raises.
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        check_pid()?;
        let writer = self.writer.clone();
        py.detach(|| {
            let mut guard = writer.blocking_lock();
            match guard.as_mut() {
                Some(writer) => writer.finish().map_err(|e| PyRuntimeError::new_err(e.to_string())),
                None => Ok(()),
            }
        })
    }

    /// RESET_STREAM with an `HttpStreamResetCode` number, and STOP_SENDING with the same.
    fn reset(&self, py: Python<'_>, code: i32) -> PyResult<()> {
        check_pid()?;
        let code = reset_code(code)?;
        let (writer, reader) = (self.writer.clone(), self.reader.clone());
        py.detach(|| {
            if let Some(w) = writer.blocking_lock().as_mut() {
                w.reset(code);
            }
            if let Some(r) = reader.blocking_lock().as_mut() {
                r.stop(code);
            }
        });
        Ok(())
    }

    /// STOP_SENDING only: the response went out whole and nothing more of the request is wanted.
    fn stop(&self, py: Python<'_>, code: i32) -> PyResult<()> {
        check_pid()?;
        let code = reset_code(code)?;
        let reader = self.reader.clone();
        py.detach(|| {
            if let Some(r) = reader.blocking_lock().as_mut() {
                r.stop(code);
            }
        });
        Ok(())
    }
}

// --------------------------------------------------------------------- the host

struct PyWorkerHost {
    obj: Py<PyAny>,
    locals: TaskLocals,
}

impl PyWorkerHost {
    /// `host.open(kind, args)` as a scheduled coroutine; dropping the operation cancels the task.
    async fn open<F>(&self, kind: &'static str, args: F) -> Result<Py<PyAny>, HostError>
    where
        F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>>,
    {
        let (task, receiver): (CancelPythonTask, _) = Python::attach(|py| -> PyResult<_> {
            let coro = self.obj.bind(py).call_method1("open", (kind, args(py)?))?;
            schedule_coroutine(py, &self.locals, coro)
        })
        .map_err(|e| Python::attach(|py| capture_host(py, &e)))?;
        let result = receiver
            .await
            .map_err(|_| HostError::new(format!("open({kind:?}): Python task disappeared")))?;
        drop(task);
        result.map_err(|e| Python::attach(|py| capture_host(py, &e)))
    }
}

#[async_trait]
impl WorkerHost for PyWorkerHost {
    async fn on_job_request(&self, req: crate::JobRequest) -> JobDecision {
        let job_id = req.job.id.clone();
        // `resuming` is set when a call migrates off a draining worker.
        let res = self.open("job_request", |py| Ok((&req).into_pyobject(py)?.into_any())).await;
        match res {
            // A failed `request_fnc` rejects without terminating, so the server can redispatch.
            Err(e) => JobDecision::Failed(e),
            Ok(v) => Python::attach(|py| parse_decision(py, v.bind(py)))
                .unwrap_or_else(|e| JobDecision::Failed(HostError::new(format!("{job_id}: {e}")))),
        }
    }

    async fn launch_job(&self, info: crate::RunningJobInfo) -> Result<JobId, HostError> {
        let v = self.open("launch_job", |py| Ok((&info).into_pyobject(py)?.into_any())).await?;
        Python::attach(|py| v.extract::<String>(py))
            .map_err(|e| HostError::new(format!("launch_job returned a non-str job id: {e}")))
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        self.open("terminate_job", |py| {
            let d = PyDict::new(py);
            d.set_item("job_id", &id)?;
            Ok(d.into_any())
        })
        .await
        .map(|_| ())
    }

    async fn user_load(&self) -> Option<f32> {
        // A raising `load_fnc` must not read as idle (that over-schedules); `None` falls back to
        // the CPU monitor.
        let v = self.open("load", |py| Ok(PyDict::new(py).into_any())).await.ok()?;
        Python::attach(|py| v.extract::<Option<f32>>(py).ok()).flatten()
    }

    async fn serve_endpoint(&self, request: EndpointRequest) {
        let EndpointRequest { preamble, stream } = request;
        let request_id = preamble.request_id.clone();
        let (writer, reader) = stream.split();
        let reader: Half<EndpointReader> = Arc::new(tokio::sync::Mutex::new(Some(reader)));
        let writer: Half<EndpointWriter> = Arc::new(tokio::sync::Mutex::new(Some(writer)));
        let halves = (reader.clone(), writer.clone());
        let kind = match AgentEndpointKind::try_from(preamble.kind) {
            Ok(AgentEndpointKind::AekText) => "text",
            _ => "http",
        };
        let entered = std::sync::atomic::AtomicBool::new(false);
        let outcome = self
            .open("endpoint", |py| {
                let d = PyDict::new(py);
                d.set_item("stream", PyEndpointStream { reader, writer })?;
                d.set_item("kind", kind)?;
                d.set_item("request_id", &preamble.request_id)?;
                d.set_item("authorized", preamble.authorized)?;
                d.set_item("route", &preamble.route)?;
                d.set_item("timeout_ms", preamble.timeout_ms)?;
                d.set_item("client_addr", &preamble.client_addr)?;
                d.set_item("scheme", &preamble.scheme)?;
                entered.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(d.into_any())
            })
            .await;
        if let Err(error) = outcome {
            // `open` itself raising means Python never held the stream: refused is sound. A
            // coroutine that raised may have run the application: internal.
            let code = if entered.load(std::sync::atomic::Ordering::Relaxed) {
                HttpStreamResetCode::HsrInternal
            } else {
                HttpStreamResetCode::HsrRefused
            };
            log::warn!(target: LOG, request_id = request_id.as_str(), error = error.to_string().as_str();
                "endpoint request failed in the host");
            let (reader, writer) = halves;
            if let Some(w) = writer.lock().await.as_mut() {
                w.reset(code);
            }
            if let Some(r) = reader.lock().await.as_mut() {
                r.stop(code);
            }
        }
    }
}

/// The host's answer to `open("job_request", ..)` as a [`JobDecision`]. A malformed dict becomes an
/// error, which the caller turns into `JobDecision::Failed` (reject without terminating).
fn parse_decision(py: Python<'_>, v: &Bound<'_, PyAny>) -> Result<JobDecision, String> {
    let d = v.cast::<PyDict>().map_err(|_| "expected a dict".to_string())?;
    let host_err = |e: PyErr| capture_host(py, &e).to_string();
    let raw: crate::RawJobDecision = d.as_any().extract().map_err(host_err)?;
    JobDecision::from_wire(
        &raw.decision,
        || d.as_any().extract::<crate::AcceptArguments>().map_err(host_err),
        || raw.error.clone().unwrap_or_default(),
    )
}

// --------------------------------------------------------------------- options

/// One manifest entry from `options["endpoints"]`: `{"path", "methods", "kind", "public"}`, with
/// `kind` `"http"` (default) or `"text"` for a session endpoint (plan 9.1).
fn endpoint_from_py(item: &Bound<'_, PyAny>) -> PyResult<protocol::AgentEndpoint> {
    let d = item.cast::<PyDict>().map_err(|_| PyValueError::new_err("endpoints: expected dicts"))?;
    let path: String = d
        .get_item("path")?
        .ok_or_else(|| PyValueError::new_err("endpoints: an entry has no path"))?
        .extract()?;
    let methods: Vec<String> = match d.get_item("methods")? {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    let kind = match d.get_item("kind")?.map(|v| v.extract::<String>()).transpose()?.as_deref() {
        None | Some("http") => AgentEndpointKind::AekHttp,
        Some("text") => AgentEndpointKind::AekText,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "endpoints: {path}: unknown kind {other:?}, expected \"http\" or \"text\""
            )));
        }
    };
    let public = match d.get_item("public")? {
        Some(v) => v.extract()?,
        None => false,
    };
    Ok(protocol::AgentEndpoint { path, methods, kind: kind as i32, public })
}

/// `options` as [`WorkerOptions`]. Python has already resolved option and environment precedence,
/// so every string is passed explicitly (an empty one disables the environment fallback).
fn options_from_py(options: &Bound<'_, PyDict>) -> PyResult<WorkerOptions> {
    fn get<'py, T>(d: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        match d.get_item(key)? {
            Some(v) if !v.is_none() => v.extract::<T>().map_err(|e| {
                let e: PyErr = e.into();
                PyValueError::new_err(format!("options[{key:?}]: {e}"))
            }),
            _ => Ok(default),
        }
    }
    let d = WorkerConfig::default();
    let permissions = match options.get_item("permissions")? {
        Some(p) if !p.is_none() => p.extract()?,
        _ => crate::WorkerPermissions::default(),
    };
    let endpoints = match options.get_item("endpoints")? {
        Some(list) if !list.is_none() => list
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("options[\"endpoints\"]: expected a list"))?
            .iter()
            .map(|item| endpoint_from_py(&item))
            .collect::<PyResult<Vec<_>>>()?,
        _ => Vec::new(),
    };
    let heartbeat_secs: Option<f64> = get(options, "heartbeat_secs", None)?;
    let load_threshold: Option<f32> = get(options, "load_threshold", None)?;
    let num_idle_processes: Option<u32> = get(options, "num_idle_processes", None)?;
    let drain_timeout: Option<f64> = get(options, "drain_timeout", Some(3600.0))?;
    let worker_token: Option<String> = get(options, "worker_token", None)?;
    let headers: Vec<(String, String)> = get(options, "headers", Vec::new())?;
    let webtransport: bool = get(options, "webtransport", !endpoints.is_empty())?;
    let worker_url: Option<String> = get(options, "worker_url", None)?;
    Ok(WorkerOptions {
        ws_url: Some(get(options, "ws_url", String::new())?),
        worker_url: Some(worker_url.unwrap_or_default()),
        agent_name: Some(get(options, "agent_name", String::new())?),
        deployment: Some(get(options, "deployment", String::new())?),
        server_type: if get(options, "publisher", false)? {
            ServerType::Publisher
        } else {
            ServerType::Room
        },
        version: get(options, "version", "0.0.0".to_owned())?,
        permissions,
        endpoints,
        load_threshold: load_threshold.map(EnvOption::Fixed).unwrap_or(d.load_threshold),
        num_idle_processes: Some(
            num_idle_processes.map(EnvOption::Fixed).unwrap_or(d.num_idle_processes),
        ),
        devmode: get(options, "devmode", false)?,
        simulation: get(options, "simulation", false)?,
        drain_timeout: secs(drain_timeout),
        worker_token: Some(worker_token.unwrap_or_default()),
        headers,
        api_key: Some(get(options, "api_key", String::new())?),
        api_secret: Some(get(options, "api_secret", String::new())?),
        max_retry: get(options, "max_retry", 16)?,
        transport: if webtransport {
            TransportConfig::WebTransport { insecure: get(options, "insecure", false)? }
        } else {
            TransportConfig::WebSocket {
                heartbeat: heartbeat_secs
                    .map(|s| Duration::from_secs_f64(s.max(1.0)))
                    .unwrap_or(HEARTBEAT_INTERVAL),
            }
        },
        ..WorkerOptions::default()
    })
}

// --------------------------------------------------------------------- Worker

type Conn = Worker<Box<dyn WorkerTransport>, PyWorkerHost, RealClock>;

/// The Rust worker runtime, as seen from Python.
#[pyclass(name = "Worker", module = "livekit.agents._core")]
pub struct PyWorker {
    /// Taken by `run()`. A second `run()` is an error rather than a second socket.
    conn: std::sync::Mutex<Option<Box<Conn>>>,
    handle: WorkerHandle,
    notifications: Arc<tokio::sync::Mutex<tokio::sync::broadcast::Receiver<WorkerNotification>>>,
}

fn queue_full<T>(_: T) -> PyErr {
    PyRuntimeError::new_err("worker stopped or command queue full")
}

#[pymethods]
impl PyWorker {
    /// Construct on the host event loop. `options` is a dict (see the module docs for the keys);
    /// `host.open(kind, args)` serves every callback. Rust signs authorization for each
    /// connection attempt from `api_key`/`api_secret`.
    #[new]
    fn new(py: Python<'_>, options: &Bound<'_, PyDict>, host: Py<PyAny>) -> PyResult<Self> {
        check_pid()?;
        if !host.bind(py).hasattr("open")? {
            return Err(PyValueError::new_err("host object has no open(kind, args)"));
        }
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let options = options_from_py(options)?;
        let (conn, handle) = Worker::new(options, PyWorkerHost { obj: host, locals });
        Ok(PyWorker {
            conn: std::sync::Mutex::new(Some(Box::new(conn))),
            notifications: Arc::new(tokio::sync::Mutex::new(handle.subscribe())),
            handle,
        })
    }

    /// Connect, register, and serve until closed. Awaitable; resolves when the worker stops.
    fn run<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let mut conn = self
            .conn
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this worker has already been run"))?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            conn.run().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Completes when all pending requests and jobs drain, or raises `TimeoutError`.
    #[pyo3(signature = (timeout=None))]
    fn drain<'py>(&self, py: Python<'py>, timeout: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let handle = self.handle.clone();
        let timeout = timeout
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle.drain(timeout).await.map_err(|e| {
                if e == "drain timed out" {
                    pyo3::exceptions::PyTimeoutError::new_err(e)
                } else {
                    PyRuntimeError::new_err(e)
                }
            })
        })
    }

    /// Ordered runtime notifications (`kind`: `registered`, `drain_finished`, `drain_timed_out`,
    /// `stopped`), separate from host operations and current state.
    fn next_notification<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let notifications = self.notifications.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let event = notifications
                .lock()
                .await
                .recv()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| {
                let d = match &event {
                    WorkerNotification::Registered(v) => v.into_pyobject(py)?,
                    WorkerNotification::DrainFinished(v) => v.into_pyobject(py)?,
                    WorkerNotification::DrainTimedOut(v) => v.into_pyobject(py)?,
                    WorkerNotification::Stopped(v) => v.into_pyobject(py)?,
                };
                Ok::<_, PyErr>(d.unbind())
            })
        })
    }

    /// Stop reconnecting and return from `run()`. Not `drain`: this does not wait for jobs.
    fn close(&self) -> PyResult<()> {
        check_pid()?;
        if self.handle.snapshot().finished.is_some() {
            return Ok(());
        }
        self.handle.send(WorkerCommand::Close).map_err(queue_full)
    }

    /// Report status using the LiveKit job ID, not a process identifier.
    fn job_status(&self, job_id: String, status: i32) -> PyResult<()> {
        check_pid()?;
        let status = protocol::JobStatus::try_from(status)
            .map_err(|_| PyValueError::new_err(format!("unknown job status {status}")))?;
        self.handle.send(WorkerCommand::JobStatus { job_id, status }).map_err(queue_full)
    }

    /// `worker_id`, `draining`, `active_jobs`, `reserved_slots`, `worker_load`,
    /// `target_idle_processes`, `drain_outcome`, `finished`.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.handle.snapshot().into_pyobject(py)
    }
}

lk_agents_abi::register_host!(class PyWorker);
lk_agents_abi::register_host!(class PyEndpointStream);

#[cfg(test)]
mod tests {
    use super::*;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    // Catches: `ServerOptions.permissions` not reaching the config (`hidden=True` registering a
    // visible agent), or a partial dict raising instead of defaulting.
    #[test]
    fn the_permissions_dict_reaches_the_config_and_missing_keys_default() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("hidden", true).unwrap();
            d.set_item("can_publish", false).unwrap();
            d.set_item("can_publish_sources", vec![1, 2]).unwrap();
            let p: crate::WorkerPermissions = d.as_any().extract().unwrap();
            assert!(p.hidden && !p.can_publish, "{p:?}");
            assert_eq!(p.can_publish_sources, vec![1, 2]);
            assert!(p.can_subscribe && p.can_publish_data && p.can_update_metadata, "{p:?}");
            let empty: crate::WorkerPermissions = PyDict::new(py).as_any().extract().unwrap();
            assert_eq!(empty, crate::WorkerPermissions::default());
        });
    }

    // Catches: an acceptance field lost on the way in, a missing one refused rather than
    // defaulted, or a wrong-typed one read as empty.
    #[test]
    fn an_accept_decision_carries_its_fields_defaults_missing_ones_and_refuses_wrong_types() {
        attach(|py| {
            let d = PyDict::new(py);
            d.set_item("decision", "accept").unwrap();
            d.set_item("name", "agent-7").unwrap();
            d.set_item("identity", "id-7").unwrap();
            d.set_item("metadata", "{}").unwrap();
            d.set_item("attributes", vec![("k".to_string(), "v".to_string())]).unwrap();
            match parse_decision(py, d.as_any()).unwrap() {
                JobDecision::Accept(a) => {
                    assert_eq!(a.name, "agent-7");
                    assert_eq!(a.identity, "id-7");
                    assert_eq!(a.metadata, "{}");
                    assert_eq!(a.attributes, vec![("k".to_string(), "v".to_string())]);
                }
                other => panic!("{other:?}"),
            }

            let bare = PyDict::new(py);
            bare.set_item("decision", "accept").unwrap();
            match parse_decision(py, bare.as_any()).unwrap() {
                JobDecision::Accept(a) => {
                    assert_eq!(a.name, "");
                    assert!(a.attributes.is_empty());
                }
                other => panic!("{other:?}"),
            }

            bare.set_item("name", 7).unwrap();
            let err = parse_decision(py, bare.as_any()).expect_err("a non-str name must be reported");
            assert!(err.contains("name") || err.contains("str"), "{err}");
        });
    }

    // Catches: a manifest entry's kind or methods lost between Python and the register message,
    // or endpoints not selecting the WebTransport transport the data plane needs.
    #[test]
    fn endpoints_reach_the_manifest_with_their_kind_and_select_webtransport() {
        attach(|py| {
            let text = PyDict::new(py);
            text.set_item("path", "/a2a").unwrap();
            text.set_item("methods", vec!["POST"]).unwrap();
            text.set_item("kind", "text").unwrap();
            let http = PyDict::new(py);
            http.set_item("path", "/items/{id}").unwrap();
            http.set_item("methods", vec!["GET", "HEAD"]).unwrap();
            http.set_item("public", true).unwrap();
            let options = PyDict::new(py);
            options.set_item("endpoints", vec![text, http]).unwrap();
            let o = options_from_py(&options).unwrap();
            assert_eq!(o.endpoints.len(), 2);
            assert_eq!(o.endpoints[0].kind, AgentEndpointKind::AekText as i32);
            assert_eq!(o.endpoints[1].kind, AgentEndpointKind::AekHttp as i32);
            assert_eq!(o.endpoints[1].methods, vec!["GET", "HEAD"]);
            assert!(o.endpoints[1].public && !o.endpoints[0].public);
            assert!(matches!(o.transport, TransportConfig::WebTransport { .. }));

            let bad = PyDict::new(py);
            bad.set_item("path", "/x").unwrap();
            bad.set_item("kind", "grpc").unwrap();
            options.set_item("endpoints", vec![bad]).unwrap();
            assert!(options_from_py(&options).is_err(), "an unknown kind must be refused");

            let none = PyDict::new(py);
            assert!(matches!(options_from_py(&none).unwrap().transport, TransportConfig::WebSocket { .. }));
        });
    }
}

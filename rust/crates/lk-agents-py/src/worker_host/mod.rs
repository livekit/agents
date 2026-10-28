//! Python bootstrap and application adapters for the worker control plane.
//! ABI declarations generate the class surface; worker owns sockets, pools and supervision.
//! Host callbacks enter through HostInvoker under the context captured at operation entry.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::agent::Invocation;
use lk_agents_abi::python::{capture_host, check_pid};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::types::{PyDict, PyList};

use std::sync::atomic::{AtomicBool, Ordering};

pub(super) mod callback;
pub mod cli;
pub mod ipc;
mod server;

use crate::worker::transport::{
    AgentEndpointKind, EndpointReader, EndpointWriter, HttpStreamResetCode,
};
use crate::worker::{
    EndpointRequest, EnvOption, HEARTBEAT_INTERVAL, HostError, JobDecision, JobId, TransportConfig,
    WorkerCommand, WorkerHost, protocol,
};

pub mod endpoint;
use endpoint::{Half, PyEndpointStream};

// --------------------------------------------------------------------- the host

#[derive(Clone)]
struct PyWorkerHost {
    obj: Arc<Py<PyAny>>,
    entry: Invocation,
    /// The user gave a `load_fnc` the worker may call (not on Cloud).
    user_load: bool,
    /// The server has a voice door, and that door an `on_request`.
    has_rtc: bool,
    has_on_request: bool,
}

impl PyWorkerHost {
    /// The server's callbacks inherit the operation's captured host context.
    async fn open<F>(&self, kind: &'static str, args: F) -> Result<Py<PyAny>, HostError>
    where
        F: for<'py> FnOnce(Python<'py>) -> PyResult<Bound<'py, PyAny>> + Send + 'static,
    {
        let object = self.obj.clone();
        lk_agents_runtime::host::call(
            &crate::host::Entry,
            kind,
            &self.entry,
            &[],
            Box::new(move |py| {
                Ok((
                    object.bind(py).getattr("open")?,
                    PyTuple::new(py, [kind.into_pyobject(py)?.into_any(), args(py)?])?,
                ))
            }),
        )
        .await
        .map_err(|error| Python::attach(|py| capture_host(py, &error)))
    }
}

#[async_trait]
impl WorkerHost for PyWorkerHost {
    async fn on_job_request(&self, req: crate::worker::JobRequest) -> JobDecision {
        if let Some(decision) = worker_server::own_decision(&req, self.has_rtc, self.has_on_request)
        {
            return decision;
        }
        let job_id = req.job.id.clone();
        // `resuming` is set when a call migrates off a draining worker.
        let res = self.open("job_request", move |py| Ok(req.into_pyobject(py)?.into_any())).await;
        match res {
            // A failed `request_fnc` rejects without terminating, so the server can redispatch.
            Err(e) => JobDecision::Failed(e),
            Ok(v) => Python::attach(|py| parse_decision(py, v.bind(py)))
                .unwrap_or_else(|e| JobDecision::Failed(HostError::new(format!("{job_id}: {e}")))),
        }
    }

    async fn launch_job(&self, info: crate::worker::RunningJobInfo) -> Result<JobId, HostError> {
        let v = self.open("launch_job", move |py| Ok(info.into_pyobject(py)?.into_any())).await?;
        Python::attach(|py| v.extract::<String>(py))
            .map_err(|e| HostError::new(format!("launch_job returned a non-str job id: {e}")))
    }

    async fn terminate_job(&self, id: JobId) -> Result<(), HostError> {
        self.open("terminate_job", move |py| Ok(id.into_pyobject(py)?.into_any())).await.map(|_| ())
    }

    async fn user_load(&self) -> Option<f32> {
        if !self.user_load {
            return None;
        }
        // A raising `load_fnc` must not read as idle (that over-schedules); `None` falls back to
        // the CPU monitor.
        let v = self.open("load", |py| Ok(PyDict::new(py).into_any())).await.ok()?;
        Python::attach(|py| v.extract::<Option<f32>>(py).ok()).flatten()
    }

    async fn serve_endpoint(&self, request: EndpointRequest) {
        let EndpointRequest { preamble, stream } = request;
        let request_id = preamble.request_id.clone();
        let (writer, reader) = stream.split();
        let control = writer.control();
        let reader: Half<EndpointReader> = Arc::new(tokio::sync::Mutex::new(Some(reader)));
        let writer: Half<EndpointWriter> = Arc::new(tokio::sync::Mutex::new(Some(writer)));
        let halves = (reader.clone(), writer.clone());
        let kind = match AgentEndpointKind::try_from(preamble.kind) {
            Ok(AgentEndpointKind::AekText) => "text",
            _ => "http",
        };
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_entered = entered.clone();
        let outcome = self
            .open("endpoint", move |py| {
                let d = PyDict::new(py);
                d.set_item(
                    "stream",
                    crate::generated::worker::EndpointStream(Arc::new(PyEndpointStream {
                        reader,
                        writer,
                        control,
                    })),
                )?;
                d.set_item("kind", kind)?;
                d.set_item("request_id", &preamble.request_id)?;
                d.set_item("authorized", preamble.authorized)?;
                d.set_item("route", &preamble.route)?;
                d.set_item("timeout_ms", preamble.timeout_ms)?;
                d.set_item("client_addr", &preamble.client_addr)?;
                d.set_item("scheme", &preamble.scheme)?;
                callback_entered.store(true, std::sync::atomic::Ordering::Relaxed);
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
            crate::worker::diagnostics::emit(
                log::Level::Warn,
                format!(
                    "endpoint request failed in the host request_id={request_id} error={error}"
                ),
            );
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
    let raw: crate::worker::RawJobDecision = d.as_any().extract().map_err(host_err)?;
    JobDecision::from_wire(
        &raw.decision,
        || d.as_any().extract::<crate::worker::AcceptArguments>().map_err(host_err),
        || raw.error.clone().unwrap_or_default(),
    )
}

// --------------------------------------------------------------------- the routes

/// One manifest entry: `{"path", "methods", "kind", "public"}`, with `kind` `"http"` (default) or
/// `"text"` for a session endpoint (plan 9.1).
fn endpoint_from_py(item: &Bound<'_, PyAny>) -> PyResult<protocol::AgentEndpoint> {
    let d =
        item.cast::<PyDict>().map_err(|_| PyValueError::new_err("endpoints: expected dicts"))?;
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

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).unwrap_or_default().to_lowercase().as_str(), "1" | "true" | "yes")
}

/// The transport the routes need: the endpoint data plane rides WebTransport; without routes the
/// WebSocket control connection suffices (`LIVEKIT_WORKER_WEBTRANSPORT` asks for it anyway).
fn transport_for(routes: bool) -> TransportConfig {
    if routes || env_flag("LIVEKIT_WORKER_WEBTRANSPORT") {
        TransportConfig::WebTransport { insecure: env_flag("LIVEKIT_WORKER_INSECURE") }
    } else {
        TransportConfig::WebSocket { heartbeat: HEARTBEAT_INTERVAL }
    }
}

// --------------------------------------------------------------------- ServerConfig

use crate::worker::server::{self as worker_server, ServerSettings};

/// `AgentServer`'s options (`_core.ServerConfig`): what it was given, with the environment read
/// where an option was left empty. Writable until the server starts.
pub struct PyServerConfig {
    settings: Arc<std::sync::Mutex<ServerSettings>>,
    running: Arc<AtomicBool>,
}

impl PyServerConfig {
    fn get(&self) -> std::sync::MutexGuard<'_, ServerSettings> {
        self.settings.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl PyServerConfig {
    /// `options`: `AgentServer`'s arguments by name; `http_proxy` left out reads `HTTPS_PROXY`,
    /// else `HTTP_PROXY`.
    fn new(options: lk_agents_abi::worker::worker::ServerOptions) -> PyResult<Self> {
        let mut s = ServerSettings::default();
        let env = |name: &str| std::env::var(name).ok();
        s.http_proxy = ServerSettings::env_proxy(env);
        s.apply(options).map_err(PyValueError::new_err)?;
        Ok(PyServerConfig {
            settings: Arc::new(std::sync::Mutex::new(s.with_env(env))),
            running: Arc::default(),
        })
    }

    /// `update_options`: the given options, before the server starts.
    fn update(&self, options: lk_agents_abi::worker::worker::ServerOptions) -> PyResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("cannot update options after starting the server"));
        }
        self.get().apply(options).map_err(PyValueError::new_err)
    }

    fn ws_url(&self) -> String {
        self.get().ws_url.clone()
    }

    fn api_key(&self) -> String {
        self.get().api_key.clone()
    }

    fn api_secret(&self) -> String {
        self.get().api_secret.clone()
    }

    fn agent_name(&self) -> String {
        self.get().agent_name.clone()
    }

    fn deployment(&self) -> String {
        self.get().deployment.clone()
    }

    /// Hosted on Cloud (`LIVEKIT_WORKER_TOKEN` set).
    fn cloud(&self) -> bool {
        self.get().cloud()
    }

    fn drain_timeout(&self) -> f64 {
        self.get().drain_timeout
    }

    fn simulation(&self) -> bool {
        self.get().simulation
    }

    fn set_simulation(&self, value: bool) {
        self.get().simulation = value;
    }
}

// --------------------------------------------------------------------- ServerRuntime

fn empty(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyDict::new(py).into_any())
}

#[async_trait]
impl crate::worker::server::ServerHost for PyWorkerHost {
    fn configured(&self, has_rtc: bool, has_on_request: bool, user_load: bool) -> Self {
        Self { has_rtc, has_on_request, user_load, ..self.clone() }
    }

    async fn start(
        &self,
        start: ServerSettings,
        devmode: bool,
        has_rtc: bool,
    ) -> Result<(), String> {
        self.open("start", move |py| {
            use lk_agents_abi::worker::worker::{ProcessPoolOptions, ProcessStartOptions};
            Ok(ProcessStartOptions {
                devmode,
                job_processes: has_rtc,
                env: start
                    .child_env()
                    .into_iter()
                    .map(|(key, value)| (key.into(), value))
                    .collect(),
                job_executor_type: start.job_executor_type.clone(),
                multiprocessing_context: start.multiprocessing_context.clone(),
                pool: ProcessPoolOptions {
                    num_idle_processes: start
                        .num_idle_processes
                        .unwrap_or(EnvOption::Fixed(0))
                        .get(devmode),
                    initialize_timeout: start.initialize_process_timeout,
                    close_timeout: start.shutdown_process_timeout,
                    session_end_timeout: start.session_end_timeout,
                    memory_warn_mb: start.job_memory_warn_mb,
                    memory_limit_mb: start.job_memory_limit_mb,
                    http_proxy: start.http_proxy.filter(|proxy| !proxy.is_empty()),
                },
            }
            .into_pyobject(py)?
            .into_any())
        })
        .await
        .map(|_| ())
        .map_err(|error| error.to_string())
    }

    async fn started(&self) -> Result<(), String> {
        self.open("started", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn registered(&self, info: crate::worker::WorkerRegistered) -> Result<(), String> {
        self.open("registered", move |py| Ok(info.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn idle_processes(&self, target: u32) -> Result<(), String> {
        self.open("idle_processes", move |py| Ok(target.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn worker_load(&self, load: f32) -> Result<(), String> {
        self.open("worker_load", move |py| Ok(load.into_pyobject(py)?.into_any()))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    async fn close_websockets(&self) -> Result<(), String> {
        self.open("close_websockets", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn join_jobs(&self) -> Result<(), String> {
        self.open("join_jobs", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn close(&self) -> Result<(), String> {
        self.open("close", empty).await.map(|_| ()).map_err(|error| error.to_string())
    }

    async fn active_jobs(&self) -> usize {
        match self.open("health", empty).await {
            Ok(value) => Python::attach(|py| value.extract::<usize>(py)).unwrap_or(0),
            Err(_) => 0,
        }
    }

    async fn metrics(&self) -> Option<(String, String)> {
        let value = self.open("metrics", empty).await.ok()?;
        Python::attach(|py| value.extract::<(String, String)>(py).ok())
    }
}

pub struct PyServer {
    settings: Arc<std::sync::Mutex<ServerSettings>>,

    /// Seals configuration when the server starts; lifecycle lives in `ServerLifecycle`.
    running: Arc<AtomicBool>,
    runtime: crate::worker::server::ServerRuntime<PyWorkerHost>,
}

impl PyServer {
    fn settings(&self) -> ServerSettings {
        self.settings.lock().unwrap_or_else(|error| error.into_inner()).clone()
    }

    fn id(&self) -> String {
        self.runtime.snapshot().worker_id
    }
}

impl PyServer {
    fn new(py: Python<'_>, config: &PyServerConfig, host: Py<PyAny>) -> PyResult<Self> {
        check_pid()?;
        if !host.bind(py).hasattr("open")? {
            return Err(PyValueError::new_err("host object has no open(kind, args)"));
        }
        let context = crate::host::capture_context(py)?;
        let entry = Invocation {
            operation: 0,
            invocation_id: format!("worker:{}", context.id()),
            agent_name: config.get().agent_name.clone(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            function_call: None,
            approval: None,
            host_context: Some(context),
        };
        let host = PyWorkerHost {
            obj: Arc::new(host),
            entry,
            user_load: true,
            has_rtc: false,
            has_on_request: false,
        };
        Ok(Self {
            settings: config.settings.clone(),
            running: config.running.clone(),
            runtime: crate::worker::server::ServerRuntime::new(
                host,
                Some(
                    crate::process::worker_io()
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
                ),
            ),
        })
    }

    fn worker_id(&self) -> String {
        self.id()
    }

    fn draining(&self) -> bool {
        self.runtime.snapshot().phase == crate::worker::server::ServerPhase::Draining
    }

    fn http_port(&self) -> u16 {
        self.runtime.snapshot().http_port
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &self,
        py: Python<'py>,
        devmode: bool,
        unregistered: bool,
        has_rtc: bool,
        has_text_doors: bool,
        publisher: bool,
        routes: &Bound<'py, PyList>,
        has_load_fnc: bool,
        has_on_request: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let endpoints =
            routes.iter().map(|item| endpoint_from_py(&item)).collect::<PyResult<Vec<_>>>()?;
        let start = crate::worker::server::ServerStart {
            settings: self.settings(),
            devmode,
            unregistered,
            has_rtc,
            has_text_doors,
            publisher,
            transport: transport_for(!endpoints.is_empty()),
            endpoints,
            has_load_fnc,
            has_on_request,
            worker_url: std::env::var("LIVEKIT_WORKER_URL").ok().filter(|url| !url.is_empty()),
        };
        self.running.store(true, Ordering::Release);
        let runtime = self.runtime.clone();
        crate::host::future_into_py(py, async move {
            runtime.run(start).await.map_err(PyRuntimeError::new_err)
        })
    }

    fn drain<'py>(&self, py: Python<'py>, timeout: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let timeout = timeout
            .filter(|timeout| *timeout != 0.0)
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| PyValueError::new_err("drain timeout must be finite and nonnegative"))?;
        let runtime = self.runtime.clone();
        crate::host::future_into_py(py, async move {
            runtime.drain(timeout).await.map_err(|error| {
                if error == "drain timed out" {
                    pyo3::exceptions::PyTimeoutError::new_err(error)
                } else {
                    PyRuntimeError::new_err(error)
                }
            })
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let runtime = self.runtime.clone();
        crate::host::future_into_py(py, async move {
            runtime.close().await.map_err(PyRuntimeError::new_err)
        })
    }

    fn job_status(&self, job_id: String, status: &str) {
        let status = match status {
            "failed" => protocol::JobStatus::JsFailed,
            "success" => protocol::JobStatus::JsSuccess,
            _ => protocol::JobStatus::JsRunning,
        };
        if let Some(worker) = self.runtime.snapshot().worker {
            if worker.send(WorkerCommand::JobStatus { job_id: job_id.clone(), status }).is_err() {
                crate::worker::diagnostics::emit(
                    log::Level::Error,
                    format!("worker job status admission failed job_id={job_id}"),
                );
            }
        }
    }

    fn simulate_job<'py>(
        &self,
        py: Python<'py>,
        room: String,
        fake_job: bool,
        agent_identity: Option<String>,
        room_info: Option<Vec<u8>>,
        token: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        if !self.runtime.snapshot().has_rtc {
            return Err(PyRuntimeError::new_err("a server with no rtc_session runs no job"));
        }
        if self.runtime.snapshot().phase != crate::worker::server::ServerPhase::Running {
            return Err(PyRuntimeError::new_err("the server is not accepting jobs"));
        }
        let room_info = room_info
            .map(|bytes| worker_server::decode_room(&bytes))
            .transpose()
            .map_err(PyValueError::new_err)?;
        let info = worker_server::simulated_job(
            &self.settings(),
            &self.id(),
            &room,
            fake_job,
            agent_identity,
            room_info,
            token,
        )
        .map_err(PyValueError::new_err)?;
        let runtime = self.runtime.clone();
        crate::host::future_into_py(py, async move {
            runtime.simulate_job(info).await.map_err(PyRuntimeError::new_err)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attach<T>(f: impl FnOnce(Python<'_>) -> T) -> T {
        Python::initialize();
        Python::attach(f)
    }

    #[test]
    fn typed_permissions_record_reaches_configuration() {
        attach(|py| {
            let module = py.import("livekit.agents._generated.worker_options").unwrap();
            let fields = PyDict::new(py);
            fields.set_item("hidden", true).unwrap();
            fields.set_item("can_publish", false).unwrap();
            fields.set_item("can_publish_sources", vec![1, 2]).unwrap();
            let permissions = crate::host::construct(
                &module.getattr("WorkerPermissions").unwrap(),
                (),
                Some(&fields),
            )
            .unwrap();
            let options = PyDict::new(py);
            options.set_item("permissions", permissions).unwrap();
            let options = crate::host::construct(
                &module.getattr("ServerOptions").unwrap(),
                (),
                Some(&options),
            )
            .unwrap();
            let mut config = ServerSettings::default();
            config.apply(options.extract().unwrap()).unwrap();
            assert!(config.permissions.hidden);
            assert!(!config.permissions.can_publish);
            assert_eq!(config.permissions.can_publish_sources, vec![1, 2]);
            assert!(config.permissions.can_subscribe && config.permissions.can_publish_data);
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
            let err =
                parse_decision(py, bare.as_any()).expect_err("a non-str name must be reported");
            assert!(err.contains("name") || err.contains("str"), "{err}");
        });
    }

    // Catches: a manifest entry's kind or methods lost between Python and the register message,
    // or routes not selecting the WebTransport transport the data plane needs.
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
            let text = endpoint_from_py(text.as_any()).unwrap();
            let http = endpoint_from_py(http.as_any()).unwrap();
            assert_eq!(text.kind, AgentEndpointKind::AekText as i32);
            assert_eq!(http.kind, AgentEndpointKind::AekHttp as i32);
            assert_eq!(http.methods, vec!["GET", "HEAD"]);
            assert!(http.public && !text.public);
            assert!(matches!(transport_for(true), TransportConfig::WebTransport { .. }));

            let bad = PyDict::new(py);
            bad.set_item("path", "/x").unwrap();
            bad.set_item("kind", "grpc").unwrap();
            assert!(endpoint_from_py(bad.as_any()).is_err(), "an unknown kind must be refused");
        });
    }
}

use super::http::{HttpServer, Response, method_not_allowed, response, text};
use super::*;
use hyper::header::HeaderValue;
use hyper::{Method, StatusCode};
use lk_agents_abi::worker::server_host::HealthState;
use lk_agents_net::export::prometheus;

/// What `/worker` reads from the host, beside the server's own state.
#[async_trait::async_trait]
pub trait HealthProbe: Send + Sync {
    /// The active jobs.
    async fn health(&self) -> usize;
}

/// How long the host may go without answering before the worker reads as stuck.
pub const STUCK_AFTER: Duration = Duration::from_secs(5);
/// How long a registered worker may stay disconnected before `/` fails: a short reconnect is
/// expected, a longer one is what a restart fixes.
pub const RECONNECT_GRACE: Duration = Duration::from_secs(30);

/// The server state the listener and `/worker` read, written by the lifecycle as it happens.
pub struct ServerStatus {
    /// The host started: the job process pool filled for the first time.
    pub started: AtomicBool,
    /// Registered at least once, or running unregistered (the console).
    pub registered: AtomicBool,
    /// Since when the worker has not been registered on a connection; `None` while it is.
    disconnected_since: parking_lot::Mutex<Option<tokio::time::Instant>>,
    pub draining: AtomicBool,
    pub failed: AtomicBool,
    /// The host's last answer, refreshed on every load update.
    heartbeat: parking_lot::Mutex<tokio::time::Instant>,
    /// `f32` bits.
    pub worker_load: AtomicU32,
    /// Where the load is recorded for `/metrics`.
    pub instruments: &'static telemetry::process::Instruments,
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self::new(&telemetry::process::INSTRUMENTS)
    }
}

impl ServerStatus {
    /// A starting server recording its load into `instruments`.
    pub fn new(instruments: &'static telemetry::process::Instruments) -> Self {
        Self {
            started: AtomicBool::new(false),
            registered: AtomicBool::new(false),
            disconnected_since: parking_lot::Mutex::new(Some(tokio::time::Instant::now())),
            draining: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            heartbeat: parking_lot::Mutex::new(tokio::time::Instant::now()),
            worker_load: AtomicU32::new(0),
            instruments,
        }
    }

    pub fn load(&self) -> f32 {
        f32::from_bits(self.worker_load.load(Ordering::Relaxed))
    }

    pub fn set_load(&self, load: f32) {
        self.worker_load.store(load.to_bits(), Ordering::Relaxed);
        self.instruments.set::<telemetry::WorkerLoad>(f64::from(load));
    }

    /// Registered on the current connection, or not; a disconnection keeps its first instant.
    pub fn set_connected(&self, connected: bool) {
        let mut since = self.disconnected_since.lock();
        match (connected, *since) {
            (true, _) => *since = None,
            (false, None) => *since = Some(tokio::time::Instant::now()),
            (false, Some(_)) => {}
        }
    }

    pub fn heartbeat(&self) {
        *self.heartbeat.lock() = tokio::time::Instant::now();
    }

    pub fn state(&self) -> HealthState {
        let flag = |flag: &AtomicBool| flag.load(Ordering::Acquire);
        if flag(&self.failed) {
            HealthState::Failed
        } else if !flag(&self.started) || !flag(&self.registered) {
            HealthState::Starting
        } else if self.heartbeat.lock().elapsed() > STUCK_AFTER {
            HealthState::Stuck
        } else if flag(&self.draining) {
            HealthState::Draining
        } else if let Some(since) = *self.disconnected_since.lock() {
            if since.elapsed() > RECONNECT_GRACE {
                HealthState::Disconnected
            } else {
                HealthState::Reconnecting
            }
        } else {
            HealthState::Ready
        }
    }
}

/// `livekit.WorkerInfo` under its proto field names: the LiveKit Cloud watcher decodes `/worker`
/// with Go's `encoding/json`, which reads only those, not pbjson's camelCase.
#[derive(serde::Serialize)]
struct WorkerInfo<'a> {
    worker_type: &'static str,
    agent_name: &'a str,
    active_jobs: f32,
    sdk_version: &'a str,
    worker_load: f32,
    protocol_version: i32,
}

fn state_response(ok: bool, state: HealthState) -> Response {
    let status = if ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    text(status, if ok { "OK" } else { state.as_str() })
}

/// `/` answers whether the worker is up, `/ready` whether it should receive traffic, `/worker`
/// what it is.
pub(super) async fn health_route(
    method: &Method,
    path: &str,
    settings: &ServerSettings,
    status: &ServerStatus,
    probe: &dyn HealthProbe,
) -> Response {
    let path = path.split('?').next().unwrap_or_default();
    if !matches!(path, "/" | "/ready" | "/worker") {
        return text(StatusCode::NOT_FOUND, "404: Not Found");
    }
    if method != Method::GET && method != Method::HEAD {
        return method_not_allowed();
    }
    let state = status.state();
    match path {
        "/" => return state_response(state.live(), state),
        "/ready" => return state_response(state.ready(), state),
        _ => {}
    }
    let info = WorkerInfo {
        worker_type: proto::JobType::JtRoom.as_str_name(),
        agent_name: &settings.agent_name,
        active_jobs: probe.health().await as f32,
        sdk_version: &settings.version,
        worker_load: status.load(),
        protocol_version: WORKER_PROTOCOL_VERSION as i32,
    };
    match serde_json::to_string_pretty(&info) {
        Ok(body) => response(StatusCode::OK, HeaderValue::from_static("application/json"), body),
        Err(_) => text(StatusCode::INTERNAL_SERVER_ERROR, "worker info unavailable"),
    }
}

/// `/metrics`: the Prometheus exposition of `instruments`.
pub(super) fn metrics_route(
    method: &Method,
    path: &str,
    instruments: &telemetry::process::Instruments,
) -> Response {
    if path.split('?').next().unwrap_or_default() != "/metrics" {
        return text(StatusCode::NOT_FOUND, "404: Not Found");
    }
    if method != Method::GET && method != Method::HEAD {
        return method_not_allowed();
    }
    let content_type = HeaderValue::from_static(prometheus::CONTENT_TYPE);
    response(StatusCode::OK, content_type, prometheus::exposition(instruments))
}

/// The health listener on `listener` until dropped.
pub async fn serve_health(
    listener: tokio::net::TcpListener,
    settings: Arc<ServerSettings>,
    status: Arc<ServerStatus>,
    probe: Arc<dyn HealthProbe>,
    stop: tokio_util::sync::CancellationToken,
) -> Result<(), ServerError> {
    HttpServer::new(listener)
        .serve(stop, move |request| {
            let (settings, status, probe) = (settings.clone(), status.clone(), probe.clone());
            async move {
                health_route(request.method(), request.uri().path(), &settings, &status, &*probe)
                    .await
            }
        })
        .await
}

/// The Prometheus listener on `listener`, serving `instruments`, until dropped.
pub async fn serve_metrics(
    listener: tokio::net::TcpListener,
    instruments: &'static telemetry::process::Instruments,
    stop: tokio_util::sync::CancellationToken,
) -> Result<(), ServerError> {
    HttpServer::new(listener)
        .serve(stop, move |request| {
            let route = metrics_route(request.method(), request.uri().path(), instruments);
            std::future::ready(route)
        })
        .await
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod tests;

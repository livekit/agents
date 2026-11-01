//! Health and Prometheus routes; HTTP transport and limits live in `http`.
use super::http::{HttpServer, Response, method_not_allowed, response, text};
use super::*;
use hyper::header::HeaderValue;
use hyper::{Method, StatusCode};
/// What the health and metrics listeners read from the host, beside the server's own state.
#[async_trait::async_trait]
pub trait HealthProbe: Send + Sync {
    /// The active jobs, for `/worker`.
    async fn health(&self) -> usize;
    /// The Prometheus exposition, `(content type, body)`; `None` when it cannot be rendered.
    async fn metrics(&self) -> Option<(String, String)>;
}

/// The server state the listener and `/worker` read.
pub struct ServerStatus {
    pub ready: tokio::sync::watch::Sender<bool>,
    pub connection_failed: AtomicBool,

    /// `f32` bits.
    pub worker_load: AtomicU32,
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self {
            ready: tokio::sync::watch::channel(false).0,
            connection_failed: AtomicBool::new(false),
            worker_load: AtomicU32::new(0),
        }
    }
}

impl ServerStatus {
    pub fn load(&self) -> f32 {
        f32::from_bits(self.worker_load.load(Ordering::Relaxed))
    }

    pub fn set_load(&self, load: f32) {
        self.worker_load.store(load.to_bits(), Ordering::Relaxed);
    }
}

/// `/` answers whether the worker is serving, `/worker` what it is.
pub(super) async fn health_route(
    method: &Method,
    path: &str,
    settings: &ServerSettings,
    status: &ServerStatus,
    probe: &dyn HealthProbe,
) -> Response {
    let path = path.split('?').next().unwrap_or_default();
    if path != "/" && path != "/worker" {
        return text(StatusCode::NOT_FOUND, "404: Not Found");
    }
    if method != Method::GET && method != Method::HEAD {
        return method_not_allowed();
    }
    if path == "/" {
        if !*status.ready.borrow() {
            return text(StatusCode::SERVICE_UNAVAILABLE, "worker startup incomplete");
        }
        if status.connection_failed.load(Ordering::Relaxed) {
            return text(StatusCode::SERVICE_UNAVAILABLE, "failed to connect to livekit");
        }
        return text(StatusCode::OK, "OK");
    }
    let info = proto::WorkerInfo {
        worker_type: "JT_ROOM".into(),
        agent_name: settings.agent_name.clone(),
        active_jobs: probe.health().await as f32,
        sdk_version: settings.version.clone(),
        worker_load: status.load(),
        protocol_version: WORKER_PROTOCOL_VERSION as i32,
    };
    match serde_json::to_string_pretty(&info) {
        Ok(body) => response(StatusCode::OK, HeaderValue::from_static("application/json"), body),
        Err(_) => text(StatusCode::INTERNAL_SERVER_ERROR, "worker info unavailable"),
    }
}

/// `/metrics`: the host's Prometheus exposition.
pub(super) async fn metrics_route(
    method: &Method,
    path: &str,
    probe: &dyn HealthProbe,
) -> Response {
    if path.split('?').next().unwrap_or_default() != "/metrics" {
        return text(StatusCode::NOT_FOUND, "404: Not Found");
    }
    if method != Method::GET && method != Method::HEAD {
        return method_not_allowed();
    }
    match probe.metrics().await {
        Some((content_type, body)) => match HeaderValue::from_str(&content_type) {
            Ok(content_type) => response(StatusCode::OK, content_type, body),
            Err(_) => text(StatusCode::INTERNAL_SERVER_ERROR, "metrics unavailable"),
        },
        None => text(StatusCode::INTERNAL_SERVER_ERROR, "metrics unavailable"),
    }
}

/// The health listener on `listener` until dropped.
pub async fn serve_health(
    listener: tokio::net::TcpListener,
    settings: Arc<ServerSettings>,
    status: Arc<ServerStatus>,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
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

/// The Prometheus listener (`prometheus_port`) on `listener` until dropped.
pub async fn serve_metrics(
    listener: tokio::net::TcpListener,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
    HttpServer::new(listener)
        .serve(stop, move |request| {
            let probe = probe.clone();
            async move { metrics_route(request.method(), request.uri().path(), &*probe).await }
        })
        .await
}

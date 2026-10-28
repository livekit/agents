//! Health and Prometheus listeners with bounded connection ownership.
use super::*;
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

/// What `/worker` reports, as `MessageToJson(WorkerInfo, preserving_proto_field_name=True)` prints
/// it: field order, zero values left out, two-space indent.
pub fn worker_info_json(info: &proto::WorkerInfo) -> String {
    let mut m = serde_json::Map::new();
    let mut put = |k: &str, v: serde_json::Value, zero: bool| {
        if !zero {
            m.insert(k.to_owned(), v);
        }
    };
    put("worker_type", info.worker_type.clone().into(), info.worker_type.is_empty());
    put("agent_name", info.agent_name.clone().into(), info.agent_name.is_empty());
    put("active_jobs", info.active_jobs.into(), info.active_jobs == 0.0);
    put("sdk_version", info.sdk_version.clone().into(), info.sdk_version.is_empty());
    put("worker_load", info.worker_load.into(), info.worker_load == 0.0);
    put("protocol_version", info.protocol_version.into(), info.protocol_version == 0);
    serde_json::to_string_pretty(&serde_json::Value::Object(m)).unwrap_or_default()
}

/// One response: status line, content type, body.
type Response = (u16, &'static str, String, String);

const TEXT: &str = "text/plain; charset=utf-8";

fn text(code: u16, reason: &'static str, body: &str) -> Response {
    (code, reason, TEXT.into(), body.into())
}

/// `/` answers whether the worker is serving, `/worker` what it is.
pub(super) async fn health_route(
    method: &str,
    path: &str,
    settings: &ServerSettings,
    server_type: ServerType,
    status: &ServerStatus,
    probe: &dyn HealthProbe,
) -> Response {
    let path = path.split('?').next().unwrap_or_default();
    if path != "/" && path != "/worker" {
        return text(404, "Not Found", "404: Not Found");
    }
    if method != "GET" && method != "HEAD" {
        return text(405, "Method Not Allowed", "405: Method Not Allowed");
    }
    if path == "/" {
        if !*status.ready.borrow() {
            return text(503, "Service Unavailable", "worker startup incomplete");
        }
        if status.connection_failed.load(Ordering::Relaxed) {
            return text(503, "Service Unavailable", "failed to connect to livekit");
        }
        return text(200, "OK", "OK");
    }
    let info = proto::WorkerInfo {
        worker_type: match server_type {
            ServerType::Room => "JT_ROOM",
            ServerType::Publisher => "JT_PUBLISHER",
        }
        .into(),
        agent_name: settings.agent_name.clone(),
        active_jobs: probe.health().await as f32,
        sdk_version: settings.version.clone(),
        worker_load: status.load(),
        protocol_version: WORKER_PROTOCOL_VERSION as i32,
    };
    (200, "OK", "application/json".into(), worker_info_json(&info))
}

/// `/metrics`: the host's Prometheus exposition.
pub(super) async fn metrics_route(method: &str, path: &str, probe: &dyn HealthProbe) -> Response {
    if path.split('?').next().unwrap_or_default() != "/metrics" {
        return text(404, "Not Found", "404: Not Found");
    }
    if method != "GET" && method != "HEAD" {
        return text(405, "Method Not Allowed", "405: Method Not Allowed");
    }
    match probe.metrics().await {
        Some((content_type, body)) => (200, "OK", content_type, body),
        None => text(500, "Internal Server Error", "metrics unavailable"),
    }
}

/// The health listener on `listener` until dropped.
pub async fn serve_health(
    listener: tokio::net::TcpListener,
    settings: Arc<ServerSettings>,
    server_type: ServerType,
    status: Arc<ServerStatus>,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
    serve(listener, stop, move |method, path| {
        let (settings, status, probe) = (settings.clone(), status.clone(), probe.clone());
        async move { health_route(&method, &path, &settings, server_type, &status, &*probe).await }
    })
    .await
}

/// The Prometheus listener (`prometheus_port`) on `listener` until dropped.
pub async fn serve_metrics(
    listener: tokio::net::TcpListener,
    probe: Arc<dyn HealthProbe>,
    stop: tokio::sync::watch::Receiver<bool>,
) -> Result<(), String> {
    serve(listener, stop, move |method, path| {
        let probe = probe.clone();
        async move { metrics_route(&method, &path, &*probe).await }
    })
    .await
}

/// Serves `listener` until dropped: one request per connection, answered by
/// `respond(method, path)`, then closed.
async fn serve<F, R>(
    listener: tokio::net::TcpListener,
    mut stop: tokio::sync::watch::Receiver<bool>,
    respond: F,
) -> Result<(), String>
where
    F: Fn(String, String) -> R + Send + Sync + 'static,
    R: std::future::Future<Output = Response> + Send + 'static,
{
    let respond = Arc::new(respond);
    let mut connections = tokio::task::JoinSet::new();
    let mut result = Ok(());
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow_and_update() { break; }
            }
            joined = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = joined {
                    result = Err(format!("health/metrics connection: {error}"));
                    break;
                }
            }
            accepted = listener.accept(), if connections.len() < 64 => {
                match accepted {
                    Ok((socket, _)) => {
                        connections.spawn(serve_connection(socket, respond.clone()));
                    }
                    Err(error) => { result = Err(error.to_string()); break; }
                }
            }
        }
    }
    connections.abort_all();
    while let Some(joined) = connections.join_next().await {
        if let Err(error) = joined {
            if !error.is_cancelled() {
                result = Err(format!("health/metrics connection: {error}"));
            }
        }
    }
    result
}

async fn serve_connection<F, R>(mut socket: tokio::net::TcpStream, respond: Arc<F>)
where
    F: Fn(String, String) -> R + Send + Sync + 'static,
    R: std::future::Future<Output = Response> + Send + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let (method, path) = loop {
        let Ok(n) = socket.read(&mut chunk).await else { return };
        if n == 0 || buf.len() > 16 * 1024 {
            return;
        }
        buf.extend_from_slice(&chunk[..n]);
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut req = httparse::Request::new(&mut headers);
        match req.parse(&buf) {
            Ok(httparse::Status::Complete(_)) => {
                break (
                    req.method.unwrap_or_default().to_owned(),
                    req.path.unwrap_or_default().to_owned(),
                );
            }
            Ok(httparse::Status::Partial) => continue,
            Err(_) => return,
        }
    };
    let (code, reason, content_type, body) = respond(method.clone(), path).await;
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
        body.len()
    );
    let _ = socket.write_all(head.as_bytes()).await;
    if method != "HEAD" {
        let _ = socket.write_all(body.as_bytes()).await;
    }
    let _ = socket.shutdown().await;
}

/// A listener's socket on `host:port` (`host` empty: every interface) and the port it got.
pub async fn bind_listener(
    host: &str,
    port: u16,
) -> std::io::Result<(tokio::net::TcpListener, u16)> {
    let host = if host.is_empty() { "0.0.0.0" } else { host };
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

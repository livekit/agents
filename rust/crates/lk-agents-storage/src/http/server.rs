//! A loopback listener owned by the worker; closing waits for its connections to settle.

use std::convert::Infallible;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::{StreamExt, stream::FuturesUnordered};
use http_body_util::{BodyExt, Full, Limited};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode, body::Incoming};
use hyper_util::rt::TokioIo;
use lk_agents_abi::SpawnerExt;
use lk_agents_net::task::TokioSpawner;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, watch};

use super::{ApiError, ThreadService};
use crate::{Storage, StorageError};

const BODY_LIMIT: usize = 64 * 1024;
const CONNECTION_LIMIT: usize = 64;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub struct ThreadServer {
    port: u16,
    stop: watch::Sender<bool>,
    task: Mutex<Option<lk_agents_abi::OwnedTask<()>>>,
}

impl ThreadServer {
    pub async fn start(
        storage: Arc<dyn Storage>,
        project_id: String,
        api_key: String,
        api_secret: String,
        port: u16,
    ) -> Result<Self, StorageError> {
        if project_id.is_empty()
            || project_id == "p_"
            || api_key.is_empty()
            || api_secret.is_empty()
        {
            return Err(StorageError::Invalid(
                "local ThreadService requires a project id and verification credentials".into(),
            ));
        }
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
            .await
            .map_err(|error| StorageError::Unavailable(format!("thread API listener: {error}")))?;
        let port =
            listener.local_addr().map_err(|e| StorageError::Unavailable(e.to_string()))?.port();
        let service = Arc::new(ThreadService::new(storage, project_id, api_key, api_secret));
        let (stop, stopped) = watch::channel(false);
        let task = TokioSpawner::current().spawn(serve(listener, service, stopped));
        Ok(Self { port, stop, task: Mutex::new(Some(task)) })
    }
}

impl lk_agents_abi::storage::http::ThreadServer for ThreadServer {
    fn port(&self) -> u16 {
        self.port
    }
    async fn aclose(&self) {
        self.stop.send_replace(true);
        let mut task = self.task.lock().await;
        if let Some(task) = task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for ThreadServer {
    fn drop(&mut self) {
        self.stop.send_replace(true);
    }
}

async fn serve(
    listener: TcpListener,
    service: Arc<ThreadService>,
    mut stopped: watch::Receiver<bool>,
) {
    let mut connections = FuturesUnordered::new();
    loop {
        if *stopped.borrow_and_update() {
            break;
        }
        tokio::select! {
            biased;
            _ = stopped.changed() => break,
            _ = connections.next(), if !connections.is_empty() => {},
            accepted = listener.accept(), if connections.len() < CONNECTION_LIMIT => {
                let accepted = accepted.inspect_err(|error| log::warn!("thread API accept failed: {error}"));
                let Ok((socket, _)) = accepted else { break };
                let served = connection(socket, service.clone(), stopped.clone());
                connections.push(TokioSpawner::current().spawn(served));
            }
        }
    }
    // Each connection observes the same stop signal and closes its socket before we return.
    while connections.next().await.is_some() {}
}

async fn connection(
    socket: TcpStream,
    service: Arc<ThreadService>,
    mut stopped: watch::Receiver<bool>,
) {
    let handler = service_fn(move |request| {
        let service = service.clone();
        async move { Ok::<_, Infallible>(handle(service, request).await) }
    });
    let connection = http1::Builder::new()
        .keep_alive(false)
        .max_buf_size(64 * 1024)
        .serve_connection(TokioIo::new(socket), handler);
    if *stopped.borrow_and_update() {
        return;
    }
    tokio::select! {
        biased;
        _ = stopped.changed() => {},
        _ = tokio::time::timeout(REQUEST_TIMEOUT, connection) => {},
    }
}

fn local_origin(origin: &str) -> bool {
    url::Url::parse(origin).is_ok_and(|url| {
        matches!(url.scheme(), "http" | "https")
            && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
            && url.username().is_empty()
            && url.password().is_none()
            && url.path() == "/"
            && url.query().is_none()
            && url.fragment().is_none()
    })
}

async fn handle(service: Arc<ThreadService>, request: Request<Incoming>) -> Response<Full<Bytes>> {
    let origin =
        request.headers().get("origin").and_then(|value| value.to_str().ok()).map(str::to_owned);
    if origin.as_deref().is_some_and(|origin| !local_origin(origin)) {
        return response(
            Err(ApiError::forbidden("origin is not a local development frontend")),
            None,
        );
    }
    if request.method() == hyper::Method::OPTIONS {
        return response(Ok(serde_json::json!({})), origin.as_deref());
    }
    let (parts, body) = request.into_parts();
    let result = match Limited::new(body, BODY_LIMIT).collect().await {
        Ok(body) => {
            service
                .request(
                    parts.method.as_str(),
                    parts.uri.path(),
                    parts.uri.query().unwrap_or_default(),
                    parts.headers.get("authorization").and_then(|value| value.to_str().ok()),
                    &body.to_bytes(),
                )
                .await
        }
        Err(_) => Err(ApiError {
            status: 413,
            code: "resource_exhausted",
            message: "request body exceeds limit".into(),
        }),
    };
    response(result, origin.as_deref())
}

fn response(
    result: Result<serde_json::Value, ApiError>,
    origin: Option<&str>,
) -> Response<Full<Bytes>> {
    let (status, body) = match result {
        Ok(body) => (StatusCode::OK, body),
        Err(error) => (
            StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            error.body(),
        ),
    };
    let mut response = Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("cache-control", "no-store")
        .header("vary", "Origin");
    if let Some(origin) = origin {
        response = response
            .header("access-control-allow-origin", origin)
            .header("access-control-allow-methods", "GET, POST, PATCH, DELETE, OPTIONS")
            .header("access-control-allow-headers", "Authorization, Content-Type, Api-Version");
    }
    response.body(Full::new(Bytes::from(body.to_string()))).expect("fixed response headers")
}

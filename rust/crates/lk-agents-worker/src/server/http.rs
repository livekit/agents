use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{StreamExt, stream::FuturesUnordered};
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::header::{
    ALLOW, CACHE_CONTROL, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST, TRANSFER_ENCODING,
};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{HeaderMap, Request, StatusCode, Version};
use hyper_util::rt::{TokioIo, TokioTimer};
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskError};
use lk_agents_net::task::TokioSpawner;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use super::ServerError;

// Hyper owns parsing and response framing; this layer owns admission, deadlines and shutdown.

pub(super) type Response = hyper::Response<Full<Bytes>>;

const MAX_CONNECTIONS: usize = 64;
const MAX_HEADERS: usize = 64;
const MAX_HEAD_BYTES: usize = 16 * 1024;
const HEADER_TIMEOUT: Duration = Duration::from_secs(5);
// Includes route execution and response writes, so stalled probes and readers release capacity.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

pub(super) fn response(
    status: StatusCode,
    content_type: hyper::header::HeaderValue,
    body: impl Into<Bytes>,
) -> Response {
    hyper::Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .header(CACHE_CONTROL, "no-store")
        .header("x-content-type-options", "nosniff")
        .header(CONNECTION, "close")
        .body(Full::new(body.into()))
        .expect("fixed, valid response headers")
}

pub(super) fn text(status: StatusCode, body: &'static str) -> Response {
    response(status, hyper::header::HeaderValue::from_static("text/plain; charset=utf-8"), body)
}

pub(super) fn method_not_allowed() -> Response {
    let mut response = text(StatusCode::METHOD_NOT_ALLOWED, "405: Method Not Allowed");
    response.headers_mut().insert(ALLOW, hyper::header::HeaderValue::from_static("GET, HEAD"));
    response
}

/// A bounded listener with one request per connection. Dropping or stopping it cancels and
/// joins its connections; malformed requests and peer I/O errors only end that connection.
pub(super) struct HttpServer {
    listener: TcpListener,
}

impl HttpServer {
    pub(super) fn new(listener: TcpListener) -> Self {
        Self { listener }
    }

    pub(super) async fn serve<F, R>(
        self,
        stop: CancellationToken,
        route: F,
    ) -> Result<(), ServerError>
    where
        F: Fn(Request<()>) -> R + Send + Sync + 'static,
        R: Future<Output = Response> + Send + 'static,
    {
        if stop.is_cancelled() {
            return Ok(());
        }
        let mut connections = FuturesUnordered::<OwnedTask>::new();
        let accepted = self.accept_until_stopped(&stop, Arc::new(route), &mut connections).await;
        cancel_connections(connections, accepted).await
    }

    /// Accepts connections up to the cap until `stop`, a failed accept or a failed connection
    /// task.
    async fn accept_until_stopped<F, R>(
        &self,
        stop: &CancellationToken,
        route: Arc<F>,
        connections: &mut FuturesUnordered<OwnedTask>,
    ) -> Result<(), ServerError>
    where
        F: Fn(Request<()>) -> R + Send + Sync + 'static,
        R: Future<Output = Response> + Send + 'static,
    {
        let spawner = TokioSpawner::current();
        loop {
            tokio::select! {
                biased;
                () = stop.cancelled() => return Ok(()),
                joined = connections.next(), if !connections.is_empty() => {
                    if let Some(Err(error)) = joined {
                        return Err(format!("HTTP connection task: {error}").into());
                    }
                }
                accepted = self.listener.accept(), if connections.len() < MAX_CONNECTIONS => {
                    let (socket, _) = accepted?;
                    connections.push(spawner.spawn(serve_connection(socket, route.clone())));
                }
            }
        }
    }
}

/// Cancels and joins every connection; a join failure other than the cancel replaces `result`.
async fn cancel_connections(
    mut connections: FuturesUnordered<OwnedTask>,
    mut result: Result<(), ServerError>,
) -> Result<(), ServerError> {
    connections.iter().for_each(OwnedTask::cancel);
    while let Some(joined) = connections.next().await {
        if let Err(error) = joined
            && !matches!(error, TaskError::Cancelled)
        {
            result = Err(format!("HTTP connection task: {error}").into());
        }
    }
    result
}

fn valid_host(headers: &HeaderMap, version: Version) -> bool {
    let mut hosts = headers.get_all(HOST).iter();
    let Some(host) = hosts.next() else { return version == Version::HTTP_10 };
    if hosts.next().is_some() {
        return false;
    }
    host.to_str()
        .ok()
        .and_then(|host| {
            if host.contains('@') {
                return None;
            }
            host.parse::<hyper::http::uri::Authority>().ok()
        })
        .is_some_and(|host| {
            !host.host().is_empty() && (host.port().is_none() || host.port_u16().is_some())
        })
}

async fn serve_connection<I, F, R>(io: I, route: Arc<F>)
where
    I: AsyncRead + AsyncWrite + Unpin,
    F: Fn(Request<()>) -> R + Send + Sync + 'static,
    R: Future<Output = Response> + Send + 'static,
{
    let service = service_fn(move |request: Request<hyper::body::Incoming>| {
        let route = route.clone();
        async move {
            // These listeners accept origin-form, bodyless requests only. Reject framing before
            // calling the host, and close rather than draining an attacker-controlled body.
            let invalid = !valid_host(request.headers(), request.version())
                || !request.uri().path().starts_with('/')
                || request.uri().scheme().is_some()
                || request.uri().authority().is_some()
                || request.headers().contains_key(TRANSFER_ENCODING)
                || request.headers().get_all(CONTENT_LENGTH).iter().any(|length| length != "0");
            let response = if invalid {
                text(StatusCode::BAD_REQUEST, "400: Bad Request")
            } else {
                route(request.map(|_| ())).await
            };
            Ok::<_, Infallible>(response)
        }
    });
    let mut http = http1::Builder::new();
    http.keep_alive(false)
        .ignore_invalid_headers(false)
        .allow_multiple_spaces_in_request_line_delimiters(false)
        .max_headers(MAX_HEADERS)
        .max_header_size(MAX_HEAD_BYTES)
        .max_buf_size(MAX_HEAD_BYTES)
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_TIMEOUT);
    // Hyper also enforces the header deadline; this bounds the entire exchange, including
    // application callbacks and a peer that stops reading a response.
    match tokio::time::timeout(CONNECTION_TIMEOUT, http.serve_connection(TokioIo::new(io), service))
        .await
    {
        Err(_) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = CONNECTION_TIMEOUT.as_millis() as u64; "worker HTTP exchange timed out")
        }
        Ok(Err(error)) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, error:% = error; "worker HTTP exchange failed")
        }
        Ok(Ok(())) => {}
    }
}

/// Bind numeric addresses directly; resolve names through the shared bounded resolver.
pub async fn bind_listener(host: &str, port: u16) -> std::io::Result<(TcpListener, u16)> {
    let host = if host.is_empty() { "0.0.0.0" } else { host };
    let listener = if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        TcpListener::bind(std::net::SocketAddr::new(ip, port)).await?
    } else {
        let addresses = lk_agents_net::dns::lookup((host.to_owned(), port)).await?;
        TcpListener::bind(addresses.as_slice()).await?
    };
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;

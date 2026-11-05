//! HTTP/1 transport for the worker's bodyless operational endpoints.
//!
//! Hyper owns parsing and response framing. This layer owns admission, deadlines and shutdown;
//! routes only receive a validated request head and return a typed response.

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
use tokio::sync::watch;

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
        mut stop: watch::Receiver<bool>,
        route: F,
    ) -> Result<(), String>
    where
        F: Fn(Request<()>) -> R + Send + Sync + 'static,
        R: Future<Output = Response> + Send + 'static,
    {
        if *stop.borrow_and_update() {
            return Ok(());
        }
        let mut connections = FuturesUnordered::<OwnedTask>::new();
        let accepted =
            self.accept_until_stopped(&mut stop, Arc::new(route), &mut connections).await;
        cancel_connections(connections, accepted).await
    }

    /// Accepts connections up to the cap until `stop`, a failed accept or a failed connection
    /// task.
    async fn accept_until_stopped<F, R>(
        &self,
        stop: &mut watch::Receiver<bool>,
        route: Arc<F>,
        connections: &mut FuturesUnordered<OwnedTask>,
    ) -> Result<(), String>
    where
        F: Fn(Request<()>) -> R + Send + Sync + 'static,
        R: Future<Output = Response> + Send + 'static,
    {
        let spawner = TokioSpawner::current();
        loop {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow_and_update() { return Ok(()); }
                }
                joined = connections.next(), if !connections.is_empty() => {
                    if let Some(Err(error)) = joined {
                        return Err(format!("HTTP connection task: {error}"));
                    }
                }
                accepted = self.listener.accept(), if connections.len() < MAX_CONNECTIONS => {
                    let (socket, _) = accepted.map_err(|error| error.to_string())?;
                    connections.push(spawner.spawn(serve_connection(socket, route.clone())));
                }
            }
        }
    }
}

/// Cancels and joins every connection; a join failure other than the cancel replaces `result`.
async fn cancel_connections(
    mut connections: FuturesUnordered<OwnedTask>,
    mut result: Result<(), String>,
) -> Result<(), String> {
    connections.iter().for_each(OwnedTask::cancel);
    while let Some(joined) = connections.next().await {
        if let Err(error) = joined
            && !matches!(error, TaskError::Cancelled)
        {
            result = Err(format!("HTTP connection task: {error}"));
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
    let _ =
        tokio::time::timeout(CONNECTION_TIMEOUT, http.serve_connection(TokioIo::new(io), service))
            .await;
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
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn exchange(request: &[u8], calls: Arc<AtomicUsize>) -> Vec<u8> {
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(serve_connection(
            server,
            Arc::new(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { text(StatusCode::OK, "hello") }
            }),
        ));
        client.write_all(request).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        serving.await.unwrap();
        response
    }

    #[tokio::test]
    async fn invalid_authorities_and_request_bodies_never_reach_a_route() {
        let calls = Arc::new(AtomicUsize::new(0));
        for request in [
            "GET / HTTP/1.1\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: \r\n\r\n",
            "GET / HTTP/1.1\r\nHost: user@localhost\r\n\r\n",
            "GET http://localhost/ HTTP/1.1\r\nHost: localhost\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n",
        ] {
            let response = exchange(request.as_bytes(), calls.clone()).await;
            assert!(response.starts_with(b"HTTP/1.1 400"), "{request:?}: {response:?}");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn oversized_heads_and_header_counts_are_rejected_before_routing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let oversized = format!(
            "GET / HTTP/1.1\r\nHost: localhost\r\nX-Pad: {}\r\n\r\n",
            "a".repeat(MAX_HEAD_BYTES)
        );
        let headers = format!(
            "GET / HTTP/1.1\r\nHost: localhost\r\n{}\r\n",
            "X-Test: a\r\n".repeat(MAX_HEADERS)
        );
        for request in [oversized, headers] {
            let response = exchange(request.as_bytes(), calls.clone()).await;
            assert!(response.starts_with(b"HTTP/1.1 431"), "{response:?}");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_connection_cannot_pipeline_another_request() {
        let calls = Arc::new(AtomicUsize::new(0));
        let response = exchange(
            b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\nGET /other HTTP/1.1\r\nHost: localhost\r\n\r\n",
            calls.clone(),
        ).await;
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(response.windows(8).filter(|bytes| *bytes == b"HTTP/1.1").count(), 1);
    }

    #[tokio::test]
    async fn bodyless_http10_and_http11_requests_remain_supported() {
        for request in [
            "GET / HTTP/1.0\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: [::1]:8080\r\nContent-Length: 0\r\n\r\n",
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let response = exchange(request.as_bytes(), calls.clone()).await;
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(response.ends_with(b"hello"));
        }
    }

    #[tokio::test]
    async fn a_malformed_peer_does_not_stop_the_listener() {
        let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
        let (stop, stopped) = watch::channel(false);
        let serving = tokio::spawn(
            HttpServer::new(listener).serve(stopped, |_| async { text(StatusCode::OK, "hello") }),
        );
        for (request, status) in [
            ("GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n", "HTTP/1.1 400"),
            ("GET / HTTP/1.1\r\nHost: localhost\r\n\r\n", "HTTP/1.1 200"),
        ] {
            let mut peer = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            peer.write_all(request.as_bytes()).await.unwrap();
            let mut response = String::new();
            tokio::time::timeout(Duration::from_secs(1), peer.read_to_string(&mut response))
                .await
                .unwrap()
                .unwrap();
            assert!(response.starts_with(status), "{response}");
        }
        stop.send_replace(true);
        serving.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn head_has_the_get_content_length_without_the_body() {
        let response =
            exchange(b"HEAD / HTTP/1.1\r\nHost: localhost\r\n\r\n", Arc::new(AtomicUsize::new(0)))
                .await;
        let response = String::from_utf8(response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.to_ascii_lowercase().contains("content-length: 5\r\n"));
        assert!(body.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn partial_headers_release_the_connection_before_the_exchange_deadline() {
        let (mut client, server) = tokio::io::duplex(1024);
        let serving = tokio::spawn(serve_connection(
            server,
            Arc::new(|_| async { panic!("partial request reached the route") }),
        ));
        client.write_all(b"GET / HTTP/1.1\r\n").await.unwrap();
        tokio::task::yield_now().await;
        tokio::time::advance(HEADER_TIMEOUT).await;
        tokio::time::timeout(Duration::from_secs(1), serving).await.unwrap().unwrap();
    }

    struct InFlight(Arc<AtomicBool>);
    impl Drop for InFlight {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn the_exchange_deadline_drops_a_stalled_route() {
        let (mut client, server) = tokio::io::duplex(1024);
        let active = Arc::new(AtomicBool::new(false));
        let started = Arc::new(tokio::sync::Notify::new());
        let (observed, running) = (active.clone(), started.clone());
        let serving = tokio::spawn(serve_connection(
            server,
            Arc::new(move |_| {
                let (active, started) = (active.clone(), started.clone());
                async move {
                    active.store(true, Ordering::SeqCst);
                    let _guard = InFlight(active);
                    started.notify_one();
                    std::future::pending::<Response>().await
                }
            }),
        ));
        client.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").await.unwrap();
        running.notified().await;
        assert!(observed.load(Ordering::SeqCst));
        tokio::time::advance(CONNECTION_TIMEOUT).await;
        serving.await.unwrap();
        assert!(!observed.load(Ordering::SeqCst));
    }

    #[tokio::test(start_paused = true)]
    async fn a_client_that_stops_reading_cannot_hold_a_connection_forever() {
        let (mut client, server) = tokio::io::duplex(64);
        let started = Arc::new(tokio::sync::Notify::new());
        let running = started.clone();
        let serving = tokio::spawn(serve_connection(
            server,
            Arc::new(move |_| {
                let started = started.clone();
                async move {
                    started.notify_one();
                    response(
                        StatusCode::OK,
                        hyper::header::HeaderValue::from_static("text/plain"),
                        vec![b'a'; 32 * 1024],
                    )
                }
            }),
        ));
        client.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n").await.unwrap();
        running.notified().await;
        tokio::time::advance(CONNECTION_TIMEOUT).await;
        serving.await.unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).await.unwrap();
        assert!(response.len() < 32 * 1024, "the response was meant to stall");
    }

    #[tokio::test]
    async fn an_already_stopped_listener_returns_without_accepting() {
        let (listener, _) = bind_listener("127.0.0.1", 0).await.unwrap();
        let (_stop, stopped) = watch::channel(true);
        tokio::time::timeout(
            Duration::from_secs(1),
            HttpServer::new(listener)
                .serve(stopped, |_| async { panic!("stopped listener accepted a request") }),
        )
        .await
        .unwrap()
        .unwrap();
    }
}

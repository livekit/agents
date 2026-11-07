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
    let headers =
        format!("GET / HTTP/1.1\r\nHost: localhost\r\n{}\r\n", "X-Test: a\r\n".repeat(MAX_HEADERS));
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
    )
    .await;
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
    let stop = CancellationToken::new();
    let serving = tokio::spawn(
        HttpServer::new(listener).serve(stop.clone(), |_| async { text(StatusCode::OK, "hello") }),
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
    stop.cancel();
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
    let stopped = CancellationToken::new();
    stopped.cancel();
    tokio::time::timeout(
        Duration::from_secs(1),
        HttpServer::new(listener)
            .serve(stopped, |_| async { panic!("stopped listener accepted a request") }),
    )
    .await
    .unwrap()
    .unwrap();
}

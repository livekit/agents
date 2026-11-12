use super::*;
use crate::server::bind_listener;
use crate::server::settings::tests::settings;
use http_body_util::BodyExt as _;
use hyper::{Method, StatusCode};
use lk_agents_abi::worker::server_host::HealthState;
use lk_agents_telemetry::process::Instruments;

// Catches: a drain or a dropped connection restarting the pod, or traffic before the pool filled.
#[test]
fn health_states_answer_liveness_and_readiness() {
    use std::sync::atomic::Ordering::Release;
    let status = ServerStatus::default();
    assert_eq!(status.state(), HealthState::Starting);
    status.registered.store(true, Release);
    assert_eq!(status.state(), HealthState::Starting, "the pool has not filled");
    status.started.store(true, Release);
    assert_eq!(status.state(), HealthState::Reconnecting, "registered, then disconnected");
    status.set_connected(true);
    assert_eq!(status.state(), HealthState::Ready);
    status.draining.store(true, Release);
    assert_eq!(status.state(), HealthState::Draining);
    status.failed.store(true, Release);
    assert_eq!(status.state(), HealthState::Failed);
    let table = [
        (HealthState::Starting, false, false),
        (HealthState::Ready, true, true),
        (HealthState::Reconnecting, true, false),
        (HealthState::Disconnected, false, false),
        (HealthState::Draining, true, false),
        (HealthState::Failed, false, false),
        (HealthState::Stuck, false, false),
    ];
    for (state, live, ready) in table {
        assert_eq!((state.live(), state.ready()), (live, ready), "{state:?}");
    }
}

#[tokio::test(start_paused = true)]
async fn a_host_that_stops_answering_reads_as_stuck() {
    use std::sync::atomic::Ordering::Release;
    let status = ServerStatus::default();
    for flag in [&status.registered, &status.started] {
        flag.store(true, Release);
    }
    status.set_connected(true);
    status.heartbeat();
    tokio::time::advance(Duration::from_secs(4)).await;
    assert_eq!(status.state(), HealthState::Ready);
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(status.state(), HealthState::Stuck);
    let response = health_route(&Method::GET, "/", &settings(), &status, &Probe).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, "stuck");
}

// Catches: a pod stuck reconnecting reported live forever (kubelet never restarts it), or a short
// reconnect failing liveness; a drain keeps the pod live so its calls finish.
#[tokio::test(start_paused = true)]
async fn a_disconnection_fails_liveness_only_past_the_reconnect_grace() {
    use std::sync::atomic::Ordering::Release;
    let status = ServerStatus::default();
    for flag in [&status.registered, &status.started] {
        flag.store(true, Release);
    }
    status.set_connected(true);
    status.set_connected(false);
    for _ in 0..RECONNECT_GRACE.as_secs() {
        tokio::time::advance(Duration::from_secs(1)).await;
        status.heartbeat();
        status.set_connected(false);
    }
    assert_eq!(status.state(), HealthState::Reconnecting, "the grace counts from the drop");
    tokio::time::advance(Duration::from_millis(1)).await;
    status.heartbeat();
    assert_eq!(status.state(), HealthState::Disconnected);
    let response = health_route(&Method::GET, "/", &settings(), &status, &Probe).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.into_body().collect().await.unwrap().to_bytes(), "disconnected");
    status.draining.store(true, Release);
    assert_eq!(status.state(), HealthState::Draining);
    status.draining.store(false, Release);
    status.set_connected(true);
    assert_eq!(status.state(), HealthState::Ready);
}

struct Probe;

#[async_trait::async_trait]
impl HealthProbe for Probe {
    async fn health(&self) -> usize {
        0
    }
}

#[tokio::test]
async fn endpoint_route_inventory_health_listener_serves_no_content() {
    use std::sync::atomic::Ordering::Release;
    let status = ServerStatus::default();
    assert_eq!(
        health_route(&Method::GET, "/", &settings(), &status, &Probe).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    for flag in [&status.registered, &status.started] {
        flag.store(true, Release);
    }
    status.heartbeat();
    assert_eq!(
        health_route(&Method::GET, "/", &settings(), &status, &Probe).await.status(),
        StatusCode::OK
    );
    let ready = health_route(&Method::GET, "/ready", &settings(), &status, &Probe).await;
    assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE, "reconnecting");
    let body = ready.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, "reconnecting");
    for path in [
        "/threads",
        "/threads/AT_unknown/tasks",
        "/content/file.txt",
        "/.well-known/agent-card.json",
    ] {
        assert_eq!(
            health_route(&Method::GET, path, &settings(), &status, &Probe).await.status(),
            StatusCode::NOT_FOUND
        );
    }
}

/// The raw response to `request` from a metrics listener serving `instruments`.
async fn scrape(instruments: &'static Instruments, request: &[u8]) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
    let stop = tokio_util::sync::CancellationToken::new();
    let serving = tokio::spawn(serve_metrics(listener, instruments, stop.clone()));
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    client.write_all(request).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(1), client.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    stop.cancel();
    serving.await.unwrap().unwrap();
    response
}

// Catches: the load the worker admits by not reaching `/metrics`, a body its declared length or
// content type misdescribes, `HEAD` sending a body, or another path or method answered.
#[tokio::test]
async fn the_metrics_listener_serves_the_worker_load_over_http() {
    let instruments: &'static Instruments = Box::leak(Box::new(Instruments::new()));
    let status = ServerStatus::new(instruments);
    status.set_load(0.5);
    for method in ["GET", "HEAD"] {
        let request = format!("{method} /metrics?format=text HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let response = scrape(instruments, request.as_bytes()).await;
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        let head = head.to_ascii_lowercase();
        assert!(head.starts_with("http/1.1 200"), "{head}");
        assert!(head.contains("content-type: text/plain; version=0.0.4; charset=utf-8"));
        let expected =
            "# TYPE lk_agents_worker_load_ratio gauge\nlk_agents_worker_load_ratio 0.5\n";
        assert!(head.contains(&format!("content-length: {}", expected.len())), "{head}");
        assert_eq!(body, if method == "HEAD" { "" } else { expected });
    }
    let refused = scrape(instruments, b"POST /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
    assert!(refused.starts_with("HTTP/1.1 405"));
    assert!(refused.to_ascii_lowercase().contains("allow: get, head\r\n"));
    let other = scrape(instruments, b"GET /worker HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
    assert!(other.starts_with("HTTP/1.1 404"));
}

// Catches: pbjson's camelCase keys, which the Cloud watcher's `encoding/json` drops: it then reads
// no agent, no deployment support and no active jobs.
#[tokio::test]
async fn the_worker_endpoint_uses_proto_field_names() {
    let response =
        health_route(&Method::GET, "/worker", &settings(), &ServerStatus::default(), &Probe).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        body,
        r#"{
  "worker_type": "JT_ROOM",
  "agent_name": "",
  "active_jobs": 0.0,
  "sdk_version": "2.0.0",
  "worker_load": 0.0,
  "protocol_version": 1
}"#
    );
}

#[tokio::test]
async fn server_shutdown_resources_joins_stalled_health_and_metrics_connections() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for count in [1, 8, 32, 64] {
        let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
        let stop = tokio_util::sync::CancellationToken::new();
        let serving =
            serve_metrics(listener, Box::leak(Box::new(Instruments::new())), stop.clone());
        tokio::pin!(serving);
        let mut peers = Vec::new();
        for _ in 0..count {
            let mut peer = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            peer.write_all(b"GET /metrics HTTP/1.1\r\n").await.unwrap();
            peers.push(peer);
            assert!(futures_util::poll!(&mut serving).is_pending());
        }
        tokio::task::yield_now().await;
        stop.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), serving).await.unwrap().unwrap();
        for mut peer in peers {
            let result =
                tokio::time::timeout(std::time::Duration::from_secs(1), peer.read(&mut [0_u8; 1]))
                    .await
                    .unwrap();
            assert!(
                matches!(result, Ok(0)) || result.is_err(),
                "every accepted connection must be closed"
            );
        }
    }
}

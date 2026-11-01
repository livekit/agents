use super::*;
use http_body_util::BodyExt as _;
use hyper::{Method, StatusCode};
use lk_agents_abi::logging::LogWriter;

fn settings() -> ServerSettings {
    ServerSettings { version: "2.0.0".into(), ..ServerSettings::default() }
}

fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned())
}

// Catches: an argument losing to the environment, the override losing to the argument, or an
// empty variable read as a value.
#[test]
fn options_fall_back_to_the_environment_and_the_override_wins() {
    let mut given = settings();
    given.ws_url = "wss://given".into();
    given.agent_name = "given-agent".into();
    let s = given.with_env(env(&[
        ("LIVEKIT_URL", "wss://env"),
        ("LIVEKIT_API_KEY", "key"),
        ("LIVEKIT_API_SECRET", ""),
        ("LIVEKIT_AGENT_NAME_OVERRIDE", "launcher"),
        ("LIVEKIT_WORKER_TOKEN", "hosted"),
    ]));
    assert_eq!(s.ws_url, "wss://given");
    assert_eq!(s.api_key, "key");
    assert_eq!(s.api_secret, "");
    assert_eq!(s.agent_name, "launcher");
    assert!(s.cloud());
    let s = settings().with_env(env(&[("LIVEKIT_AGENT_NAME", "from-env")]));
    assert_eq!(s.agent_name, "from-env");
    assert!(!s.cloud());
}

// Catches: the worker dialing ws_url when LIVEKIT_WORKER_URL names its address, an empty worker
// token sent as a credential, or the deployment and an empty override lost on the way to
// registration.
#[test]
fn the_worker_registers_and_connects_with_the_resolved_settings() {
    let s = settings().with_env(env(&[
        ("LIVEKIT_URL", "wss://env"),
        ("LIVEKIT_WORKER_URL", "https://env:7883"),
        ("LIVEKIT_WORKER_TOKEN", "hosted"),
        ("LIVEKIT_AGENT_DEPLOYMENT", "staging"),
        ("LIVEKIT_AGENT_NAME_OVERRIDE", ""),
        ("LIVEKIT_AGENT_NAME", "from-env"),
    ]));
    let (worker, connection) = s.worker(false, vec![]);
    assert_eq!(worker.ws_url, "wss://env");
    assert_eq!(connection.server_url, "https://env:7883");
    assert_eq!(connection.worker_token.as_deref(), Some("hosted"));
    assert_eq!((worker.agent_name.as_str(), worker.deployment.as_str()), ("from-env", "staging"));
    let s = settings().with_env(env(&[("LIVEKIT_URL", "wss://env"), ("LIVEKIT_WORKER_URL", "")]));
    let (worker, connection) = s.worker(false, vec![]);
    assert_eq!(connection.server_url, "wss://env", "the worker dials ws_url by default");
    assert_eq!(connection.worker_token, None);
    assert!(worker.deployment.is_empty());
}

// Catches: a missing credential reported as another, or routes allowed without an agent name.
#[test]
fn registration_needs_the_url_the_credentials_and_a_name_for_routes() {
    let mut s = settings();
    assert!(s.check_registration(false).unwrap_err().starts_with("ws_url"));
    s.ws_url = "wss://x".into();
    assert!(s.check_registration(false).unwrap_err().starts_with("api_key"));
    s.api_key = "k".into();
    assert!(s.check_registration(false).unwrap_err().starts_with("api_secret"));
    s.api_secret = "s".into();
    assert!(s.check_registration(false).is_ok());
    assert!(s.check_registration(true).unwrap_err().contains("agent_name"));
}

#[test]
fn invalid_server_update_cannot_partially_change_launch_configuration() {
    use lk_agents_abi::Maybe::Given;
    use lk_agents_abi::worker::types::ServerOptions;
    let mut server = settings();
    let original_url = server.ws_url.clone();
    let error = server
        .apply(ServerOptions {
            ws_url: Given("wss://new.example".into()),
            initialize_process_timeout: Given(f64::NAN),
            ..Default::default()
        })
        .unwrap_err();
    assert!(error.contains("initialize_process_timeout"));
    assert_eq!(server.ws_url, original_url, "a refused update changes no option");
    server
        .apply(ServerOptions {
            ws_url: Given("wss://new.example".into()),
            initialize_process_timeout: Given(4.0),
            drain_timeout: Given(0.0),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(server.ws_url, "wss://new.example");
    assert_eq!(server.initialize_process_timeout, 4.0);
    assert_eq!(server.drain_timeout, 0.0, "zero skips waiting for natural completion");
    assert_eq!(server.shutdown_process_timeout, 10.0, "omission preserves other fields");
}

#[test]
fn unsupported_start_method_is_refused_before_launch() {
    use lk_agents_abi::Maybe::Given;
    use lk_agents_abi::worker::types::ServerOptions;
    let mut server = settings();
    let method = server.multiprocessing_context.clone();
    let error = server
        .apply(ServerOptions {
            multiprocessing_context: Given("fork".into()),
            ..Default::default()
        })
        .unwrap_err();
    assert!(error.contains("multiprocessing_context"));
    assert_eq!(server.multiprocessing_context, method);
}

struct Probe;

#[async_trait::async_trait]
impl HealthProbe for Probe {
    async fn health(&self) -> usize {
        0
    }

    async fn metrics(&self) -> Option<(String, String)> {
        Some(("text/plain; version=0.0.4".into(), "lk_agents_worker_load 0.5\n".into()))
    }
}

#[tokio::test]
async fn endpoint_route_inventory_health_listener_serves_no_content() {
    let status = ServerStatus::default();
    assert_eq!(
        health_route(&Method::GET, "/", &settings(), &status, &Probe).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    status.ready.send_replace(true);
    assert_eq!(
        health_route(&Method::GET, "/", &settings(), &status, &Probe).await.status(),
        StatusCode::OK
    );
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

// Catches: the metrics listener answering another path, or losing the host's content type.
#[tokio::test]
async fn the_metrics_listener_serves_the_host_exposition_at_metrics() {
    let response = metrics_route(&Method::GET, "/metrics", &Probe).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/plain; version=0.0.4");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, "lk_agents_worker_load 0.5\n");
    assert_eq!(metrics_route(&Method::GET, "/", &Probe).await.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        metrics_route(&Method::POST, "/metrics", &Probe).await.status(),
        StatusCode::METHOD_NOT_ALLOWED
    );
}

struct MetricsProbe(String);

#[async_trait::async_trait]
impl HealthProbe for MetricsProbe {
    async fn health(&self) -> usize {
        0
    }

    async fn metrics(&self) -> Option<(String, String)> {
        Some((self.0.clone(), "lk_agents_worker_load 0.5\n".into()))
    }
}

async fn metrics_exchange(request: &[u8], content_type: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let serving =
        tokio::spawn(serve_metrics(listener, Arc::new(MetricsProbe(content_type.into())), stopped));
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    client.write_all(request).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(1), client.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    stop.send_replace(true);
    serving.await.unwrap().unwrap();
    response
}

#[tokio::test]
async fn metrics_responses_use_http_framing_and_preserve_scraper_contracts() {
    for method in ["GET", "HEAD"] {
        let request = format!("{method} /metrics?format=text HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let response = metrics_exchange(request.as_bytes(), "text/plain; version=0.0.4").await;
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200"));
        let head = head.to_ascii_lowercase();
        assert!(head.contains("content-type: text/plain; version=0.0.4"));
        assert!(head.contains("content-length: 26"));
        assert!(head.contains("connection: close"));
        assert_eq!(body, if method == "HEAD" { "" } else { "lk_agents_worker_load 0.5\n" });
    }
    let response =
        metrics_exchange(b"POST /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n", "text/plain").await;
    assert!(response.starts_with("HTTP/1.1 405"));
    assert!(response.to_ascii_lowercase().contains("allow: get, head\r\n"));
}

#[tokio::test]
async fn the_metrics_content_type_cannot_inject_response_headers() {
    for content_type in ["text/plain\r\nX-Injected: yes", "text/plain\n", "text/plain\0"] {
        let response =
            metrics_exchange(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n", content_type)
                .await;
        assert!(response.starts_with("HTTP/1.1 500"));
        assert!(!response.to_ascii_lowercase().contains("x-injected"));
        assert!(response.ends_with("metrics unavailable"));
    }
}

// The endpoint uses livekit-protocol's pbjson-generated Serialize implementation directly.
#[tokio::test]
async fn the_worker_endpoint_uses_generated_protobuf_json() {
    let response =
        health_route(&Method::GET, "/worker", &settings(), &ServerStatus::default(), &Probe).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        body,
        r#"{
  "workerType": "JT_ROOM",
  "sdkVersion": "2.0.0",
  "protocolVersion": 1
}"#
    );
}

// Catches: a text-only server launching a process for a job, or a default accept that differs
// from the agent's identity.
#[test]
fn a_server_takes_a_job_only_with_a_voice_door() {
    let job = proto::Job { id: "AJ_1".into(), ..Default::default() };
    let req = JobRequest { job: crate::worker::job_info(&job), resuming: false };
    assert_eq!(job_decision(&req, false, LogWriter::disabled()), JobDecision::Reject);
    let JobDecision::Accept(args) = job_decision(&req, true, LogWriter::disabled()) else {
        panic!("a voice door accepts")
    };
    assert_eq!((args.identity.as_str(), args.name.as_str()), ("agent-AJ_1", ""));
}

// Catches: a fake job without credentials refused, a real one without its room accepted, or the
// identity a given token names ignored.
#[test]
fn a_simulated_job_names_its_agent_and_room() {
    let s = settings();
    let fake =
        simulated_job(&s, "W1", "console-room", true, Some("console".into()), None, None).unwrap();
    assert!(fake.job.id.starts_with("mock-job-"));
    assert_eq!(fake.job.room.as_deref(), Some("console-room"));
    assert_eq!((fake.accept_arguments.identity.as_str(), fake.token.as_str()), ("console", ""));
    assert_eq!(fake.worker_id, "W1");
    let real = simulated_job(&s, "W1", "r", false, Some("a".into()), None, None).unwrap_err();
    assert_eq!(real, "room_info is None but fake_job is False");
    let mut creds = settings();
    creds.api_key = "APIkey".into();
    creds.api_secret = "thiskeyistotallyunsafe".into();
    let minted = simulated_job(&creds, "W", "r", true, None, None, None).unwrap();
    assert!(minted.accept_arguments.identity.starts_with("fake-agent-"));
    let named =
        simulated_job(&creds, "W", "r", true, Some("x".into()), None, Some(minted.token)).unwrap();
    assert_eq!(named.accept_arguments.identity, minted.accept_arguments.identity);
}

#[tokio::test]
async fn server_shutdown_resources_joins_stalled_health_and_metrics_connections() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for count in [1, 8, 32, 64] {
        let (listener, port) = bind_listener("127.0.0.1", 0).await.unwrap();
        let (stop, stopped) = tokio::sync::watch::channel(false);
        let serving = serve_metrics(listener, Arc::new(Probe), stopped);
        tokio::pin!(serving);
        let mut peers = Vec::new();
        for _ in 0..count {
            let mut peer = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            peer.write_all(b"GET /metrics HTTP/1.1\r\n").await.unwrap();
            peers.push(peer);
            assert!(futures_util::poll!(&mut serving).is_pending());
        }
        tokio::task::yield_now().await;
        stop.send_replace(true);
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

#[test]
fn simulation_is_resolved_before_constructing_the_worker_load_policy() {
    let mut server = settings();
    let (worker, _) = server.worker(false, vec![]);
    assert_eq!(worker.load_threshold.get(false), crate::config::DEFAULT_LOAD_THRESHOLD_PROD);
    server.simulation = true;
    let (worker, _) = server.worker(false, vec![]);
    assert!(worker.load_threshold.get(false).is_infinite());
}

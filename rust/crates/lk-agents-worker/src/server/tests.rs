use super::*;

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
    use lk_agents_abi::worker::worker::ServerOptions;
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
            drain_timeout: Given(None),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(server.ws_url, "wss://new.example");
    assert_eq!(server.initialize_process_timeout, 4.0);
    assert_eq!(server.drain_timeout, 0.0, "explicit null disables the deadline");
    assert_eq!(server.shutdown_process_timeout, 10.0, "omission preserves other fields");
}

#[test]
fn unsupported_start_method_is_refused_before_launch() {
    use lk_agents_abi::Maybe::Given;
    use lk_agents_abi::worker::worker::ServerOptions;
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
    assert_eq!(health_route("GET", "/", &settings(), &status, &Probe).await.0, 503);
    status.ready.send_replace(true);
    assert_eq!(health_route("GET", "/", &settings(), &status, &Probe).await.0, 200);
    for path in [
        "/threads",
        "/threads/AT_unknown/tasks",
        "/content/file.txt",
        "/.well-known/agent-card.json",
    ] {
        assert_eq!(health_route("GET", path, &settings(), &status, &Probe).await.0, 404);
    }
}

// Catches: the metrics listener answering another path, or losing the host's content type.
#[tokio::test]
async fn the_metrics_listener_serves_the_host_exposition_at_metrics() {
    let (code, _, content_type, body) = metrics_route("GET", "/metrics", &Probe).await;
    assert_eq!((code, content_type.as_str()), (200, "text/plain; version=0.0.4"));
    assert_eq!(body, "lk_agents_worker_load 0.5\n");
    assert_eq!(metrics_route("GET", "/", &Probe).await.0, 404);
    assert_eq!(metrics_route("POST", "/metrics", &Probe).await.0, 405);
}

// Catches: `/worker` printing zero fields, out of order, or without the protobuf's indent.
#[test]
fn the_worker_record_prints_as_message_to_json() {
    let info = proto::WorkerInfo {
        worker_type: "JT_ROOM".into(),
        agent_name: String::new(),
        active_jobs: 1.0,
        sdk_version: "2.0.0".into(),
        worker_load: 0.0,
        protocol_version: 1,
    };
    assert_eq!(
        worker_info_json(&info),
        "{\n  \"worker_type\": \"JT_ROOM\",\n  \"active_jobs\": 1.0,\n  \"sdk_version\": \"2.0.0\",\n  \"protocol_version\": 1\n}"
    );
}

// Catches: a text-only server launching a process for a job, or a default accept that differs
// from the agent's identity.
#[test]
fn a_server_takes_a_job_only_with_a_voice_door() {
    let job = proto::Job { id: "AJ_1".into(), ..Default::default() };
    let req = JobRequest { job: crate::protocol::job_info(&job), resuming: false };
    assert_eq!(job_decision(&req, false), JobDecision::Reject);
    let JobDecision::Accept(args) = job_decision(&req, true) else {
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

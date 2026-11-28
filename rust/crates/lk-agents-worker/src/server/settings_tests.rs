use super::*;

pub(crate) fn settings() -> ServerSettings {
    ServerSettings { version: "2.0.0".into(), ..ServerSettings::default() }
}

fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned())
}

// Catches: an empty `HTTP_PROXY` read as a proxy (`Some("")`) instead of unset.
#[test]
fn an_empty_proxy_variable_is_unset() {
    assert_eq!(ServerSettings::env_proxy(env(&[("HTTPS_PROXY", ""), ("HTTP_PROXY", "")])), None);
    let http = ServerSettings::env_proxy(env(&[("HTTPS_PROXY", ""), ("HTTP_PROXY", "http://p")]));
    assert_eq!(http.as_deref(), Some("http://p"));
}

// Catches: an argument losing to the environment, the override losing to the argument, an
// empty variable read as a value, or a thread API port that is not a port taken as one.
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
    assert_eq!(s.thread_api_port, None);
    let s = settings().with_env(env(&[("LIVEKIT_THREAD_API_PORT", "7890")]));
    assert_eq!(s.thread_api_port, Some(7890));
    let s = settings().with_env(env(&[("LIVEKIT_THREAD_API_PORT", "not a port")]));
    assert_eq!(s.thread_api_port, None);
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
    let (worker, connection) = s.worker(false, vec![], Default::default());
    assert_eq!(worker.ws_url, "wss://env");
    assert_eq!(connection.server_url, "https://env:7883");
    assert_eq!(connection.worker_token.as_deref(), Some("hosted"));
    assert_eq!((worker.agent_name.as_str(), worker.deployment.as_str()), ("from-env", "staging"));
    let s = settings().with_env(env(&[("LIVEKIT_URL", "wss://env"), ("LIVEKIT_WORKER_URL", "")]));
    let (worker, connection) = s.worker(false, vec![], Default::default());
    assert_eq!(connection.server_url, "wss://env", "the worker dials ws_url by default");
    assert_eq!(connection.worker_token, None);
    assert!(worker.deployment.is_empty());
}

// Reads settings with a worker token alone, and with a key, a secret or an empty token. Catches
// token mode without a worker token or with a key or secret, and a registration check that asks
// for a key in token mode. Also catches job processes that are not told about token mode, or key
// mode job processes that are told, since they would ask the server for tokens they can sign.
#[test]
fn token_mode_is_a_worker_token_without_a_key_and_its_jobs_are_told() {
    let s = settings().with_env(env(&[("LIVEKIT_URL", "wss://x"), ("LIVEKIT_TOKEN", "wt")]));
    assert!(s.token_mode());
    s.check_registration().unwrap();
    assert!(s.child_env().contains(&("LIVEKIT_TOKEN", "wt".to_owned())));
    let token = std::sync::Arc::new(parking_lot::Mutex::new("wt".to_owned()));
    let (_, connection) = s.worker(false, vec![], token);
    assert_eq!(*connection.token.lock(), "wt");
    assert!(connection.api_key.is_empty() && connection.api_secret.is_empty());

    for keyed in [
        &[("LIVEKIT_TOKEN", "wt"), ("LIVEKIT_API_KEY", "k"), ("LIVEKIT_API_SECRET", "s")][..],
        &[("LIVEKIT_TOKEN", "wt"), ("LIVEKIT_API_KEY", "k")][..],
        &[("LIVEKIT_TOKEN", "")][..],
    ] {
        let vars: Vec<_> = keyed.to_vec();
        let s = settings().with_env(move |name: &str| {
            vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned())
        });
        assert!(!s.token_mode(), "{keyed:?}");
        assert!(s.child_env().iter().all(|(name, _)| *name != "LIVEKIT_TOKEN"), "{keyed:?}");
    }
}

// Catches: a missing credential reported as another, or routes allowed without an agent name.
#[test]
fn registration_needs_the_url_the_credentials_and_a_name_for_routes() {
    let mut s = settings();
    assert!(s.check_registration().unwrap_err().to_string().starts_with("ws_url"));
    s.ws_url = "wss://x".into();
    assert!(s.check_registration().unwrap_err().to_string().starts_with("api_key"));
    s.api_key = "k".into();
    assert!(s.check_registration().unwrap_err().to_string().starts_with("api_secret"));
    s.api_secret = "s".into();
    assert!(s.check_registration().is_ok());
    assert!(s.check_routes(false).is_ok());
    assert!(s.check_routes(true).unwrap_err().to_string().contains("agent_name"));
}

// Catches: a listener left out not keeping its default, or development binding every interface.
#[test]
fn listeners_default_per_mode() {
    use lk_agents_abi::Maybe::Given;
    use lk_agents_abi::worker::types::{ListenerAddress, Listeners, PublicListener, ServerOptions};
    let mut s = settings();
    let listeners = Listeners {
        public: Some(PublicListener { port: Some(9000), ..Default::default() }),
        prometheus: Some(ListenerAddress::default()),
        ..Default::default()
    };
    s.apply(
        ServerOptions { listeners: Given(listeners), ..Default::default() },
        Default::default(),
    )
    .unwrap();
    let l = &s.listeners;
    let at = |b: Bind| (b.host, b.port);
    assert_eq!(at(Bind::public(l, false)), ("0.0.0.0".into(), 9000));
    assert_eq!(at(Bind::health(l, false)), ("0.0.0.0".into(), 8081));
    assert_eq!(at(Bind::health(l, true)), ("127.0.0.1".into(), 0));
    assert_eq!(Bind::prometheus(l, false).map(|b| b.port), Some(9090));
    let off = Bind::prometheus(&settings().listeners, false);
    assert_eq!(off, None, "metrics are off unless given");
}

// Catches: a development worker delegating to production, or a self-hosted worker calling the
// LiveKit server, which does not serve the agent's endpoints.
#[test]
fn delegation_follows_who_routes_the_endpoints() {
    let mut s = settings();
    s.ws_url = "wss://proj.livekit.cloud".into();
    s.agent_name = "dentist".into();
    s.deployment = "dev-123".into();
    assert_eq!(
        s.delegation_url(&Serving::Cloud, "backend").unwrap(),
        "https://proj.livekit.cloud/agents/dentist/dev-123/backend"
    );
    let local = Serving::Local { base: "http://127.0.0.1:5000/agents/dentist/_".into() };
    assert_eq!(
        s.delegation_url(&local, "backend").unwrap(),
        "http://127.0.0.1:5000/agents/dentist/_/backend"
    );
    assert_eq!(
        s.endpoint_url(&local, "backend").as_deref(),
        Some("/agents/dentist/dev-123/backend"),
        "the card puts the path on the address the request reached"
    );
    assert!(s.delegation_url(&Serving::Unknown, "backend").is_err());
    // a job process inherits what its server resolved
    let job = settings().with_env(env(&[(
        "LIVEKIT_AGENT_ENDPOINTS_URL",
        "http://127.0.0.1:5000/agents/dentist/_",
    )]));
    assert_eq!(
        job.delegation_url(&Serving::Unknown, "backend").unwrap(),
        "http://127.0.0.1:5000/agents/dentist/_/backend"
    );
    assert!(job.child_env().iter().any(|(name, _)| *name == "LIVEKIT_AGENT_ENDPOINTS_URL"));
    s.worker_token = "hosted".into();
    assert!(s.delegation_url(&Serving::Unknown, "backend").is_ok(), "hosted on Cloud");
}

#[test]
fn invalid_server_update_cannot_partially_change_launch_configuration() {
    use lk_agents_abi::Maybe::Given;
    use lk_agents_abi::worker::types::{ProcessOptions, ServerOptions};
    let mut server = settings();
    let original_url = server.ws_url.clone();
    let error = server
        .apply(
            ServerOptions { ws_url: Given("wss://new.example".into()), ..Default::default() },
            ProcessOptions { initialize_process_timeout: Given(f64::NAN), ..Default::default() },
        )
        .unwrap_err();
    assert!(error.to_string().contains("initialize_process_timeout"));
    assert_eq!(server.ws_url, original_url, "a refused update changes no option");
    server
        .apply(
            ServerOptions {
                ws_url: Given("wss://new.example".into()),
                drain_timeout: Given(0.0),
                ..Default::default()
            },
            ProcessOptions { initialize_process_timeout: Given(4.0), ..Default::default() },
        )
        .unwrap();
    assert_eq!(server.ws_url, "wss://new.example");
    assert_eq!(server.initialize_process_timeout, 4.0);
    assert_eq!(server.drain_timeout, 0.0, "zero skips waiting for natural completion");
    assert_eq!(server.shutdown_process_timeout, 10.0, "omission preserves other fields");
}

#[test]
fn simulation_is_resolved_before_constructing_the_worker_load_policy() {
    let mut server = settings();
    let (worker, _) = server.worker(false, vec![], Default::default());
    assert_eq!(worker.load_threshold.get(false), crate::config::DEFAULT_LOAD_THRESHOLD_PROD);
    server.simulation = true;
    let (worker, _) = server.worker(false, vec![], Default::default());
    assert!(worker.load_threshold.get(false).is_infinite());
}

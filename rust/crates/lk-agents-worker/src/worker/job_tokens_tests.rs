#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use livekit_protocol::worker_message::Message as Sent;
use livekit_protocol::{self as protocol, ServerMessage, server_message};
use lk_agents_abi::SpawnerExt as _;
use parking_lot::Mutex;

use crate::transport::{Connected, TransportError, WorkerTransport};
use crate::worker::tests::*;
use crate::*;

const CLAIMS: &str = r#"{"identity":"avatar","video":{"roomJoin":true,"room":"r"}}"#;

type Run = lk_agents_abi::OwnedTask<Result<(), RuntimeError>>;

/// Starts a worker loop and waits until it is registered.
async fn registered()
-> (WorkerClient, Peer, tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>, Run) {
    let RuntimeRig { mut runtime, handle, peer, sent, .. } = runtime_rig();
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let spawner = lk_agents_net::task::TokioSpawner::current();
    let task = spawner.spawn(async move { runtime.run().await });
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}
    (handle, peer, sent, task)
}

async fn close(handle: WorkerClient, task: Run) {
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
}

async fn next_request(
    sent: &mut tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
) -> protocol::CreateJobTokenRequest {
    let message = until_sent(sent, |m| matches!(m.message, Some(Sent::CreateJobToken(_)))).await;
    let Some(Sent::CreateJobToken(request)) = message.message else { unreachable!() };
    request
}

fn answer(request_id: &str, token: &str, error: &str) -> Result<ServerMessage, TransportError> {
    let expires_at =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as i64 + 600;
    Ok(ServerMessage {
        message: Some(server_message::Message::CreateJobToken(protocol::CreateJobTokenResponse {
            request_id: request_id.into(),
            token: token.into(),
            expires_at,
            error: error.into(),
        })),
    })
}

fn ask(
    handle: &WorkerClient,
    claims: &'static str,
) -> lk_agents_abi::OwnedTask<Result<lk_agents_net::job_token::Issued, JobTokenError>> {
    let handle = handle.clone();
    let spawner = lk_agents_net::task::TokioSpawner::current();
    spawner.spawn(async move { handle.create_job_token("AJ_1", claims).await })
}

// One job sends two token requests, and the server answers them in reverse order. Catches an
// answer that goes to the wrong request, or a request without its job id or claims.
#[tokio::test(start_paused = true)]
async fn a_job_token_request_is_answered_by_its_request_id() {
    let (handle, peer, mut sent, task) = registered().await;
    let first = ask(&handle, CLAIMS);
    let second = ask(&handle, r#"{"video":{"roomAdmin":true,"room":"r"}}"#);
    let (a, b) = (next_request(&mut sent).await, next_request(&mut sent).await);
    assert_eq!((a.job_id.as_str(), b.job_id.as_str()), ("AJ_1", "AJ_1"));
    let (a, b) = if a.claims == CLAIMS { (a, b) } else { (b, a) };
    assert_ne!(a.request_id, b.request_id);
    peer.send(answer(&b.request_id, "signed-b", "")).unwrap();
    peer.send(answer(&a.request_id, "signed-a", "")).unwrap();
    let issued = first.await.unwrap().unwrap();
    assert_eq!(issued.token, "signed-a");
    assert!(issued.expires_at > 0);
    assert_eq!(second.await.unwrap().unwrap().token, "signed-b");
    close(handle, task).await;
}

// The server refuses a request with a reason. Catches a refusal that is read as a token, or an
// error that loses the server's reason.
#[tokio::test(start_paused = true)]
async fn a_refused_job_token_carries_the_servers_reason() {
    let (handle, peer, mut sent, task) = registered().await;
    let asking = ask(&handle, CLAIMS);
    let request = next_request(&mut sent).await;
    peer.send(answer(&request.request_id, "", "job not on this worker")).unwrap();
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.0.contains("refused") && error.0.contains("job not on this worker"), "{error}");
    close(handle, task).await;
}

// The server never answers a request. Catches a job that waits forever, for example on a
// server that is older than token mode.
#[tokio::test(start_paused = true)]
async fn an_unanswered_job_token_request_times_out() {
    let (handle, _peer, mut sent, task) = registered().await;
    let started = tokio::time::Instant::now();
    let asking = ask(&handle, CLAIMS);
    next_request(&mut sent).await;
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.0.contains("did not answer"), "{error}");
    assert_eq!(started.elapsed(), JOB_TOKEN_TIMEOUT);
    close(handle, task).await;
}

// A job asks for a token before registration, and again just before the connection drops.
// Catches a request that waits through a lost connection, because its answer can only come on
// that connection. Also catches a request sent before registration, which would only time
// out.
#[tokio::test(start_paused = true)]
async fn job_tokens_need_a_registered_connection_and_fail_when_it_drops() {
    let RuntimeRig { mut runtime, handle, peer, mut sent, .. } = runtime_rig_with_retry(5);
    let spawner = lk_agents_net::task::TokioSpawner::current();
    let task = spawner.spawn(async move { runtime.run().await });
    until_sent(&mut sent, |m| matches!(m.message, Some(Sent::Register(_)))).await;
    let started = tokio::time::Instant::now();
    let error = handle.create_job_token("AJ_1", CLAIMS).await.unwrap_err();
    assert!(error.0.contains("not connected"), "{error}");
    assert!(started.elapsed() < JOB_TOKEN_TIMEOUT);

    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}
    let asking = ask(&handle, CLAIMS);
    next_request(&mut sent).await;
    peer.send(Err(TransportError::Failed("peer disconnected".into()))).unwrap();
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.0.contains("lost"), "{error}");
    close(handle, task).await;
}

fn kept_job(id: &str) -> RunningJobInfo {
    RunningJobInfo {
        accept_arguments: AcceptArguments { identity: "agent-7".into(), ..Default::default() },
        job: protocol::Job { id: id.into(), ..Default::default() },
        url: "wss://sfu".into(),
        token: "kept".into(),
        worker_id: "old-worker".into(),
        fake_job: false,
        received_at: None,
        accepted_at: None,
        assigned_at: None,
        launched_at: 0.0,
    }
}

// The worker adopts a kept job, asks for its token, and then reconnects. Catches these bugs:
// - the token request goes out before the `MigrateJobRequest`;
// - the migrated job has no accepted identity;
// - the worker refuses the job's status because it does not know the job;
// - the job is not reported again after the reconnect.
#[tokio::test(start_paused = true)]
async fn adopted_jobs_are_reported_before_their_tokens_and_after_a_reconnect() {
    let RuntimeRig { mut runtime, handle, peer, mut sent, connections, .. } =
        runtime_rig_with_retry(1);
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let spawner = lk_agents_net::task::TokioSpawner::current();
    let task = spawner.spawn(async move { runtime.run().await });
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}

    handle.send(WorkerCommand::AdoptJobs { jobs: vec![kept_job("AJ_1")] }).unwrap();
    let asking = ask(&handle, CLAIMS);
    let mut order = Vec::new();
    while order.len() < 2 {
        match sent.recv().await.unwrap().message {
            Some(Sent::MigrateJob(migrated)) => {
                let [job] = migrated.jobs.as_slice() else { panic!("{migrated:?}") };
                assert_eq!(job.id, "AJ_1");
                let identity = job.state.as_ref().map(|s| s.participant_identity.as_str());
                assert_eq!(identity, Some("agent-7"));
                order.push("migrate");
            }
            Some(Sent::CreateJobToken(request)) => {
                peer.send(answer(&request.request_id, "signed", "")).unwrap();
                order.push("token");
            }
            _ => {}
        }
    }
    assert_eq!(order, ["migrate", "token"]);
    asking.await.unwrap().unwrap();
    assert_eq!(handle.snapshot().active_jobs, 1);

    handle
        .send(WorkerCommand::JobStatus {
            job_id: "AJ_1".into(),
            status: protocol::JobStatus::JsRunning,
        })
        .unwrap();
    until_sent(&mut sent, |m| matches!(&m.message, Some(Sent::UpdateJob(u)) if u.job_id == "AJ_1"))
        .await;

    let (next, incoming) = tokio::sync::mpsc::unbounded_channel();
    next.send(register_message("worker")).unwrap();
    connections.lock().push_back((next.downgrade(), incoming));
    peer.send(Err(TransportError::Failed("lost".into()))).unwrap();
    let migrated = until_sent(&mut sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    let Some(Sent::MigrateJob(migrated)) = migrated.message else { unreachable!() };
    assert_eq!(migrated.job_ids, ["AJ_1"]);
    close(handle, task).await;
}

/// A transport that records the headers of each connect. After each connect it replaces the
/// worker token, as the CLI's `Credentials` message does.
struct Refreshing {
    inner: ScriptedTransport,
    token: Option<WorkerToken>,
}

#[async_trait::async_trait]
impl WorkerTransport for Refreshing {
    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        let connected = self.inner.connect(url, worker_token, headers).await;
        if let Some(token) = &self.token {
            *token.lock() = "refreshed-worker-token".into();
        }
        connected
    }
}

/// A register response from a server that reports `agent_protocol`.
fn register_on(worker_id: &str, agent_protocol: i32) -> Result<ServerMessage, TransportError> {
    let server_info = protocol::ServerInfo { agent_protocol, ..Default::default() };
    Ok(ServerMessage {
        message: Some(server_message::Message::Register(protocol::RegisterWorkerResponse {
            worker_id: worker_id.to_owned(),
            server_info: Some(server_info),
            ..Default::default()
        })),
    })
}

fn bearers(cfg: RuntimeConfig, refresh: bool) -> Vec<Option<String>> {
    let wire = Arc::new(Mutex::new(Wire::default()));
    let steps = vec![
        Step::Serves(vec![
            register_on("W-1", JOB_TOKEN_AGENT_PROTOCOL),
            Err(TransportError::Failed("peer disconnected".into())),
        ]),
        Step::Serves(vec![register_on("W-2", JOB_TOKEN_AGENT_PROTOCOL)]),
    ];
    let inner = ScriptedTransport { script: Mutex::new(steps.into()), wire: wire.clone() };
    let transport = Refreshing { inner, token: refresh.then(|| cfg.token.clone()) };
    let runtime =
        tokio::runtime::Builder::new_current_thread().enable_all().start_paused(true).build();
    runtime.unwrap().block_on(async {
        let (mut worker, _handle) = Worker::with_components(
            Arc::new(transport),
            cfg,
            WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() },
            Arc::new(LoadHost::default()),
            rt(Arc::new(TestClock(tokio::time::Instant::now()))),
            blocking_io(),
        );
        let _ = tokio::time::timeout(Duration::from_millis(50), worker.run()).await;
    });
    let headers = wire.lock().headers.clone();
    headers
        .iter()
        .map(|headers| {
            headers.iter().find(|(name, _)| name == "Authorization").map(|(_, v)| v.clone())
        })
        .collect()
}

// Registers twice in token mode, in key mode, and with no credentials. Catches these bugs:
// - in token mode, the worker signs, drops or reuses an old worker token, instead of sending the
//   current worker token on each registration;
// - in key mode, the worker token wins over the key and secret;
// - a bearer is sent when there are no credentials.
#[test]
fn token_mode_registers_with_the_current_worker_token_and_a_key_pair_wins() {
    let token: WorkerToken = Arc::new(Mutex::new("lk-worker-token".into()));
    let cfg = RuntimeConfig { token, ..runtime_cfg(5) };
    assert_eq!(
        bearers(cfg, true),
        [Some("Bearer lk-worker-token".into()), Some("Bearer refreshed-worker-token".into())]
    );

    let token: WorkerToken = Arc::new(Mutex::new("lk-worker-token".into()));
    let keyed = RuntimeConfig {
        api_key: "APIkey".into(),
        api_secret: "thiskeyistotallyunsafe".into(),
        token,
        ..runtime_cfg(5)
    };
    for bearer in bearers(keyed, true) {
        let jwt = bearer.as_deref().and_then(|b| b.strip_prefix("Bearer ")).expect("a bearer");
        // Checks that the key pair signed the token.
        lk_agents_net::token::extend_job_token(jwt, "APIkey", "thiskeyistotallyunsafe")
            .expect("signed with the key pair");
    }

    assert_eq!(bearers(runtime_cfg(5), false), [None, None]);
}

/// Runs a worker against servers that report `agent_protocol`.
///
/// Returns the outcome of the run, or `None` if it is still running, and the number of connect
/// attempts.
fn register_against(
    cfg: RuntimeConfig,
    agent_protocol: i32,
) -> (Option<Result<(), RuntimeError>>, u32) {
    let wire = Arc::new(Mutex::new(Wire::default()));
    let answer = || vec![register_on("W-1", agent_protocol)];
    let steps = vec![Step::Serves(answer()), Step::Serves(answer())];
    let transport = ScriptedTransport { script: Mutex::new(steps.into()), wire: wire.clone() };
    let runtime =
        tokio::runtime::Builder::new_current_thread().enable_all().start_paused(true).build();
    let outcome = runtime.unwrap().block_on(async {
        let (mut worker, _handle) = Worker::with_components(
            Arc::new(transport),
            cfg,
            WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() },
            Arc::new(LoadHost::default()),
            rt(Arc::new(TestClock(tokio::time::Instant::now()))),
            blocking_io(),
        );
        tokio::time::timeout(Duration::from_millis(50), worker.run()).await.ok()
    });
    let attempts = wire.lock().attempts;
    (outcome, attempts)
}

// Registers in token mode and in key mode with servers that do not support job tokens. Catches
// a token mode worker that stays registered or retries, since its first job token request would
// close the connection. Also catches a key mode worker that the same check stops.
#[test]
fn token_mode_stops_on_a_server_without_job_tokens_and_a_key_pair_does_not() {
    let token: WorkerToken = Arc::new(Mutex::new("lk-worker-token".into()));
    let cfg = RuntimeConfig { token, ..runtime_cfg(5) };
    let (outcome, attempts) = register_against(cfg, 1);
    let error = outcome.expect("the worker stopped").unwrap_err();
    assert!(matches!(error, RuntimeError::Configuration(_)), "{error}");
    let message = error.to_string();
    assert!(message.contains("does not support token mode"), "{message}");
    assert!(message.contains("LIVEKIT_API_KEY and LIVEKIT_API_SECRET"), "{message}");
    assert_eq!(attempts, 1, "not retried");

    let token: WorkerToken = Arc::new(Mutex::new("lk-worker-token".into()));
    let keyed = RuntimeConfig {
        api_key: "APIkey".into(),
        api_secret: "thiskeyistotallyunsafe".into(),
        token,
        ..runtime_cfg(5)
    };
    let (outcome, attempts) = register_against(keyed, 0);
    assert!(outcome.is_none(), "a key pair stays registered: {outcome:?}");
    assert_eq!(attempts, 1);
}

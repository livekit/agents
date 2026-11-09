#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use super::*;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::ipc_host::{JobLauncher, JobLaunchers};
use lk_agents_abi::worker::server_host::ServerHost;
use lk_agents_abi::worker::types::ProcessStartOptions;

use super::settings::tests::settings;
use crate::RunningJobInfo;
use host::{JobPool, PoolFactory};

#[derive(Clone)]
struct Host {
    started: Arc<tokio::sync::Notify>,
    joining: Arc<tokio::sync::Notify>,
    closing: Arc<tokio::sync::Notify>,
    release_close: Arc<tokio::sync::Semaphore>,
    /// Holds the host's start until released.
    start_gate: Option<Arc<tokio::sync::Semaphore>>,
    fail_start: bool,
    /// The host's start panics.
    panic_start: bool,
}

impl Host {
    fn new() -> Self {
        Self {
            started: Arc::default(),
            joining: Arc::default(),
            closing: Arc::default(),
            release_close: Arc::new(tokio::sync::Semaphore::new(0)),
            start_gate: None,
            fail_start: false,
            panic_start: false,
        }
    }
}

struct Launchers;

impl JobLaunchers for Launchers {
    fn create(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn JobLauncher>, HostError>> {
        async { Err("the fake pool launches nothing".into()) }.boxed()
    }
}

impl ServerHost for Host {
    fn load(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<f64, HostError>> {
        async { Ok(0.0) }.boxed()
    }
    fn start(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: ProcessStartOptions,
    ) -> BoxFuture<'_, Result<Option<Arc<dyn JobLaunchers>>, HostError>> {
        let (fail, gate, panics) = (self.fail_start, self.start_gate.clone(), self.panic_start);
        async move {
            if let Some(gate) = gate {
                let _permit = gate.acquire().await.unwrap();
            }
            assert!(!panics, "the host's start panicked");
            if fail {
                Err("pool warmup timed out".into())
            } else {
                Ok(Some(Arc::new(Launchers) as Arc<dyn JobLaunchers>))
            }
        }
        .boxed()
    }
    fn started(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        self.started.notify_one();
        async { Ok(()) }.boxed()
    }
    fn heartbeat(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        async { Ok(()) }.boxed()
    }
}

/// A pool whose drain never ends and whose close waits for the test.
struct Pool(Host);

#[async_trait::async_trait]
impl JobPool for Pool {
    async fn start(&self) -> Result<(), HostError> {
        Ok(())
    }
    async fn launch(&self, _: RunningJobInfo) -> Result<(), HostError> {
        Ok(())
    }
    async fn terminate(&self, _: &str) {}
    fn set_target_idle(&self, _: usize) {}
    fn active_jobs(&self) -> Vec<RunningJobInfo> {
        Vec::new()
    }
    async fn join(&self) {
        self.0.joining.notify_one();
        std::future::pending().await
    }
    async fn close(&self) {
        self.0.closing.notify_one();
        let _permit = self.0.release_close.acquire().await.unwrap();
    }
}

fn hosted(host: &Host) -> ServerCallbacks {
    let pool = host.clone();
    let pools: PoolFactory =
        Arc::new(move |_| Ok(Arc::new(Pool(pool.clone())) as Arc<dyn JobPool>));
    ServerCallbacks::new(Arc::new(host.clone()), pools)
}

fn options() -> ServerStart {
    ServerStart {
        settings: ServerSettings {
            simulation: true,
            num_idle_processes: Some(crate::EnvOption::Fixed(0)),
            ..Default::default()
        },
        devmode: true,
        unregistered: true,
        has_rtc: true,
        has_text_doors: true,
        has_load_fnc: false,
        endpoints: vec![],
        transport: crate::TransportConfig::WebSocket { heartbeat: crate::HEARTBEAT_INTERVAL },
        serving: Arc::default(),
        context: Default::default(),
        asyncio_debug: false,
    }
}

#[tokio::test]
async fn server_drain_close_callers_join_without_holding_a_lifecycle_lock() {
    let host = Host::new();
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    tokio::time::timeout(Duration::from_secs(1), host.started.notified()).await.unwrap();
    let first_drain = runtime.drain(None);
    let second_drain = runtime.drain(None);
    tokio::pin!(first_drain, second_drain);
    assert!(futures_util::poll!(&mut first_drain).is_pending());
    assert!(futures_util::poll!(&mut second_drain).is_pending());
    tokio::time::timeout(Duration::from_secs(1), host.joining.notified()).await.unwrap();
    let first_close = runtime.close();
    let second_close = runtime.close();
    tokio::pin!(first_close, second_close);
    assert!(futures_util::poll!(&mut first_close).is_pending());
    assert!(futures_util::poll!(&mut second_close).is_pending());
    tokio::time::timeout(Duration::from_secs(1), host.closing.notified()).await.unwrap();
    host.release_close.add_permits(1);
    assert_eq!(first_close.await, Ok(()));
    assert_eq!(second_close.await, Ok(()));
    assert_eq!(run.await, Ok(()));
    assert_eq!(first_drain.await, Err(DrainError::Closed));
    assert_eq!(second_drain.await, Err(DrainError::Closed));
    assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
}

#[tokio::test]
async fn a_drain_asked_during_startup_starts_once_the_server_runs() {
    let mut host = Host::new();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    host.start_gate = Some(gate.clone());
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    assert_eq!(runtime.snapshot().phase, ServerPhase::Starting);
    let drain = runtime.drain(None);
    tokio::pin!(drain);
    assert!(futures_util::poll!(&mut drain).is_pending());
    let joining = host.joining.notified();
    tokio::pin!(joining);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut joining).await.is_err(),
        "the drain began before the server ran"
    );
    gate.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), &mut joining).await.unwrap();
    assert_eq!(runtime.snapshot().phase, ServerPhase::Draining);
    assert_eq!(runtime.close().await, Ok(()));
    assert_eq!(drain.await, Err(DrainError::Closed));
    assert_eq!(run.await, Ok(()));
}

// Catches: a panic in the lifecycle task leaving run, drain and close waiting on outcomes only
// that task could settle.
#[tokio::test]
async fn a_panicking_startup_still_ends_run_drain_and_close() {
    let mut host = Host::new();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    host.start_gate = Some(gate.clone());
    host.panic_start = true;
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    let drain = runtime.drain(None);
    tokio::pin!(drain);
    assert!(futures_util::poll!(&mut drain).is_pending());
    gate.add_permits(1);
    let limit = Duration::from_secs(5);
    assert!(tokio::time::timeout(limit, run).await.expect("run hung").is_err());
    assert!(tokio::time::timeout(limit, drain).await.expect("drain hung").is_err());
    assert!(tokio::time::timeout(limit, runtime.close()).await.expect("close hung").is_err());
}

#[tokio::test]
async fn failed_registration_is_reported_by_run_and_close_joins_cleanup() {
    let host = Host::new();
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let mut start = options();
    start.unregistered = false;
    start.settings.ws_url = "ws://127.0.0.1:1".into();
    start.settings.api_key = "devkey".into();
    start.settings.api_secret = "devsecret".into();
    start.settings.max_retry = 0;
    let failure = tokio::time::timeout(Duration::from_secs(10), runtime.run(start))
        .await
        .unwrap()
        .unwrap_err();
    assert!(failure.to_string().contains("connection failed after 0 retries"), "{failure}");
    assert_eq!(runtime.close().await, Ok(()));
    assert_eq!(runtime.close().await, Ok(()));
    assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
}

#[tokio::test]
async fn server_startup_lifecycle_reports_warmup_failure_and_joins_cleanup() {
    let mut host = Host::new();
    host.fail_start = true;
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    assert_eq!(runtime.run(options()).await, Err("pool warmup timed out".into()));
    assert_eq!(runtime.close().await, Ok(()));
    assert_eq!(runtime.close().await, Ok(()));
    assert!(host.started.notified().now_or_never().is_none());
    assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
}

#[tokio::test]
async fn server_cleanup_deadline_returns_while_the_actor_retains_the_pool() {
    let host = Host::new();
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let mut start = options();
    start.settings.shutdown_process_timeout = 0.02;
    let run = runtime.run(start);
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    tokio::time::timeout(Duration::from_secs(1), host.started.notified()).await.unwrap();
    assert_eq!(runtime.close().await, Ok(()));
    let snapshot = runtime.snapshot();
    assert_eq!(snapshot.phase, ServerPhase::Closing);
    assert_eq!(snapshot.incomplete_cleanup, vec![ServerResource::Processes]);
    assert_eq!(run.await, Ok(()));
    host.release_close.add_permits(1);
    let closed = runtime.snapshot.wait_until(|s| (s.phase == ServerPhase::Closed).then_some(()));
    tokio::time::timeout(Duration::from_secs(1), closed).await.unwrap();
}
#[tokio::test(start_paused = true)]
async fn server_startup_waits_for_sampling_capacity() {
    let blocking_io = blocking_io();
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(blocking_io.try_admit(()).unwrap());
    }
    let host = Host::new();
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io.clone()));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    tokio::time::sleep(Duration::from_millis(1)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(futures_util::poll!(&mut run).is_pending(), "sampling must not reject startup");
    drop(held);
    host.started.notified().await;
    runtime.close().await.unwrap();
    run.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn close_interrupts_a_server_waiting_for_sampling_capacity() {
    let blocking_io = blocking_io();
    let held: Vec<_> = (0..4).map(|_| blocking_io.try_admit(()).unwrap()).collect();
    let host = Host::new();
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    tokio::time::sleep(Duration::from_millis(1)).await;
    runtime.close().await.unwrap();
    run.await.unwrap();
    assert!(host.started.notified().now_or_never().is_none());
    assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
    drop(held);
}

#[tokio::test(start_paused = true)]
async fn server_setup_capacity_timeout_keeps_close_successful() {
    let blocking_io = blocking_io();
    let held: Vec<_> = (0..4).map(|_| blocking_io.try_admit(()).unwrap()).collect();
    let host = Host::new();
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io));
    let mut start = options();
    start.settings.initialize_process_timeout = 0.01;
    assert_eq!(runtime.run(start).await, Err("server setup capacity wait timed out".into()));
    runtime.close().await.unwrap();
    assert!(host.started.notified().now_or_never().is_none());
    assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
    drop(held);
}

// Each test owns its blocking service so saturation cannot affect other tests.
fn blocking_io() -> lk_agents_net::blocking_io::BlockingIo {
    lk_agents_net::blocking_io::BlockingIo::with_limits("lk-test-io", 4, &[4]).unwrap()
}

// Catches: a text-only server launching a process for a job, or a default accept that differs
// from the agent's identity.
#[test]
fn a_server_takes_a_job_only_with_a_voice_door() {
    let job = proto::Job { id: "AJ_1".into(), ..Default::default() };
    let req = JobRequest { job, resuming: false };
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
    let fake = simulated_job(
        &s,
        "W1",
        "console-room",
        SimulatedJobOptions {
            fake_job: true,
            agent_identity: Some("console".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fake.job.id.starts_with("mock-job-"));
    assert_eq!(fake.job.room.as_ref().map(|r| r.name.as_str()), Some("console-room"));
    assert_eq!((fake.accept_arguments.identity.as_str(), fake.token.as_str()), ("console", ""));
    assert_eq!(fake.worker_id, "W1");
    let real = simulated_job(
        &s,
        "W1",
        "r",
        SimulatedJobOptions { agent_identity: Some("a".into()), ..Default::default() },
    )
    .unwrap_err();
    assert_eq!(real.to_string(), "a dispatched job requires room information");
    let mut creds = settings();
    creds.api_key = "APIkey".into();
    creds.api_secret = "thiskeyistotallyunsafe".into();
    let minted = simulated_job(
        &creds,
        "W",
        "r",
        SimulatedJobOptions { fake_job: true, ..Default::default() },
    )
    .unwrap();
    assert!(minted.accept_arguments.identity.starts_with("fake-agent-"));
    let named = simulated_job(
        &creds,
        "W",
        "r",
        SimulatedJobOptions {
            fake_job: true,
            agent_identity: Some("x".into()),
            token: Some(minted.token),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(named.accept_arguments.identity, minted.accept_arguments.identity);
}

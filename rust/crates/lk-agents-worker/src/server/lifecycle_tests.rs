use super::*;
use lk_agents_abi::logging::LogWriter;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::server_host::ServerHost;
use lk_agents_abi::worker::types::ProcessStartOptions;

use crate::RunningJobInfo;

#[derive(Clone)]
struct Host {
    started: Arc<tokio::sync::Notify>,
    joining: Arc<tokio::sync::Notify>,
    closing: Arc<tokio::sync::Notify>,
    release_close: Arc<tokio::sync::Semaphore>,
    /// Holds the host's start until released.
    start_gate: Option<Arc<tokio::sync::Semaphore>>,
    fail_start: bool,
    fail_close: bool,
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
            fail_close: false,
        }
    }
}

impl ServerHost for Host {
    fn launch_job(&self, _: RunningJobInfo) -> BoxFuture<'_, Result<String, HostError>> {
        async { Ok("job".into()) }.boxed()
    }
    fn terminate_job(&self, _: String) -> BoxFuture<'_, Result<(), HostError>> {
        async { Ok(()) }.boxed()
    }
    fn load(&self) -> BoxFuture<'_, Result<f64, HostError>> {
        async { Ok(0.0) }.boxed()
    }
    fn start(&self, _: ProcessStartOptions) -> BoxFuture<'_, Result<(), HostError>> {
        let (fail, gate) = (self.fail_start, self.start_gate.clone());
        async move {
            if let Some(gate) = gate {
                let _permit = gate.acquire().await.unwrap();
            }
            if fail { Err("pool warmup timed out".into()) } else { Ok(()) }
        }
        .boxed()
    }
    fn started(&self) -> BoxFuture<'_, Result<(), HostError>> {
        self.started.notify_one();
        async { Ok(()) }.boxed()
    }
    fn idle_processes(&self, _: u32) -> BoxFuture<'_, Result<(), HostError>> {
        async { Ok(()) }.boxed()
    }
    fn worker_load(&self, _: f64) -> BoxFuture<'_, Result<(), HostError>> {
        async { Ok(()) }.boxed()
    }
    fn join_jobs(&self) -> BoxFuture<'_, Result<(), HostError>> {
        self.joining.notify_one();
        std::future::pending().boxed()
    }
    fn close(&self) -> BoxFuture<'_, Result<(), HostError>> {
        async {
            self.closing.notify_one();
            let _permit = self.release_close.acquire().await.unwrap();
            if self.fail_close { Err("host close failed".into()) } else { Ok(()) }
        }
        .boxed()
    }
    fn active_jobs(&self) -> BoxFuture<'_, Result<usize, HostError>> {
        async { Ok(0) }.boxed()
    }
    fn metrics(&self) -> BoxFuture<'_, Result<(String, String), HostError>> {
        async { Err("no metrics".into()) }.boxed()
    }
}

fn hosted(host: &Host) -> ServerCallbacks {
    ServerCallbacks::new(Arc::new(host.clone()), LogWriter::disabled())
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
        has_rtc: false,
        has_text_doors: true,
        has_load_fnc: false,
        endpoints: vec![],
        transport: crate::TransportConfig::WebSocket { heartbeat: crate::HEARTBEAT_INTERVAL },
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
    assert!(failure.contains("connection failed after 0 retries"), "{failure}");
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
async fn server_shutdown_resources_reports_host_close_failure_to_every_caller() {
    let mut host = Host::new();
    host.fail_close = true;
    host.release_close.add_permits(1);
    let runtime = ServerRuntime::new(hosted(&host), Some(blocking_io()));
    let run = runtime.run(options());
    tokio::pin!(run);
    assert!(futures_util::poll!(&mut run).is_pending());
    tokio::time::timeout(Duration::from_secs(1), host.started.notified()).await.unwrap();
    assert_eq!(runtime.close().await, Err("Host: host close failed".into()));
    assert_eq!(run.await, Err("Host: host close failed".into()));
    assert_eq!(runtime.close().await, Err("Host: host close failed".into()));
}

#[tokio::test]
async fn server_cleanup_deadline_returns_while_the_actor_retains_the_host() {
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
    assert_eq!(snapshot.incomplete_cleanup, vec![ServerResource::Host]);
    assert_eq!(run.await, Ok(()));
    host.release_close.add_permits(1);
    let mut snapshots = runtime.snapshot.subscribe();
    tokio::time::timeout(
        Duration::from_secs(1),
        snapshots.wait_for(|snapshot| snapshot.phase == ServerPhase::Closed),
    )
    .await
    .unwrap()
    .unwrap();
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

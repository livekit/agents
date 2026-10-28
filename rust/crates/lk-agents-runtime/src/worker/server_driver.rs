//! Server execution adapter. Policy is in `server_lifecycle`; these resources move with runtime.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};

use tokio::sync::{Mutex, watch};

use crate::worker::server::{self, ServerSettings, ServerStatus};
use crate::worker::server_lifecycle::{
    ServerEffect, ServerInput, ServerLifecycle, ServerPhase, ServerResource,
};
use crate::worker::{
    Worker, WorkerCommand, WorkerHandle, WorkerHost, WorkerNotification, WorkerRegistered,
};

#[async_trait]
pub(crate) trait ServerHost: WorkerHost + Clone + Send + Sync + 'static {
    fn configured(&self, has_rtc: bool, has_on_request: bool, user_load: bool) -> Self;
    async fn start(
        &self,
        settings: ServerSettings,
        devmode: bool,
        has_rtc: bool,
    ) -> Result<(), String>;
    async fn started(&self) -> Result<(), String>;
    async fn registered(&self, info: WorkerRegistered) -> Result<(), String>;
    async fn idle_processes(&self, target: u32) -> Result<(), String>;
    async fn worker_load(&self, load: f32) -> Result<(), String>;
    async fn close_websockets(&self) -> Result<(), String>;
    async fn join_jobs(&self) -> Result<(), String>;
    async fn close(&self) -> Result<(), String>;
    async fn active_jobs(&self) -> usize;
    async fn metrics(&self) -> Option<(String, String)>;
}

pub(crate) struct ServerStart {
    pub settings: ServerSettings,
    pub devmode: bool,
    pub unregistered: bool,
    pub has_rtc: bool,
    pub has_text_doors: bool,
    pub has_load_fnc: bool,
    pub has_on_request: bool,
    pub publisher: bool,
    pub endpoints: Vec<crate::worker::protocol::AgentEndpoint>,
    pub transport: crate::worker::TransportConfig,
    pub worker_url: Option<String>,
}

#[derive(Clone)]
pub(crate) struct ServerSnapshot {
    pub phase: ServerPhase,
    pub worker_id: String,
    pub http_port: u16,
    pub worker: Option<WorkerHandle>,
    pub has_rtc: bool,
    pub incomplete_cleanup: Vec<ServerResource>,
}

#[derive(Clone, Default)]
struct Control {
    close: bool,
    drain: Option<Option<Duration>>,
}

type Outcome = watch::Sender<Option<Result<(), String>>>;

#[derive(Clone)]
pub(crate) struct ServerRuntime<H: ServerHost> {
    host: H,
    workers: Option<lk_agents_media::workers::MediaWorkers>,
    control: watch::Sender<Control>,
    receiver: Arc<Mutex<Option<watch::Receiver<Control>>>>,
    snapshot: watch::Sender<ServerSnapshot>,
    outcome: Outcome,
    cleanup_outcome: Outcome,
    drain_outcome: Outcome,
}

impl<H: ServerHost> ServerRuntime<H> {
    pub(crate) fn new(host: H, workers: Option<lk_agents_media::workers::MediaWorkers>) -> Self {
        let (control, receiver) = watch::channel(Control::default());
        Self {
            host,
            workers,
            control,
            receiver: Arc::new(Mutex::new(Some(receiver))),
            snapshot: watch::channel(ServerSnapshot {
                phase: ServerPhase::Idle,
                worker_id: "unregistered".into(),
                http_port: 0,
                worker: None,
                has_rtc: false,
                incomplete_cleanup: Vec::new(),
            })
            .0,
            outcome: watch::channel(None).0,
            cleanup_outcome: watch::channel(None).0,
            drain_outcome: watch::channel(None).0,
        }
    }

    pub(crate) fn snapshot(&self) -> ServerSnapshot {
        self.snapshot.borrow().clone()
    }

    pub(crate) async fn run(&self, start: ServerStart) -> Result<(), String> {
        let receiver = self.receiver.lock().await.take().ok_or("worker is already running")?;
        let origin = tokio::time::Instant::now();
        let timeout = Duration::try_from_secs_f64(start.settings.shutdown_process_timeout)
            .map_err(|_| "shutdown timeout must be finite and nonnegative")?;
        let process_cleanup = timeout * 3 + crate::worker::ipc::supervisor::KILL_GRACE;
        let connection_cleanup = if start.unregistered {
            Duration::ZERO
        } else {
            crate::worker::WorkerOptions::default().connection_timeout
        };
        let mut state = ServerLifecycle::new(process_cleanup.max(connection_cleanup));
        let effects = state.handle(ServerInput::Start, Duration::ZERO);
        self.publish(&state, effects);
        self.snapshot.send_modify(|snapshot| snapshot.has_rtc = start.has_rtc);
        let runtime = self.clone();
        tokio::spawn(async move { runtime.serve(start, receiver, state, origin).await });
        wait_outcome(&self.outcome).await
    }

    pub(crate) async fn simulate_job(
        &self,
        info: crate::worker::RunningJobInfo,
    ) -> Result<(), String> {
        if self.snapshot().phase != ServerPhase::Running {
            return Err("the server is not accepting jobs".into());
        }
        self.host.launch_job(info).await.map(|_| ()).map_err(|error| error.to_string())
    }

    pub(crate) async fn drain(&self, timeout: Option<Duration>) -> Result<(), String> {
        match self.snapshot().phase {
            ServerPhase::Idle | ServerPhase::Closed => return Ok(()),
            ServerPhase::Closing => return wait_outcome(&self.cleanup_outcome).await,
            _ => {}
        }
        self.control.send_if_modified(|control| {
            if control.drain.is_some() {
                return false;
            }
            control.drain = Some(timeout);
            true
        });
        wait_outcome(&self.drain_outcome).await
    }

    pub(crate) async fn close(&self) -> Result<(), String> {
        if self.snapshot().phase == ServerPhase::Idle {
            return Ok(());
        }
        self.control.send_if_modified(|control| {
            if control.close {
                return false;
            }
            control.close = true;
            true
        });
        wait_outcome(&self.cleanup_outcome).await
    }

    fn publish(&self, state: &ServerLifecycle, effects: Vec<ServerEffect>) {
        self.snapshot.send_modify(|snapshot| snapshot.phase = state.phase());
        for effect in effects {
            match effect {
                ServerEffect::DrainFinished(result) => {
                    self.drain_outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(result.clone());
                        true
                    });
                }
                ServerEffect::CloseFinished { run_result, cleanup_result } => {
                    self.cleanup_outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(cleanup_result.clone());
                        true
                    });
                    self.outcome.send_if_modified(|outcome| {
                        if outcome.is_some() {
                            return false;
                        }
                        *outcome = Some(run_result.clone());
                        true
                    });
                }
                ServerEffect::IncompleteCleanup { resources } => {
                    self.snapshot
                        .send_modify(|snapshot| snapshot.incomplete_cleanup = resources.clone());
                    crate::worker::diagnostics::emit(
                        log::Level::Error,
                        format!(
                            "server cleanup deadline missed; runtime retains ownership resources={resources:?}"
                        ),
                    );
                }
                _ => {}
            }
        }
    }

    async fn serve(
        self,
        start: ServerStart,
        mut control: watch::Receiver<Control>,
        mut state: ServerLifecycle,
        origin: tokio::time::Instant,
    ) {
        let mut expected = vec![ServerResource::Setup, ServerResource::Host, ServerResource::Load];
        if !start.settings.simulation {
            expected.push(ServerResource::Health);
        }
        if start.settings.prometheus_port.is_some() {
            expected.push(ServerResource::Metrics);
        }
        if !start.unregistered {
            expected.push(ServerResource::Worker);
        }
        for resource in expected {
            state.handle(ServerInput::ResourceStarted(resource), origin.elapsed());
        }
        let (stop, _) = watch::channel(false);
        let mut resources = Resources::default();
        let started = {
            let startup = self.start(&start, &stop, &mut resources);
            tokio::pin!(startup);
            loop {
                tokio::select! {
                    biased;
                    changed = control.changed() => {
                        if changed.is_err() { break None; }
                        let request = control.borrow_and_update().clone();
                        if request.drain.is_some() {
                            let effects = state.handle(ServerInput::Drain, origin.elapsed());
                            self.publish(&state, effects);
                        }
                        if request.close {
                            let effects = state.handle(ServerInput::Close, origin.elapsed());
                            self.publish(&state, effects);
                            break None;
                        }
                    }
                    result = &mut startup => break Some(result),
                }
            }
        };
        let mut drain = None;
        let mut started_notification = None;
        if let Some(result) = started {
            let readiness = result.as_ref().ok().cloned();
            let effects =
                state.handle(ServerInput::StartupFinished(result.map(|_| ())), origin.elapsed());
            let notify_started = effects.contains(&ServerEffect::Started);
            if effects.contains(&ServerEffect::Drain) {
                drain =
                    Some(self.drain_resources(
                        resources.worker.clone(),
                        control.borrow().drain.flatten(),
                    ));
            }
            self.publish(&state, effects);
            if notify_started {
                started_notification =
                    Some((self.host.started().boxed(), readiness.expect("startup succeeded")));
            }
        }
        while !matches!(state.phase(), ServerPhase::Closing | ServerPhase::Closed) {
            tokio::select! {
                biased;
                changed = control.changed() => {
                    if changed.is_err() { break; }
                    let request = control.borrow_and_update().clone();
                    if request.close {
                        let effects = state.handle(ServerInput::Close, origin.elapsed());
                        self.publish(&state, effects);
                    } else if request.drain.is_some() {
                        let effects = state.handle(ServerInput::Drain, origin.elapsed());
                        if effects.contains(&ServerEffect::Drain) {
                            drain = Some(self.drain_resources(resources.worker.clone(), request.drain.flatten()));
                        }
                        self.publish(&state, effects);
                    }
                }
                result = async { started_notification.as_mut().expect("notification pending").0.as_mut().await }, if started_notification.is_some() => {
                    let (_, readiness) = started_notification.take().expect("notification pending");
                    match result {
                        Ok(()) => { readiness.send_replace(true); }
                        Err(error) => {
                            let effects = state.handle(ServerInput::Failed(error), origin.elapsed());
                            self.publish(&state, effects);
                        }
                    }
                }
                result = pending_drain(&mut drain), if drain.is_some() => {
                    drain = None;
                    let effects = state.handle(ServerInput::DrainFinished(result), origin.elapsed());
                    self.publish(&state, effects);
                }
                result = pending_worker(&mut resources.connection), if resources.connection.is_some() => {
                    resources.connection = None;
                    let result = result.unwrap_or_else(|error| Err(error.to_string()));
                    let error = result.clone().err().unwrap_or_else(|| "worker connection stopped".into());
                    state.handle(ServerInput::ResourceStopped { resource: ServerResource::Worker, result }, origin.elapsed());
                    let effects = state.handle(ServerInput::Failed(error), origin.elapsed());
                    self.publish(&state, effects);
                }
            }
        }
        drop(drain);
        let effects = state.handle(ServerInput::Close, origin.elapsed());
        self.publish(&state, effects);
        drop(started_notification);
        stop.send_replace(true);
        let mut closing = resources.close(self.host.clone());
        while !closing.is_empty() {
            let delay = state.wake_at().map(|at| at.saturating_sub(origin.elapsed()));
            tokio::select! {
                biased;
                _ = deadline(delay) => {
                    let effects = state.tick(origin.elapsed());
                    self.publish(&state, effects);
                }
                result = closing.next() => {
                    if let Some((resource, result)) = result {
                        if let Err(error) = &result {
                            crate::worker::diagnostics::emit(log::Level::Error, format!("server resource cleanup failed resource={resource:?} error={error}"));
                        }
                        let effects = state.handle(ServerInput::ResourceStopped { resource, result }, origin.elapsed());
                        self.publish(&state, effects);
                    }
                }
            }
        }
    }

    async fn start(
        &self,
        start: &ServerStart,
        stop: &watch::Sender<bool>,
        resources: &mut Resources,
    ) -> Result<watch::Sender<bool>, String> {
        let mut settings = start.settings.clone();
        if !start.has_rtc && !start.has_text_doors {
            return Err("No RTC session entrypoint has been registered".into());
        }
        if !start.unregistered {
            settings.check_registration(!start.endpoints.is_empty())?;
        }
        if settings.cloud() {
            settings.load_threshold = server::DEFAULT_LOAD_THRESHOLD;
        }
        let host = self.host.configured(
            start.has_rtc,
            start.has_on_request,
            start.has_load_fnc && !settings.cloud(),
        );
        let configured = settings.prometheus_multiproc_dir.clone();
        let discover_cpus = settings.num_idle_processes.is_none();
        let workers = match &self.workers {
            Some(workers) => workers.clone(),
            None => crate::process::workers().map_err(|error| error.to_string())?,
        };
        let admission_deadline =
            Duration::try_from_secs_f64(settings.initialize_process_timeout)
                .map_err(|_| "initialization timeout must be finite and nonnegative")?;
        let prepare = async {
            loop {
                match workers.try_admit(lk_agents_abi::utils::workers::WorkerClass::WorkerIo) {
                    Ok(permit) => {
                        break Ok(workers.spawn_admitted(permit, move || {
                            crate::worker::diagnostics::initialize()?;
                            Ok((
                                server::prepare_prometheus_dir(configured.as_deref()),
                                discover_cpus.then(|| crate::worker::CpuMonitor::new().cpu_count()),
                            ))
                        }));
                    }
                    Err(lk_agents_abi::utils::workers::WorkerError::Exhausted(_)) => {
                        tokio::time::sleep(crate::worker::ipc::pool::REFILL_INTERVAL).await;
                    }
                    Err(error) => break Err(error.to_string()),
                }
            }
        };
        resources.setup = Some(
            tokio::time::timeout(admission_deadline, prepare)
                .await
                .map_err(|_| "server setup capacity wait timed out".to_owned())??,
        );
        let setup = resources.setup.as_mut().expect("setup owned before awaiting").await;
        resources.setup.take();
        let (directory, cpus) = setup.map_err(|error| error.to_string())??;
        settings.prometheus_multiproc_dir = directory;
        if let Some(cpus) = cpus {
            settings.num_idle_processes =
                Some(crate::worker::EnvOption::Split { dev: 0, prod: cpus.ceil() as u32 });
        }
        if !start.has_rtc {
            settings.num_idle_processes = Some(crate::worker::EnvOption::Fixed(0));
        }
        let server_type = if start.publisher {
            crate::worker::ServerType::Publisher
        } else {
            crate::worker::ServerType::Room
        };
        let status = Arc::new(ServerStatus::default());
        let probe = Arc::new(HostHealth(host.clone()));
        if !settings.simulation {
            let (listener, port) =
                server::bind_listener(&settings.host, settings.port.get(start.devmode))
                    .await
                    .map_err(|error| error.to_string())?;
            self.snapshot.send_modify(|snapshot| snapshot.http_port = port);
            resources.health = Some(tokio::spawn(server::serve_health(
                listener,
                Arc::new(settings.clone()),
                server_type,
                status.clone(),
                probe.clone(),
                stop.subscribe(),
            )));
        }
        if let Some(port) = settings.prometheus_port {
            let (listener, _) = server::bind_listener(&settings.host, port)
                .await
                .map_err(|error| error.to_string())?;
            resources.metrics =
                Some(tokio::spawn(server::serve_metrics(listener, probe, stop.subscribe())));
        }
        if settings.multiprocessing_fallback {
            crate::worker::diagnostics::emit(
                log::Level::Warn,
                "the forkserver is used on Linux only, job processes are spawned".into(),
            );
        }
        host.start(settings.clone(), start.devmode, start.has_rtc).await?;
        if !start.unregistered {
            let options = settings.worker_options(
                start.devmode,
                server_type,
                start.endpoints.clone(),
                match &start.transport {
                    crate::worker::TransportConfig::WebSocket { heartbeat } => {
                        crate::worker::TransportConfig::WebSocket { heartbeat: *heartbeat }
                    }
                    crate::worker::TransportConfig::WebTransport { insecure } => {
                        crate::worker::TransportConfig::WebTransport { insecure: *insecure }
                    }
                },
                start.worker_url.clone(),
            );
            let (mut worker, handle) = Worker::new(options, host.clone(), workers.clone());
            resources.worker = Some(handle.clone());
            self.snapshot.send_modify(|snapshot| snapshot.worker = Some(handle.clone()));
            let mut notifications = handle.subscribe();
            let (host, snapshots, failed) =
                (self.host.clone(), self.snapshot.clone(), status.clone());
            resources.connection = Some(tokio::spawn(async move {
                let forward = async {
                    while let Ok(event) = notifications.recv().await {
                        match event {
                            WorkerNotification::Registered(info) => {
                                snapshots.send_modify(|snapshot| {
                                    snapshot.worker_id = info.worker_id.clone()
                                });
                                host.registered(info).await?;
                            }
                            WorkerNotification::Stopped(_) => break,
                            _ => {}
                        }
                    }
                    Ok::<(), String>(())
                };
                let run = worker.run();
                tokio::pin!(run);
                let result = tokio::select! {
                    result = &mut run => result.map_err(|error| error.to_string()),
                    result = forward => match result {
                        Ok(()) => run.await.map_err(|error| error.to_string()),
                        Err(error) => Err(error),
                    },
                };
                if result.is_err() {
                    failed.connection_failed.store(true, std::sync::atomic::Ordering::Release);
                }
                result
            }));
        }
        let (host, worker) = (self.host.clone(), resources.worker.clone());
        let readiness = status.ready.clone();
        resources.load = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(crate::worker::UPDATE_LOAD_INTERVAL);
            loop {
                interval.tick().await;
                if let Some(worker) = &worker {
                    let snapshot = worker.snapshot();
                    status.set_load(snapshot.worker_load);
                    if let Some(target) = snapshot.target_idle_processes {
                        host.idle_processes(target).await?;
                    }
                }
                host.worker_load(status.load()).await?;
            }
        }));
        Ok(readiness)
    }

    fn drain_resources(
        &self,
        worker: Option<WorkerHandle>,
        timeout: Option<Duration>,
    ) -> BoxFuture<'static, Result<(), String>> {
        let host = self.host.clone();
        async move {
            let drain = async {
                host.close_websockets().await?;
                match worker {
                    Some(worker) => worker.drain(timeout).await,
                    None => host.join_jobs().await,
                }
            };
            match timeout {
                Some(timeout) => tokio::time::timeout(timeout, drain)
                    .await
                    .map_err(|_| "drain timed out".to_string())?,
                None => drain.await,
            }
        }
        .boxed()
    }
}

async fn wait_outcome(outcome: &Outcome) -> Result<(), String> {
    outcome
        .subscribe()
        .wait_for(Option::is_some)
        .await
        .map_err(|error| error.to_string())?
        .clone()
        .unwrap_or(Ok(()))
}

async fn deadline(delay: Option<Duration>) {
    match delay {
        Some(delay) => tokio::time::sleep(delay).await,
        None => std::future::pending().await,
    }
}

async fn pending_drain(
    drain: &mut Option<BoxFuture<'static, Result<(), String>>>,
) -> Result<(), String> {
    match drain {
        Some(drain) => drain.await,
        None => std::future::pending().await,
    }
}

async fn pending_worker(
    worker: &mut Option<tokio::task::JoinHandle<Result<(), String>>>,
) -> Result<Result<(), String>, tokio::task::JoinError> {
    match worker {
        Some(worker) => worker.await,
        None => std::future::pending().await,
    }
}

#[derive(Default)]
struct Resources {
    setup: Option<lk_agents_media::workers::MediaWorker<Result<(Option<String>, Option<f32>), String>>>,
    worker: Option<WorkerHandle>,
    connection: Option<tokio::task::JoinHandle<Result<(), String>>>,
    load: Option<tokio::task::JoinHandle<Result<(), String>>>,
    health: Option<tokio::task::JoinHandle<Result<(), String>>>,
    metrics: Option<tokio::task::JoinHandle<Result<(), String>>>,
}

impl Resources {
    fn close<H: ServerHost>(
        self,
        host: H,
    ) -> FuturesUnordered<BoxFuture<'static, (ServerResource, Result<(), String>)>> {
        let closing = FuturesUnordered::new();
        closing.push(async move { (ServerResource::Host, host.close().await) }.boxed());
        closing.push(
            async move {
                let result = match self.setup {
                    Some(setup) => setup
                        .await
                        .map_err(|error| error.to_string())
                        .and_then(|result| result.map(|_| ())),
                    None => Ok(()),
                };
                (ServerResource::Setup, result)
            }
            .boxed(),
        );
        let close_admission = self
            .worker
            .map(|worker| {
                worker.send(WorkerCommand::Close).map_err(|_| {
                    worker
                        .snapshot()
                        .finished
                        .unwrap_or_else(|| "worker close admission failed".into())
                })
            })
            .unwrap_or(Ok(()));
        for (resource, task) in [
            (ServerResource::Worker, self.connection),
            (ServerResource::Load, self.load),
            (ServerResource::Health, self.health),
            (ServerResource::Metrics, self.metrics),
        ] {
            let admission =
                if resource == ServerResource::Worker { close_admission.clone() } else { Ok(()) };
            if resource == ServerResource::Load
                && let Some(task) = &task
            {
                task.abort();
            }
            closing.push(
                async move {
                    let result = match task {
                        Some(task) => match task.await {
                            Ok(result) => result,
                            Err(error)
                                if resource == ServerResource::Load && error.is_cancelled() =>
                            {
                                Ok(())
                            }
                            Err(error) => Err(error.to_string()),
                        },
                        None => Ok(()),
                    };
                    (resource, admission.and(result))
                }
                .boxed(),
            );
        }
        closing
    }
}

struct HostHealth<H>(H);

#[async_trait]
impl<H: ServerHost> server::HealthProbe for HostHealth<H> {
    async fn health(&self) -> usize {
        self.0.active_jobs().await
    }
    async fn metrics(&self) -> Option<(String, String)> {
        self.0.metrics().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{JobDecision, JobRequest, RunningJobInfo};

    #[derive(Clone)]
    struct Host {
        started: Arc<tokio::sync::Notify>,
        joining: Arc<tokio::sync::Notify>,
        closing: Arc<tokio::sync::Notify>,
        release_close: Arc<tokio::sync::Semaphore>,
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
                fail_start: false,
                fail_close: false,
            }
        }
    }

    #[async_trait]
    impl WorkerHost for Host {
        async fn on_job_request(&self, _: JobRequest) -> JobDecision {
            JobDecision::Unanswered
        }
        async fn launch_job(
            &self,
            _: RunningJobInfo,
        ) -> crate::worker::HostResult<crate::worker::JobId> {
            Ok("job".into())
        }
        async fn terminate_job(&self, _: crate::worker::JobId) -> crate::worker::HostResult {
            Ok(())
        }
        async fn user_load(&self) -> Option<f32> {
            Some(0.0)
        }
    }

    #[async_trait]
    impl ServerHost for Host {
        fn configured(&self, _: bool, _: bool, _: bool) -> Self {
            self.clone()
        }
        async fn start(&self, _: ServerSettings, _: bool, _: bool) -> Result<(), String> {
            if self.fail_start { Err("pool warmup timed out".into()) } else { Ok(()) }
        }
        async fn started(&self) -> Result<(), String> {
            self.started.notify_one();
            Ok(())
        }
        async fn registered(&self, _: WorkerRegistered) -> Result<(), String> {
            Ok(())
        }
        async fn idle_processes(&self, _: u32) -> Result<(), String> {
            Ok(())
        }
        async fn worker_load(&self, _: f32) -> Result<(), String> {
            Ok(())
        }
        async fn close_websockets(&self) -> Result<(), String> {
            Ok(())
        }
        async fn join_jobs(&self) -> Result<(), String> {
            self.joining.notify_one();
            std::future::pending().await
        }
        async fn close(&self) -> Result<(), String> {
            self.closing.notify_one();
            let _permit = self.release_close.acquire().await.unwrap();
            if self.fail_close { Err("host close failed".into()) } else { Ok(()) }
        }
        async fn active_jobs(&self) -> usize {
            0
        }
        async fn metrics(&self) -> Option<(String, String)> {
            None
        }
    }

    fn options() -> ServerStart {
        ServerStart {
            settings: ServerSettings {
                simulation: true,
                num_idle_processes: Some(crate::worker::EnvOption::Fixed(0)),
                ..Default::default()
            },
            devmode: true,
            unregistered: true,
            has_rtc: false,
            has_text_doors: true,
            has_load_fnc: false,
            has_on_request: false,
            publisher: false,
            endpoints: vec![],
            transport: crate::worker::TransportConfig::WebSocket {
                heartbeat: crate::worker::HEARTBEAT_INTERVAL,
            },
            worker_url: None,
        }
    }

    #[tokio::test]
    async fn server_drain_close_callers_join_without_holding_a_lifecycle_lock() {
        let host = Host::new();
        let runtime =
            ServerRuntime::new(host.clone(), Some(crate::worker::test_support::native_workers()));
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
        assert_eq!(first_drain.await, Err("server closed during drain".into()));
        assert_eq!(second_drain.await, Err("server closed during drain".into()));
        assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
    }

    #[tokio::test]
    async fn failed_registration_is_reported_by_run_and_close_joins_cleanup() {
        let host = Host::new();
        host.release_close.add_permits(1);
        let runtime = ServerRuntime::new(host, Some(crate::worker::test_support::native_workers()));
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
        let runtime =
            ServerRuntime::new(host.clone(), Some(crate::worker::test_support::native_workers()));
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
        let runtime =
            ServerRuntime::new(host.clone(), Some(crate::worker::test_support::native_workers()));
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
        let runtime =
            ServerRuntime::new(host.clone(), Some(crate::worker::test_support::native_workers()));
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
        let workers = crate::worker::test_support::native_workers();
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(
                workers.try_admit(lk_agents_abi::utils::workers::WorkerClass::WorkerIo).unwrap(),
            );
        }
        let host = Host::new();
        host.release_close.add_permits(1);
        let runtime = ServerRuntime::new(host.clone(), Some(workers.clone()));
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
        let workers = crate::worker::test_support::native_workers();
        let held: Vec<_> = (0..4)
            .map(|_| {
                workers.try_admit(lk_agents_abi::utils::workers::WorkerClass::WorkerIo).unwrap()
            })
            .collect();
        let host = Host::new();
        host.release_close.add_permits(1);
        let runtime = ServerRuntime::new(host.clone(), Some(workers));
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
        let workers = crate::worker::test_support::native_workers();
        let held: Vec<_> = (0..4)
            .map(|_| {
                workers.try_admit(lk_agents_abi::utils::workers::WorkerClass::WorkerIo).unwrap()
            })
            .collect();
        let host = Host::new();
        host.release_close.add_permits(1);
        let runtime = ServerRuntime::new(host.clone(), Some(workers));
        let mut start = options();
        start.settings.initialize_process_timeout = 0.01;
        assert_eq!(runtime.run(start).await, Err("server setup capacity wait timed out".into()));
        runtime.close().await.unwrap();
        assert!(host.started.notified().now_or_never().is_none());
        assert_eq!(runtime.snapshot().phase, ServerPhase::Closed);
        drop(held);
    }
}

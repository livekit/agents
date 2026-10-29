//! `AgentServer` around the worker connection (`worker.py`): its options resolved once against the
//! environment, the health listener and its `/worker` record, the job `simulate_job` hands the
//! host, and the lifecycle (start, load, drain, close) the binding runs. The host keeps the job
//! processes and the user's callbacks.

use crate::{
    Worker, WorkerCommand, WorkerHandle, WorkerHost, WorkerNotification, WorkerRegistered,
};
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use lifecycle::{ServerEffect, ServerInput, ServerLifecycle};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, watch};
mod health;
mod lifecycle;
pub use health::{
    HealthProbe, ServerStatus, bind_listener, serve_health, serve_metrics, worker_info_json,
};
#[cfg(test)]
use health::{health_route, metrics_route};
pub use lifecycle::{ServerPhase, ServerResource};
mod settings;
pub use settings::{
    DEFAULT_LOAD_THRESHOLD, Environment, ServerSettings, endpoint_url_base, prepare_prometheus_dir,
};

use livekit_protocol as proto;
use prost::Message as _;

use crate::{
    AcceptArguments, JobDecision, JobRequest, RunningJobInfo, ServerType, WORKER_PROTOCOL_VERSION,
};

// --------------------------------------------------------------------- simulated jobs

/// `simulate_job`'s job: a room job under `room` for `agent_identity` (read from `token` when one
/// is given), with a join token minted from the credentials unless given, or none for a fake job
/// without credentials.
#[allow(clippy::too_many_arguments)]
pub fn simulated_job(
    settings: &ServerSettings,
    worker_id: &str,
    room: &str,
    fake_job: bool,
    agent_identity: Option<String>,
    room_info: Option<proto::Room>,
    token: Option<String>,
) -> Result<RunningJobInfo, String> {
    let token = token.filter(|t| !t.is_empty());
    let agent_identity = match (&token, agent_identity) {
        (Some(token), _) => Some(crate::token_identity(token).map_err(|e| e.to_string())?),
        (None, identity) => identity,
    };
    let identity = match agent_identity {
        Some(identity) => identity,
        None if fake_job => lk_agents_net::strings::shortuuid("fake-agent-"),
        None => return Err("agent_identity is None but fake_job is False".into()),
    };
    let room_info = match room_info {
        Some(info) => info,
        None if fake_job => proto::Room {
            sid: lk_agents_net::strings::shortuuid("SRM_"),
            name: room.to_owned(),
            ..Default::default()
        },
        None => return Err("room_info is None but fake_job is False".into()),
    };
    let job = proto::Job {
        id: lk_agents_net::strings::shortuuid(if fake_job { "mock-job-" } else { "job-" }),
        room: Some(room_info),
        r#type: proto::JobType::JtRoom as i32,
        ..Default::default()
    };
    let token = match token {
        Some(token) => token,
        None if fake_job && (settings.api_key.is_empty() || settings.api_secret.is_empty()) => {
            String::new()
        }
        None => crate::agent_join_token(&settings.api_key, &settings.api_secret, &identity, room)
            .map_err(|e| e.to_string())?,
    };
    Ok(RunningJobInfo {
        accept_arguments: AcceptArguments {
            identity,
            name: String::new(),
            metadata: String::new(),
            attributes: Vec::new(),
        },
        job: crate::protocol::job_info(&job),
        url: settings.ws_url.clone(),
        token,
        worker_id: worker_id.to_owned(),
        fake_job,
        received_at: 0.0,
        accepted_at: 0.0,
        assigned_at: 0.0,
    })
}

// --------------------------------------------------------------------- job requests

/// The answer to a job request when the host's `on_request` is not needed: a server with no
/// voice door rejects it (without terminating, so another worker may take it) and one whose door
/// has no `on_request` accepts it as `JobRequest.accept()` does. `None`: ask the host.
pub fn own_decision(req: &JobRequest, has_rtc: bool, has_on_request: bool) -> Option<JobDecision> {
    if !has_rtc {
        crate::diagnostics::emit(
            log::Level::Warn,
            format!(
                "a job reached a server with no rtc_session; rejecting it job_id={} room={}",
                req.job.id,
                req.room_name()
            ),
        );
        return Some(JobDecision::Reject { terminate: false });
    }
    (!has_on_request).then(|| {
        JobDecision::Accept(AcceptArguments {
            identity: format!("agent-{}", req.job.id),
            ..AcceptArguments::default()
        })
    })
}

/// A room as `simulate_job` takes it, from its protobuf bytes.
pub fn decode_room(bytes: &[u8]) -> Result<proto::Room, String> {
    proto::Room::decode(bytes).map_err(|e| format!("room_info: {e}"))
}

// --------------------------------------------------------------------- the listeners

#[async_trait]
pub trait ServerHost: WorkerHost + Clone + Send + Sync + 'static {
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

pub struct ServerStart {
    pub settings: ServerSettings,
    pub devmode: bool,
    pub unregistered: bool,
    pub has_rtc: bool,
    pub has_text_doors: bool,
    pub has_load_fnc: bool,
    pub has_on_request: bool,
    pub publisher: bool,
    pub endpoints: Vec<crate::protocol::AgentEndpoint>,
    pub transport: crate::TransportConfig,
    pub worker_url: Option<String>,
}

#[derive(Clone)]
pub struct ServerSnapshot {
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
pub struct ServerRuntime<H: ServerHost> {
    host: H,
    workers: Option<crate::io::WorkerIo>,
    control: watch::Sender<Control>,
    receiver: Arc<Mutex<Option<watch::Receiver<Control>>>>,
    snapshot: watch::Sender<ServerSnapshot>,
    outcome: Outcome,
    cleanup_outcome: Outcome,
    drain_outcome: Outcome,
}

impl<H: ServerHost> ServerRuntime<H> {
    pub fn new(host: H, workers: Option<crate::io::WorkerIo>) -> Self {
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

    pub fn snapshot(&self) -> ServerSnapshot {
        self.snapshot.borrow().clone()
    }

    pub async fn run(&self, start: ServerStart) -> Result<(), String> {
        let receiver = self.receiver.lock().await.take().ok_or("worker is already running")?;
        let origin = tokio::time::Instant::now();
        let timeout = Duration::try_from_secs_f64(start.settings.shutdown_process_timeout)
            .map_err(|_| "shutdown timeout must be finite and nonnegative")?;
        let process_cleanup = timeout * 3 + crate::ipc::supervisor::KILL_GRACE;
        let connection_cleanup = if start.unregistered {
            Duration::ZERO
        } else {
            crate::WorkerOptions::default().connection_timeout
        };
        let mut state = ServerLifecycle::new(process_cleanup.max(connection_cleanup));
        let effects = state.handle(ServerInput::Start, Duration::ZERO);
        self.publish(&state, effects);
        self.snapshot.send_modify(|snapshot| snapshot.has_rtc = start.has_rtc);
        let runtime = self.clone();
        tokio::spawn(async move { runtime.serve(start, receiver, state, origin).await });
        wait_outcome(&self.outcome).await
    }

    pub async fn simulate_job(&self, info: crate::RunningJobInfo) -> Result<(), String> {
        if self.snapshot().phase != ServerPhase::Running {
            return Err("the server is not accepting jobs".into());
        }
        self.host.launch_job(info).await.map(|_| ()).map_err(|error| error.to_string())
    }

    pub async fn drain(&self, timeout: Option<Duration>) -> Result<(), String> {
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

    pub async fn close(&self) -> Result<(), String> {
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
                    crate::diagnostics::emit(
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
                            crate::diagnostics::emit(log::Level::Error, format!("server resource cleanup failed resource={resource:?} error={error}"));
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
            settings.load_threshold = DEFAULT_LOAD_THRESHOLD;
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
            None => crate::io::WorkerIo::new(tokio::runtime::Handle::current(), 4)
                .map_err(|error| error.to_string())?,
        };
        let admission_deadline =
            Duration::try_from_secs_f64(settings.initialize_process_timeout)
                .map_err(|_| "initialization timeout must be finite and nonnegative")?;
        let prepare = async {
            loop {
                match workers.try_admit() {
                    Ok(permit) => {
                        break Ok(workers.spawn_admitted(permit, move || {
                            crate::diagnostics::initialize()?;
                            Ok((
                                prepare_prometheus_dir(configured.as_deref()),
                                discover_cpus.then(|| crate::CpuMonitor::new().cpu_count()),
                            ))
                        }));
                    }
                    Err(crate::io::WorkerIoError::Exhausted) => {
                        tokio::time::sleep(crate::ipc::pool::REFILL_INTERVAL).await;
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
                Some(crate::EnvOption::Split { dev: 0, prod: cpus.ceil() as u32 });
        }
        if !start.has_rtc {
            settings.num_idle_processes = Some(crate::EnvOption::Fixed(0));
        }
        let server_type =
            if start.publisher { crate::ServerType::Publisher } else { crate::ServerType::Room };
        let status = Arc::new(ServerStatus::default());
        let probe = Arc::new(HostHealth(host.clone()));
        if !settings.simulation {
            let (listener, port) = bind_listener(&settings.host, settings.port.get(start.devmode))
                .await
                .map_err(|error| error.to_string())?;
            self.snapshot.send_modify(|snapshot| snapshot.http_port = port);
            resources.health = Some(tokio::spawn(serve_health(
                listener,
                Arc::new(settings.clone()),
                server_type,
                status.clone(),
                probe.clone(),
                stop.subscribe(),
            )));
        }
        if let Some(port) = settings.prometheus_port {
            let (listener, _) =
                bind_listener(&settings.host, port).await.map_err(|error| error.to_string())?;
            resources.metrics =
                Some(tokio::spawn(serve_metrics(listener, probe, stop.subscribe())));
        }
        if settings.multiprocessing_fallback {
            crate::diagnostics::emit(
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
                    crate::TransportConfig::WebSocket { heartbeat } => {
                        crate::TransportConfig::WebSocket { heartbeat: *heartbeat }
                    }
                    crate::TransportConfig::WebTransport { insecure } => {
                        crate::TransportConfig::WebTransport { insecure: *insecure }
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
            let mut interval = tokio::time::interval(crate::UPDATE_LOAD_INTERVAL);
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
    setup: Option<crate::io::WorkerIoTask<Result<(Option<String>, Option<f32>), String>>>,
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
impl<H: ServerHost> HealthProbe for HostHealth<H> {
    async fn health(&self) -> usize {
        self.0.active_jobs().await
    }
    async fn metrics(&self) -> Option<(String, String)> {
        self.0.metrics().await
    }
}

#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod tests;

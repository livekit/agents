//! `AgentServer` around the worker connection (`worker.py`): its options resolved once against the
//! environment, the health listener and its `/worker` record, the job `simulate_job` hands the
//! host, and the lifecycle (start, load, drain, close) the binding runs. The host keeps the job
//! processes and the user's callbacks.

use crate::agent_server::ServerCallbacks;
use crate::{Worker, WorkerClient, WorkerCommand, WorkerHost, WorkerNotification};
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskError};
use lk_agents_net::task::TokioSpawner;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, watch};
mod health;
pub use health::{
    HealthProbe, ServerStatus, bind_listener, serve_health, serve_metrics, worker_info_json,
};
#[cfg(test)]
use health::{health_route, metrics_route};
mod settings;
pub use settings::{
    DEFAULT_LOAD_THRESHOLD, Environment, ServerSettings, endpoint_url_base, prepare_prometheus_dir,
};

use livekit_protocol as proto;
use prost::Message as _;

use crate::{AcceptArguments, JobDecision, JobRequest, RunningJobInfo, WORKER_PROTOCOL_VERSION};

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
        (Some(token), _) => {
            Some(lk_agents_net::token::token_identity(token).map_err(|e| e.to_string())?)
        }
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
        None => lk_agents_net::token::room_join_token(
            &settings.api_key,
            &settings.api_secret,
            &identity,
            room,
            true,
            None,
        )
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

/// The answer to a job request: a server with no voice door rejects it (another worker may take
/// it), one with a voice door accepts it.
pub fn job_decision(req: &JobRequest, has_rtc: bool) -> JobDecision {
    if !has_rtc {
        crate::diagnostics::emit(
            log::Level::Warn,
            format!(
                "a job reached a server with no rtc_session; rejecting it job_id={} room={}",
                req.job.id,
                req.room_name()
            ),
        );
        return JobDecision::Reject;
    }
    JobDecision::Accept(AcceptArguments {
        identity: format!("agent-{}", req.job.id),
        ..AcceptArguments::default()
    })
}

/// A room as `simulate_job` takes it, from its protobuf bytes.
pub fn decode_room(bytes: &[u8]) -> Result<proto::Room, String> {
    proto::Room::decode(bytes).map_err(|e| format!("room_info: {e}"))
}

// --------------------------------------------------------------------- the listeners

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerPhase {
    Idle,
    Starting,
    Running,
    Draining,
    Closing,
    Closed,
}

/// What a running server owns and stops on close.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ServerResource {
    Setup,
    Host,
    Worker,
    Load,
    Health,
    Metrics,
}

pub struct ServerStart {
    pub settings: ServerSettings,
    pub devmode: bool,
    pub unregistered: bool,
    pub has_rtc: bool,
    pub has_text_doors: bool,
    pub has_load_fnc: bool,
    pub endpoints: Vec<crate::protocol::AgentEndpoint>,
    pub transport: crate::TransportConfig,
}

#[derive(Clone)]
pub struct ServerSnapshot {
    pub phase: ServerPhase,
    pub worker_id: String,
    pub http_port: u16,
    pub worker: Option<WorkerClient>,
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
pub struct ServerRuntime {
    host: ServerCallbacks,
    blocking_io: Option<lk_agents_net::blocking_io::BlockingIo>,
    control: watch::Sender<Control>,
    receiver: Arc<Mutex<Option<watch::Receiver<Control>>>>,
    snapshot: watch::Sender<ServerSnapshot>,
    outcome: Outcome,
    cleanup_outcome: Outcome,
    drain_outcome: Outcome,
    /// The lifecycle task `run` starts, outliving a cancelled `run` caller; `close` takes and
    /// awaits it unless a missed cleanup deadline left it holding the stragglers.
    serving: Arc<parking_lot::Mutex<Option<OwnedTask>>>,
}

impl ServerRuntime {
    pub fn new(
        host: ServerCallbacks,
        blocking_io: Option<lk_agents_net::blocking_io::BlockingIo>,
    ) -> Self {
        let (control, receiver) = watch::channel(Control::default());
        Self {
            host,
            blocking_io,
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
            serving: Arc::default(),
        }
    }

    pub fn snapshot(&self) -> ServerSnapshot {
        self.snapshot.borrow().clone()
    }

    pub async fn run(&self, start: ServerStart) -> Result<(), String> {
        let receiver = self.receiver.lock().await.take().ok_or("worker is already running")?;
        let timeout = Duration::try_from_secs_f64(start.settings.shutdown_process_timeout)
            .map_err(|_| "shutdown timeout must be finite and nonnegative")?;
        let process_cleanup = timeout * 3;
        let connection_cleanup =
            if start.unregistered { Duration::ZERO } else { crate::config::CONNECTION_TIMEOUT };
        let cleanup_timeout = process_cleanup.max(connection_cleanup);
        self.snapshot.send_modify(|snapshot| {
            snapshot.phase = ServerPhase::Starting;
            snapshot.has_rtc = start.has_rtc;
        });
        let runtime = self.clone();
        let serving = TokioSpawner::current()
            .spawn(async move { runtime.serve(start, receiver, cleanup_timeout).await });
        *self.serving.lock() = Some(serving);
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
        let result = wait_outcome(&self.cleanup_outcome).await;
        if self.snapshot().incomplete_cleanup.is_empty() {
            let serving = self.serving.lock().take();
            if let Some(serving) = serving {
                let _ = serving.await;
            }
        }
        result
    }

    fn set_phase(&self, phase: ServerPhase) {
        self.snapshot.send_modify(|snapshot| snapshot.phase = phase);
    }

    /// The first outcome sent wins; later ones are no-ops.
    fn settle(outcome: &Outcome, result: Result<(), String>) {
        outcome.send_if_modified(|outcome| {
            if outcome.is_some() {
                return false;
            }
            *outcome = Some(result);
            true
        });
    }

    async fn serve(
        self,
        start: ServerStart,
        mut control: watch::Receiver<Control>,
        cleanup_timeout: Duration,
    ) {
        let (stop, _) = watch::channel(false);
        let mut resources = Resources::default();
        let mut failure: Option<String> = None;
        let mut drain_requested = false;
        let started = {
            let startup = self.start(&start, &stop, &mut resources);
            tokio::pin!(startup);
            loop {
                tokio::select! {
                    biased;
                    changed = control.changed() => {
                        if changed.is_err() { break None; }
                        let request = control.borrow_and_update().clone();
                        drain_requested |= request.drain.is_some();
                        if request.close { break None; }
                    }
                    result = &mut startup => break Some(result),
                }
            }
        };
        let mut drain = None;
        let mut started_notification = None;
        let running = matches!(started, Some(Ok(_)));
        match started {
            Some(Ok(readiness)) => {
                self.set_phase(ServerPhase::Running);
                if drain_requested {
                    self.set_phase(ServerPhase::Draining);
                    let timeout = control.borrow().drain.flatten();
                    drain = Some(self.drain_resources(resources.worker.clone(), timeout));
                }
                started_notification = Some((self.host.proxy.started(), readiness));
            }
            Some(Err(error)) => failure = Some(error),
            None => {}
        }
        let mut drained = false;
        if running {
            loop {
                tokio::select! {
                    biased;
                    changed = control.changed() => {
                        if changed.is_err() { break; }
                        let request = control.borrow_and_update().clone();
                        if request.close { break; }
                        if request.drain.is_some() && !drain_requested {
                            drain_requested = true;
                            self.set_phase(ServerPhase::Draining);
                            drain = Some(self.drain_resources(resources.worker.clone(), request.drain.flatten()));
                        }
                    }
                    result = async { started_notification.as_mut().expect("notification pending").0.as_mut().await }, if started_notification.is_some() => {
                        let (_, readiness) = started_notification.take().expect("notification pending");
                        match result {
                            Ok(()) => { readiness.send_replace(true); }
                            Err(error) => { failure = Some(error.to_string()); break; }
                        }
                    }
                    result = pending_drain(&mut drain), if drain.is_some() => {
                        drain = None;
                        drained = true;
                        Self::settle(&self.drain_outcome, result);
                    }
                    result = pending_worker(&mut resources.connection), if resources.connection.is_some() => {
                        // The connection is gone: nothing of it is left to stop.
                        resources.connection = None;
                        resources.worker = None;
                        failure = Some(match result.unwrap_or_else(|error| Err(error.to_string())) {
                            Ok(()) => "worker connection stopped".into(),
                            Err(error) => format!("{:?}: {error}", ServerResource::Worker),
                        });
                        break;
                    }
                }
            }
        }
        self.set_phase(ServerPhase::Closing);
        if drain_requested && !drained {
            Self::settle(&self.drain_outcome, Err("server closed during drain".into()));
        }
        drop(drain);
        drop(started_notification);
        stop.send_replace(true);
        let mut closing = resources.close(self.host.clone());
        let mut remaining: std::collections::BTreeSet<ServerResource> = [
            ServerResource::Host,
            ServerResource::Setup,
            ServerResource::Worker,
            ServerResource::Load,
            ServerResource::Health,
            ServerResource::Metrics,
        ]
        .into();
        let mut cleanup_failure: Option<String> = None;
        let mut reported = false;
        let deadline = tokio::time::sleep(cleanup_timeout);
        tokio::pin!(deadline);
        while !remaining.is_empty() {
            tokio::select! {
                biased;
                () = &mut deadline, if !reported => {
                    // The runtime keeps the stragglers; the callers get their outcome now.
                    reported = true;
                    let resources: Vec<_> = remaining.iter().copied().collect();
                    crate::diagnostics::emit(
                        log::Level::Error,
                        format!("server cleanup deadline missed; runtime retains ownership resources={resources:?}"),
                    );
                    self.snapshot.send_modify(|snapshot| snapshot.incomplete_cleanup = resources);
                    self.finish(&failure, &cleanup_failure);
                }
                Some((resource, result)) = closing.next() => {
                    remaining.remove(&resource);
                    if let Err(error) = result {
                        crate::diagnostics::emit(log::Level::Error, format!("server resource cleanup failed resource={resource:?} error={error}"));
                        let error = format!("{resource:?}: {error}");
                        cleanup_failure.get_or_insert(error.clone());
                        failure.get_or_insert(error);
                    }
                }
            }
        }
        self.set_phase(ServerPhase::Closed);
        if !reported {
            self.finish(&failure, &cleanup_failure);
        }
    }

    fn finish(&self, failure: &Option<String>, cleanup_failure: &Option<String>) {
        Self::settle(&self.cleanup_outcome, cleanup_failure.clone().map_or(Ok(()), Err));
        Self::settle(&self.outcome, failure.clone().map_or(Ok(()), Err));
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
        let host = self.host.configured(start.has_rtc, start.has_load_fnc && !settings.cloud());
        let configured = settings.prometheus_multiproc_dir.clone();
        let discover_cpus = settings.num_idle_processes.is_none();
        let blocking_io =
            self.blocking_io.clone().unwrap_or_else(lk_agents_net::blocking_io::blocking_io);
        let admission_deadline =
            Duration::try_from_secs_f64(settings.initialize_process_timeout)
                .map_err(|_| "initialization timeout must be finite and nonnegative")?;
        let admit = async {
            loop {
                match blocking_io.try_admit(()) {
                    Ok(permit) => break permit,
                    Err(_) => blocking_io.wait_for_capacity(()).await,
                }
            }
        };
        let permit = tokio::time::timeout(admission_deadline, admit)
            .await
            .map_err(|_| "server setup capacity wait timed out".to_owned())?;
        resources.setup = Some(blocking_io.spawn_admitted(permit, move || {
            crate::diagnostics::initialize()?;
            Ok((
                prepare_prometheus_dir(configured.as_deref()),
                discover_cpus.then(|| crate::CpuMonitor::new().cpu_count()),
            ))
        }));
        let setup = resources.setup.as_mut().expect("setup owned before awaiting").await;
        resources.setup.take();
        let (directory, cpus) = setup.map_err(|error| error.to_string())??;
        settings.prometheus_multiproc_dir = directory;
        if let Some(cpus) = cpus {
            settings.num_idle_processes =
                Some(crate::EnvOption::Split { dev: 0, prod: (cpus.ceil() as u32).min(4) });
        }
        if !start.has_rtc {
            settings.num_idle_processes = Some(crate::EnvOption::Fixed(0));
        }
        let status = Arc::new(ServerStatus::default());
        let probe = Arc::new(host.clone());
        if !settings.simulation {
            let (listener, port) = bind_listener(&settings.host, settings.port.get(start.devmode))
                .await
                .map_err(|error| error.to_string())?;
            self.snapshot.send_modify(|snapshot| snapshot.http_port = port);
            resources.health = Some(TokioSpawner::current().spawn(serve_health(
                listener,
                Arc::new(settings.clone()),
                status.clone(),
                probe.clone(),
                stop.subscribe(),
            )));
        }
        if let Some(port) = settings.prometheus_port {
            let (listener, _) =
                bind_listener(&settings.host, port).await.map_err(|error| error.to_string())?;
            resources.metrics = Some(TokioSpawner::current().spawn(serve_metrics(
                listener,
                probe,
                stop.subscribe(),
            )));
        }
        if settings.multiprocessing_fallback {
            crate::diagnostics::emit(
                log::Level::Warn,
                "the forkserver is used on Linux only, job processes are spawned".into(),
            );
        }
        host.start(settings.clone(), start.devmode, start.has_rtc).await?;
        if !start.unregistered {
            let (config, connection) = settings.worker(start.devmode, start.endpoints.clone());
            let transport = match &start.transport {
                crate::TransportConfig::WebSocket { heartbeat } => {
                    crate::TransportConfig::WebSocket { heartbeat: *heartbeat }
                }
                crate::TransportConfig::WebTransport { insecure } => {
                    crate::TransportConfig::WebTransport { insecure: *insecure }
                }
            };
            let worker_host = Arc::new(host.clone());
            let (mut worker, handle) =
                Worker::new(config, connection, transport, worker_host, blocking_io.clone());
            resources.worker = Some(handle.clone());
            self.snapshot.send_modify(|snapshot| snapshot.worker = Some(handle.clone()));
            let mut notifications = handle.subscribe();
            let (snapshots, failed) = (self.snapshot.clone(), status.clone());
            resources.connection = Some(TokioSpawner::current().spawn(async move {
                let forward = async {
                    while let Ok(event) = notifications.recv().await {
                        match event {
                            WorkerNotification::Registered(info) => snapshots
                                .send_modify(|snapshot| snapshot.worker_id = info.worker_id),
                            WorkerNotification::Stopped(_) => break,
                            _ => {}
                        }
                    }
                };
                let run = worker.run();
                tokio::pin!(run);
                let result = tokio::select! {
                    result = &mut run => result,
                    () = forward => run.await,
                }
                .map_err(|error| error.to_string());
                if result.is_err() {
                    failed.connection_failed.store(true, std::sync::atomic::Ordering::Release);
                }
                result
            }));
        }
        let (host, worker) = (self.host.clone(), resources.worker.clone());
        let readiness = status.ready.clone();
        resources.load = Some(TokioSpawner::current().spawn(async move {
            let mut interval = tokio::time::interval(crate::UPDATE_LOAD_INTERVAL);
            loop {
                interval.tick().await;
                if let Some(worker) = &worker {
                    let snapshot = worker.snapshot();
                    status.set_load(snapshot.worker_load);
                    if let Some(target) = snapshot.target_idle_processes {
                        host.proxy
                            .idle_processes(target)
                            .await
                            .map_err(|error| error.to_string())?;
                    }
                }
                host.proxy
                    .worker_load(f64::from(status.load()))
                    .await
                    .map_err(|error| error.to_string())?;
            }
        }));
        Ok(readiness)
    }

    fn drain_resources(
        &self,
        worker: Option<WorkerClient>,
        timeout: Option<Duration>,
    ) -> BoxFuture<'static, Result<(), String>> {
        let host = self.host.clone();
        async move {
            let drain = async {
                host.close_websockets().await;
                match worker {
                    Some(worker) => worker.drain(timeout).await,
                    None => host.proxy.join_jobs().await.map_err(|error| error.to_string()),
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

async fn pending_drain(
    drain: &mut Option<BoxFuture<'static, Result<(), String>>>,
) -> Result<(), String> {
    match drain {
        Some(drain) => drain.await,
        None => std::future::pending().await,
    }
}

async fn pending_worker(
    worker: &mut Option<OwnedTask<Result<(), String>>>,
) -> Result<Result<(), String>, TaskError> {
    match worker {
        Some(worker) => worker.await,
        None => std::future::pending().await,
    }
}

#[derive(Default)]
struct Resources {
    setup: Option<lk_agents_net::blocking_io::Job<Result<(Option<String>, Option<f32>), String>>>,
    worker: Option<WorkerClient>,
    connection: Option<OwnedTask<Result<(), String>>>,
    load: Option<OwnedTask<Result<(), String>>>,
    health: Option<OwnedTask<Result<(), String>>>,
    metrics: Option<OwnedTask<Result<(), String>>>,
}

impl Resources {
    fn close(
        self,
        host: ServerCallbacks,
    ) -> FuturesUnordered<BoxFuture<'static, (ServerResource, Result<(), String>)>> {
        let closing = FuturesUnordered::new();
        closing.push(
            async move {
                (ServerResource::Host, host.proxy.close().await.map_err(|error| error.to_string()))
            }
            .boxed(),
        );
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
                task.cancel();
            }
            closing.push(
                async move {
                    let result = match task {
                        Some(task) => match task.await {
                            Ok(result) => result,
                            Err(TaskError::Cancelled) if resource == ServerResource::Load => Ok(()),
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

#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod tests;

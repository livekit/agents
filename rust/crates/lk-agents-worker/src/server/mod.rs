//! `AgentServer` around the worker connection (`worker.py`): its options resolved once against the
//! environment, the health listener and its `/worker` record, the job `simulate_job` hands the
//! host, and the lifecycle (start, load, drain, close) the binding runs. The host keeps the job
//! processes and the user's callbacks.

type SetupOutcome = Result<(Option<String>, Option<f32>), String>;

/// The most idle job processes a production server keeps when the count comes from its CPUs.
const MAX_DISCOVERED_IDLE_PROCESSES: u32 = 4;

pub mod host;
use crate::ipc::telemetry;
use crate::{DrainError, Worker, WorkerClient, WorkerCommand, WorkerHost, WorkerNotification};
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use host::ServerCallbacks;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::logging::LogWriter;
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskError};
use lk_agents_net::task::TokioSpawner;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, watch};
mod health;
mod http;
mod public;
pub use health::{HealthProbe, HealthState, ServerStatus, serve_health, serve_metrics};
#[cfg(test)]
use health::{health_route, metrics_route};
pub use http::bind_listener;
pub use public::{PublicRoutes, serve_public};
mod settings;
pub use settings::{
    Bind, DEFAULT_LOAD_THRESHOLD, Environment, Listeners, ServerSettings, Serving,
    endpoint_url_base, prepare_prometheus_dir,
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
        job: crate::worker::job_info(&job),
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
pub fn job_decision(req: &JobRequest, has_rtc: bool, logging: LogWriter) -> JobDecision {
    if !has_rtc {
        logging.emit(
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
    Public,
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
    pub endpoints: Vec<proto::agent_http::AgentEndpoint>,
    pub transport: crate::TransportConfig,
    /// Where the endpoints are reached, written once the server knows.
    pub serving: Arc<parking_lot::Mutex<Serving>>,
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

type Outcome<E = String> = watch::Sender<Option<Result<(), E>>>;

#[derive(Clone)]
pub struct ServerRuntime {
    host: ServerCallbacks,
    blocking_io: Option<lk_agents_net::blocking_io::BlockingIo>,
    control: watch::Sender<Control>,
    receiver: Arc<Mutex<Option<watch::Receiver<Control>>>>,
    snapshot: watch::Sender<ServerSnapshot>,
    outcome: Outcome,
    cleanup_outcome: Outcome,
    drain_outcome: Outcome<DrainError>,
    /// The lifecycle task `run` starts, outliving a cancelled `run` caller; `close` takes and
    /// awaits it unless a missed cleanup deadline left it holding the stragglers.
    serving: Arc<parking_lot::Mutex<Option<OwnedTask>>>,
}

impl ServerRuntime {
    pub(crate) fn logger(&self) -> LogWriter {
        self.host.logging
    }

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

    pub async fn drain(&self, timeout: Option<Duration>) -> Result<(), DrainError> {
        match self.snapshot().phase {
            ServerPhase::Idle | ServerPhase::Closed => return Ok(()),
            ServerPhase::Closing => {
                return wait_outcome(&self.cleanup_outcome).await.map_err(DrainError::Cleanup);
            }
            _ => {}
        }
        self.control.send_if_modified(|control| {
            if control.drain.is_some() {
                return false;
            }
            control.drain = Some(timeout);
            true
        });
        wait_drain_outcome(&self.drain_outcome).await
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
    fn settle<E>(outcome: &Outcome<E>, result: Result<(), E>) {
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
        let (started, drain_requested) =
            Self::await_startup(self.start(&start, &stop, &mut resources), &mut control).await;
        let mut running = RunningServer { drain_requested, ..RunningServer::default() };
        match started {
            Some(Ok(status)) => {
                running.started(&self, status, &control, &resources.worker);
                running.run(&self, &mut control, &mut resources).await;
            }
            Some(Err(error)) => running.failure = Some(error),
            None => {}
        }
        let failure = running.close(&self);
        stop.send_replace(true);
        self.close_resources(resources, failure, cleanup_timeout).await;
    }

    /// Startup's result, or `None` when a close (or the end of control) comes first; and
    /// whether a drain was requested meanwhile.
    async fn await_startup(
        startup: impl Future<Output = Result<Arc<ServerStatus>, String>>,
        control: &mut watch::Receiver<Control>,
    ) -> (Option<Result<Arc<ServerStatus>, String>>, bool) {
        tokio::pin!(startup);
        let mut drain_requested = false;
        loop {
            tokio::select! {
                biased;
                changed = control.changed() => {
                    if changed.is_err() { return (None, drain_requested); }
                    let request = control.borrow_and_update().clone();
                    drain_requested |= request.drain.is_some();
                    if request.close { return (None, drain_requested); }
                }
                result = &mut startup => return (Some(result), drain_requested),
            }
        }
    }

    /// Stops every resource, reporting the outcome when all are closed or at the cleanup
    /// deadline, whichever comes first.
    async fn close_resources(
        &self,
        resources: Resources,
        mut failure: Option<String>,
        cleanup_timeout: Duration,
    ) {
        let mut closing = resources.close(self.host.clone());
        let mut remaining: std::collections::BTreeSet<ServerResource> = [
            ServerResource::Host,
            ServerResource::Setup,
            ServerResource::Worker,
            ServerResource::Load,
            ServerResource::Public,
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
                    self.host.logging.emit(
                        log::Level::Error,
                        format!("server cleanup deadline missed; runtime retains ownership resources={resources:?}"),
                    );
                    self.snapshot.send_modify(|snapshot| snapshot.incomplete_cleanup = resources);
                    self.finish(&failure, &cleanup_failure);
                }
                Some((resource, result)) = closing.next() => {
                    remaining.remove(&resource);
                    if let Err(error) = result {
                        self.host.logging.emit(log::Level::Error, format!("server resource cleanup failed resource={resource:?} error={error}"));
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
    ) -> Result<Arc<ServerStatus>, String> {
        let mut settings = start.settings.clone();
        if !start.has_rtc && !start.has_text_doors {
            return Err("No RTC session entrypoint has been registered".into());
        }
        if !start.unregistered {
            settings.check_registration()?;
        }
        settings.check_routes(!start.endpoints.is_empty())?;
        if settings.cloud() {
            settings.load_threshold = DEFAULT_LOAD_THRESHOLD;
        }
        let host = self.host.configured(start.has_rtc, start.has_load_fnc && !settings.cloud());
        let blocking_io =
            self.blocking_io.clone().unwrap_or_else(lk_agents_net::blocking_io::blocking_io);
        self.prepare_host(&mut settings, &blocking_io, resources).await?;
        if !start.has_rtc {
            settings.num_idle_processes = Some(crate::EnvOption::Fixed(0));
        }
        let status = Arc::new(ServerStatus::default());
        let probe = Arc::new(host.clone());
        let local = start.devmode || start.unregistered;
        let logging = self.host.logging;
        resources.health = self.open_health(&settings, local, &status, &probe, stop).await?;
        resources.metrics = self.open_metrics(&settings, local, probe, stop).await?;
        // Decided before the job processes start, which inherit it: they delegate too.
        let cloud_routed = !start.unregistered
            && (settings.cloud() || lk_agents_net::strings::is_cloud_url(&settings.ws_url));
        if start.endpoints.is_empty() {
            logging
                .emit(log::Level::Info, "public listener off: no endpoints or HTTP routes".into());
        } else if cloud_routed {
            *start.serving.lock() = Serving::Cloud;
            logging.emit(
                log::Level::Info,
                "public listener off: LiveKit Cloud routes the agent's endpoints".into(),
            );
        } else {
            resources.public = Some(self.open_public(&settings, start, local, stop).await?);
        }
        settings.endpoints_url = settings
            .delegation_url(&start.serving.lock(), "")
            .map(|url| url.trim_end_matches('/').to_owned())
            .unwrap_or_default();
        if settings.multiprocessing_fallback {
            self.host.logging.emit(
                log::Level::Warn,
                "the forkserver is used on Linux only, job processes are spawned".into(),
            );
        }
        host.start(settings.clone(), start.devmode, start.has_rtc).await?;
        if start.unregistered {
            status.registered.store(true, Ordering::Release);
            status.connected.store(true, Ordering::Release);
        } else {
            let (worker, handle) = new_worker(&settings, start, host, &blocking_io);
            resources.worker = Some(handle.clone());
            self.snapshot.send_modify(|snapshot| snapshot.worker = Some(handle.clone()));
            let registration = Registration {
                notifications: handle.subscribe(),
                logging,
                snapshots: self.snapshot.clone(),
                status: status.clone(),
                has_routes: !start.endpoints.is_empty(),
                cloud_routed,
            };
            resources.connection =
                Some(TokioSpawner::current().spawn(run_worker_connection(worker, registration)));
        }
        let reporter = LoadReporter {
            host: self.host.clone(),
            worker: resources.worker.clone(),
            status: status.clone(),
        };
        resources.load = Some(TokioSpawner::current().spawn(reporter.run()));
        Ok(status)
    }

    /// Logging set up and the Prometheus directory prepared on the blocking pool, and the idle
    /// process count taken from the CPUs when it is not configured.
    async fn prepare_host(
        &self,
        settings: &mut ServerSettings,
        blocking_io: &lk_agents_net::blocking_io::BlockingIo,
        resources: &mut Resources,
    ) -> Result<(), String> {
        let configured = settings.prometheus_multiproc_dir.clone();
        let discover_cpus = settings.num_idle_processes.is_none();
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
        let logging = self.host.logging;
        resources.setup = Some(blocking_io.spawn_admitted(permit, move || {
            logging.initialize()?;
            Ok((
                prepare_prometheus_dir(configured.as_deref(), logging),
                discover_cpus.then(|| crate::CpuMonitor::new().cpu_count()),
            ))
        }));
        let setup = resources.setup.as_mut().expect("setup owned before awaiting").await;
        resources.setup.take();
        let (directory, cpus) = setup.map_err(|error| error.to_string())??;
        settings.prometheus_multiproc_dir = directory;
        if let Some(cpus) = cpus {
            let prod = (cpus.ceil() as u32).min(MAX_DISCOVERED_IDLE_PROCESSES);
            settings.num_idle_processes = Some(crate::EnvOption::Split { dev: 0, prod });
        }
        Ok(())
    }

    /// The health listener, off when simulation runs share a host.
    async fn open_health(
        &self,
        settings: &ServerSettings,
        local: bool,
        status: &Arc<ServerStatus>,
        probe: &Arc<ServerCallbacks>,
        stop: &watch::Sender<bool>,
    ) -> Result<Option<OwnedTask<Result<(), String>>>, String> {
        let logging = self.host.logging;
        if settings.simulation {
            logging
                .emit(log::Level::Info, "health listener off: simulation runs share a host".into());
            return Ok(None);
        }
        let (listener, url) = bind(&settings.listeners.health(local)).await?;
        self.snapshot.send_modify(|snapshot| {
            snapshot.http_port = listener.local_addr().map_or(0, |a| a.port())
        });
        logging.emit(log::Level::Info, format!("health listener url={url} paths=/,/ready,/worker"));
        Ok(Some(TokioSpawner::current().spawn(serve_health(
            listener,
            Arc::new(settings.clone()),
            status.clone(),
            probe.clone(),
            stop.subscribe(),
        ))))
    }

    /// The Prometheus listener, when one is configured.
    async fn open_metrics(
        &self,
        settings: &ServerSettings,
        local: bool,
        probe: Arc<ServerCallbacks>,
        stop: &watch::Sender<bool>,
    ) -> Result<Option<OwnedTask<Result<(), String>>>, String> {
        let logging = self.host.logging;
        let Some(address) = settings.listeners.prometheus(local) else {
            logging.emit(log::Level::Info, "prometheus listener off: not configured".into());
            return Ok(None);
        };
        let (listener, url) = bind(&address).await?;
        logging.emit(log::Level::Info, format!("prometheus listener url={url}/metrics"));
        Ok(Some(TokioSpawner::current().spawn(serve_metrics(listener, probe, stop.subscribe()))))
    }

    /// The worker's own endpoints: the console's, unauthenticated and on loopback only, or a
    /// self-hosted server's, where the endpoint verifies each caller's token.
    async fn open_public(
        &self,
        settings: &ServerSettings,
        start: &ServerStart,
        local: bool,
        stop: &watch::Sender<bool>,
    ) -> Result<OwnedTask<Result<(), String>>, String> {
        let handler =
            self.host.mounted.get().cloned().ok_or("the endpoint server is not mounted")?;
        let unauthenticated = start.unregistered;
        let mut address = settings.listeners.public(local);
        if unauthenticated {
            address.host = "127.0.0.1".into();
        }
        let (listener, url) = bind(&address).await?;
        let prefix = settings.endpoint_prefix();
        *start.serving.lock() = Serving::Local { base: format!("{url}{prefix}") };
        let logging = self.host.logging;
        if unauthenticated {
            logging.emit(
                log::Level::Warn,
                format!(
                    "public listener url={url}{prefix} is UNAUTHENTICATED: anything on this \
                     machine can call the agent's endpoints (console only, loopback)"
                ),
            );
        } else {
            logging.emit(
                log::Level::Info,
                format!("public listener url={url}{prefix} auth=bearer token"),
            );
        }
        let routes = Arc::new(PublicRoutes {
            prefix,
            routes: start.endpoints.clone(),
            unauthenticated,
            proxy_headers: settings.listeners.public.proxy_headers,
        });
        Ok(TokioSpawner::current().spawn(serve_public(listener, routes, handler, stop.subscribe())))
    }

    fn drain_resources(
        &self,
        worker: Option<WorkerClient>,
        timeout: Option<Duration>,
    ) -> BoxFuture<'static, Result<(), DrainError>> {
        let host = self.host.clone();
        async move {
            let jobs = async {
                match worker {
                    Some(worker) => worker.drain(timeout).await,
                    None => {
                        host.proxy.join_jobs(&telemetry::Cx::NONE).await.map_err(DrainError::Host)
                    }
                }
            };
            // Endpoint exchanges in flight finish beside the jobs, under the same deadline.
            let drain = async { futures_util::future::join(host.drain_endpoints(), jobs).await.1 };
            match timeout {
                Some(timeout) => {
                    tokio::time::timeout(timeout, drain).await.map_err(|_| DrainError::TimedOut)?
                }
                None => drain.await,
            }
        }
        .boxed()
    }
}

/// The host's `started` call in flight, with the status it marks started.
type StartedNotification<'a> = (BoxFuture<'a, Result<(), HostError>>, Arc<ServerStatus>);

/// The started server's state until it closes: the drain in flight and the host's `started`
/// notification, with its health status.
#[derive(Default)]
struct RunningServer<'a> {
    failure: Option<String>,
    drain_requested: bool,
    drained: bool,
    drain: Option<BoxFuture<'static, Result<(), DrainError>>>,
    started_notification: Option<StartedNotification<'a>>,
    health: Option<Arc<ServerStatus>>,
}

impl<'a> RunningServer<'a> {
    fn started(
        &mut self,
        server: &'a ServerRuntime,
        status: Arc<ServerStatus>,
        control: &watch::Receiver<Control>,
        worker: &Option<WorkerClient>,
    ) {
        server.set_phase(ServerPhase::Running);
        self.health = Some(status.clone());
        if self.drain_requested {
            let timeout = control.borrow().drain.flatten();
            self.begin_drain(server, worker, timeout);
        }
        self.started_notification = Some((server.host.proxy.started(&telemetry::Cx::NONE), status));
    }

    /// Runs until a close, the end of control, a failed `started` notification or the loss of
    /// the worker connection.
    async fn run(
        &mut self,
        server: &'a ServerRuntime,
        control: &mut watch::Receiver<Control>,
        resources: &mut Resources,
    ) {
        loop {
            let flow = tokio::select! {
                biased;
                changed = control.changed() => match changed {
                    Ok(()) => {
                        let request = control.borrow_and_update().clone();
                        self.on_control(server, request, &resources.worker)
                    }
                    Err(_) => ControlFlow::Break(()),
                },
                result = pending_notification(&mut self.started_notification), if self.started_notification.is_some() => {
                    self.on_started_notified(result)
                }
                result = pending_drain(&mut self.drain), if self.drain.is_some() => {
                    self.on_drained(server, result)
                }
                result = pending_worker(&mut resources.connection), if resources.connection.is_some() => {
                    self.on_connection_lost(resources, result)
                }
            };
            if flow.is_break() {
                return;
            }
        }
    }

    fn on_control(
        &mut self,
        server: &ServerRuntime,
        request: Control,
        worker: &Option<WorkerClient>,
    ) -> ControlFlow<()> {
        if request.close {
            return ControlFlow::Break(());
        }
        if request.drain.is_some() && !self.drain_requested {
            self.begin_drain(server, worker, request.drain.flatten());
        }
        ControlFlow::Continue(())
    }

    fn begin_drain(
        &mut self,
        server: &ServerRuntime,
        worker: &Option<WorkerClient>,
        timeout: Option<Duration>,
    ) {
        self.drain_requested = true;
        if let Some(status) = &self.health {
            status.draining.store(true, Ordering::Release);
        }
        server.set_phase(ServerPhase::Draining);
        self.drain = Some(server.drain_resources(worker.clone(), timeout));
    }

    fn on_started_notified(&mut self, result: Result<(), HostError>) -> ControlFlow<()> {
        let (_, status) = self.started_notification.take().expect("notification pending");
        if let Err(error) = result {
            self.failure = Some(error.to_string());
            return ControlFlow::Break(());
        }
        status.heartbeat();
        status.started.store(true, Ordering::Release);
        ControlFlow::Continue(())
    }

    fn on_drained(
        &mut self,
        server: &ServerRuntime,
        result: Result<(), DrainError>,
    ) -> ControlFlow<()> {
        self.drain = None;
        self.drained = true;
        ServerRuntime::settle(&server.drain_outcome, result);
        ControlFlow::Continue(())
    }

    fn on_connection_lost(
        &mut self,
        resources: &mut Resources,
        result: Result<Result<(), String>, TaskError>,
    ) -> ControlFlow<()> {
        // The connection is gone: nothing of it is left to stop.
        resources.connection = None;
        resources.worker = None;
        self.failure = Some(match result.unwrap_or_else(|error| Err(error.to_string())) {
            Ok(()) => "worker connection stopped".into(),
            Err(error) => format!("{:?}: {error}", ServerResource::Worker),
        });
        ControlFlow::Break(())
    }

    /// Enters Closing: an unfinished drain fails as closed, and the drain and the `started`
    /// notification are dropped. Returns the failure that ended the run.
    fn close(self, server: &ServerRuntime) -> Option<String> {
        server.set_phase(ServerPhase::Closing);
        if self.drain_requested && !self.drained {
            ServerRuntime::settle(&server.drain_outcome, Err(DrainError::Closed));
        }
        drop(self.drain);
        drop(self.started_notification);
        self.failure
    }
}

/// The worker connection over the configured transport, and its client.
fn new_worker(
    settings: &ServerSettings,
    start: &ServerStart,
    host: ServerCallbacks,
    blocking_io: &lk_agents_net::blocking_io::BlockingIo,
) -> (Worker, WorkerClient) {
    let (config, connection) = settings.worker(start.devmode, start.endpoints.clone());
    let transport = match &start.transport {
        crate::TransportConfig::WebSocket { heartbeat } => {
            crate::TransportConfig::WebSocket { heartbeat: *heartbeat }
        }
        crate::TransportConfig::WebTransport { insecure } => {
            crate::TransportConfig::WebTransport { insecure: *insecure }
        }
    };
    Worker::new(config, connection, transport, Arc::new(host), blocking_io.clone())
}

/// Mirrors the worker's registrations into the snapshot and the health status.
struct Registration {
    notifications: tokio::sync::broadcast::Receiver<WorkerNotification>,
    logging: LogWriter,
    snapshots: watch::Sender<ServerSnapshot>,
    status: Arc<ServerStatus>,
    has_routes: bool,
    cloud_routed: bool,
}

impl Registration {
    async fn forward(&mut self) {
        while let Ok(event) = self.notifications.recv().await {
            match event {
                WorkerNotification::Registered(info) => self.registered(info),
                WorkerNotification::Stopped(_) => break,
                _ => {}
            }
        }
    }

    fn registered(&self, info: crate::RegisteredInfo) {
        let cloud_routed = self.cloud_routed;
        if self.has_routes && info.cloud != cloud_routed {
            self.logging.emit(
                log::Level::Warn,
                format!(
                    "the server's edition (cloud={}) differs from its URL; \
                     endpoints are routed as cloud={cloud_routed}",
                    info.cloud
                ),
            );
        }
        self.snapshots.send_modify(|snapshot| snapshot.worker_id = info.worker_id);
        self.status.registered.store(true, Ordering::Release);
        self.status.connected.store(true, Ordering::Release);
    }
}

/// Runs the worker while forwarding its registrations; a failed run marks the server failed.
async fn run_worker_connection(
    mut worker: Worker,
    mut registration: Registration,
) -> Result<(), String> {
    let run = worker.run();
    tokio::pin!(run);
    let result = tokio::select! {
        result = &mut run => result,
        () = registration.forward() => run.await,
    }
    .map_err(|error| error.to_string());
    if result.is_err() {
        registration.status.failed.store(true, Ordering::Release);
    }
    result
}

/// Reports the worker's load and idle process target to the host on every load interval.
struct LoadReporter {
    host: ServerCallbacks,
    worker: Option<WorkerClient>,
    status: Arc<ServerStatus>,
}

impl LoadReporter {
    async fn run(self) -> Result<(), String> {
        let mut interval = tokio::time::interval(crate::UPDATE_LOAD_INTERVAL);
        loop {
            interval.tick().await;
            self.report_once().await?;
        }
    }

    async fn report_once(&self) -> Result<(), String> {
        let (host, status) = (&self.host, &self.status);
        if let Some(worker) = &self.worker {
            let snapshot = worker.snapshot();
            status.set_load(snapshot.worker_load);
            status.connected.store(snapshot.registered, Ordering::Release);
            if let Some(target) = snapshot.target_idle_processes {
                host.proxy
                    .idle_processes(&telemetry::Cx::NONE, target)
                    .await
                    .map_err(|error| error.to_string())?;
            }
        }
        // The host answering is its heartbeat: `/` reads it without waiting on the host.
        host.proxy
            .worker_load(&telemetry::Cx::NONE, f64::from(status.load()))
            .await
            .map_err(|error| error.to_string())?;
        status.heartbeat();
        Ok(())
    }
}

/// A listener bound at `address`, and the URL it is reached at from this host.
async fn bind(address: &Bind) -> Result<(tokio::net::TcpListener, String), String> {
    let (listener, port) =
        bind_listener(&address.host, address.port).await.map_err(|error| error.to_string())?;
    let host = match address.host.as_str() {
        "" | "0.0.0.0" => "127.0.0.1".to_owned(),
        "::" | "[::]" => "[::1]".to_owned(),
        host if host.contains(':') && !host.starts_with('[') => format!("[{host}]"),
        host => host.to_owned(),
    };
    Ok((listener, format!("http://{host}:{port}")))
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

async fn wait_drain_outcome(outcome: &Outcome<DrainError>) -> Result<(), DrainError> {
    outcome
        .subscribe()
        .wait_for(Option::is_some)
        .await
        .map_err(|_| DrainError::OutcomeUnavailable)?
        .clone()
        .ok_or(DrainError::OutcomeUnavailable)?
}

async fn pending_drain(
    drain: &mut Option<BoxFuture<'static, Result<(), DrainError>>>,
) -> Result<(), DrainError> {
    match drain {
        Some(drain) => drain.await,
        None => std::future::pending().await,
    }
}

async fn pending_notification(
    notification: &mut Option<StartedNotification<'_>>,
) -> Result<(), HostError> {
    match notification {
        Some((started, _)) => started.await,
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
    setup: Option<lk_agents_net::blocking_io::Job<SetupOutcome>>,
    worker: Option<WorkerClient>,
    connection: Option<OwnedTask<Result<(), String>>>,
    load: Option<OwnedTask<Result<(), String>>>,
    public: Option<OwnedTask<Result<(), String>>>,
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
                (
                    ServerResource::Host,
                    host.proxy.close(&telemetry::Cx::NONE).await.map_err(|error| error.to_string()),
                )
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
            (ServerResource::Public, self.public),
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
#[path = "../server_tests.rs"]
mod tests;

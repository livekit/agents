#![allow(clippy::disallowed_types, clippy::disallowed_methods, reason = "migrating to signal")]
use lk_agents_net::clock::unix_now_ms;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::executor::{Connection, Exit, Runner, RunnerExecutor};
use super::ipc_channel::{ChannelError, SendError, VERSION};
use super::wire::{Hello, StartJob, ToChild, ToParent};
use lk_agents_abi::worker::ipc::JobRunnerOptions;

#[path = "supervisor_lifecycle.rs"]
mod lifecycle;
use lifecycle::*;

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;

/// What a runner's diagnostics name: its process and the job it runs.
#[derive(Clone, Debug, Default)]
pub struct RunnerFields {
    pub pid: Option<u32>,
    pub job_id: Option<String>,
    pub room: Option<String>,
    pub room_id: Option<String>,
}

/// A line on the `livekit.agents` logger naming the runner (`pid`, `job_id`, `room`, `room_id`)
/// after its own key-values.
macro_rules! runner_log {
    ($runner:expr, $level:expr $(, $key:ident $(:$capture:tt)? = $value:expr)* ; $($message:tt)+) => {{
        let runner: $crate::ipc::RunnerFields = $runner;
        log::log!(
            target: lk_agents_abi::LOG_TARGET,
            $level,
            $($key $(:$capture)? = $value,)*
            pid = runner.pid,
            job_id = runner.job_id.as_deref(),
            room = runner.room.as_deref(),
            room_id = runner.room_id.as_deref();
            $($message)+
        )
    }};
}
pub(crate) use runner_log;

fn runner_name(opts: &JobRunnerOptions) -> &str {
    if opts.thread { "job thread" } else { "process" }
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("initialization already requested")]
    AlreadyInitializing,
    #[error("duplex closed")]
    Closed,
    #[error("initialization timed out")]
    TimedOut,
    #[error("first message must be Ready")]
    Unexpected,
    #[error("process initialization failed: {0}")]
    Failed(String),
    #[error("the job process speaks protocol version {0}, the worker {VERSION}")]
    Version(u32),
    #[error("IPC protocol failed: {0}")]
    Protocol(Arc<ProtocolError>),
}

/// Why the supervisor gave up on the connection: the channel failed, or the child said what the
/// protocol does not allow then.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error(transparent)]
    Channel(#[from] ChannelError),
    #[error("unexpected message: {0}")]
    Unexpected(String),
}

impl From<&ProtocolError> for lk_agents_abi::worker::ipc::IpcFailure {
    fn from(error: &ProtocolError) -> Self {
        let message = error.to_string();
        match error {
            ProtocolError::Channel(ChannelError::Closed | ChannelError::Truncated) => {
                let kind = format!("{:?}", std::io::ErrorKind::UnexpectedEof);
                Self::Io { kind, os_error: None, message }
            }
            ProtocolError::Channel(ChannelError::Io { kind, os_error, .. }) => {
                Self::Io { kind: format!("{kind:?}"), os_error: *os_error, message }
            }
            ProtocolError::Channel(ChannelError::Malformed(cause)) => {
                Self::Malformed { message: cause.clone() }
            }
            ProtocolError::Channel(ChannelError::TooLarge(size)) => {
                Self::TooLarge { size: *size as u64 }
            }
            ProtocolError::Channel(ChannelError::Version(received)) => {
                Self::Version { received: *received, expected: VERSION }
            }
            ProtocolError::Unexpected(message) => Self::Unexpected { message: message.clone() },
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StartJobError {
    #[error("invalid job launch fields: {0}")]
    InvalidFields(String),
    #[error("the process is not initialized")]
    NotReady,
    #[error("the process already accepted a job or is accepting one")]
    AlreadyRunning,
    #[error("the IPC connection is closed")]
    Closed,
    #[error("IPC protocol failed: {0}")]
    Protocol(Arc<ProtocolError>),
    #[error("the child refused the job: {0}")]
    Rejected(String),
    #[error("the child did not acknowledge the job before its deadline")]
    Timeout,
}

pub fn validate_start_job(job: &StartJob) -> Result<(), StartJobError> {
    let info = &job.info;
    for (field, value) in [
        ("received_at", info.received_at),
        ("accepted_at", info.accepted_at),
        ("assigned_at", info.assigned_at),
        ("launched_at", Some(info.launched_at)),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(StartJobError::InvalidFields(format!(
                "{field} must be finite and nonnegative"
            )));
        }
    }
    Ok(())
}

pub use lk_agents_abi::secs;

/// `round(x, digits)`: Python rounds the exact binary value half to even, as formatting does.
fn round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// Converts the seconds of [`JobRunnerOptions::high_ping_threshold`] to a pong's milliseconds.
const MILLIS_PER_SECOND: f64 = 1000.0;

/// What the runner still reports once it exited: a refusal, or its own end.
fn read_after_exit(msg: &ToParent) -> bool {
    matches!(
        msg,
        ToParent::Ready { error: Some(_) }
            | ToParent::JobAccepted { error: Some(_), .. }
            | ToParent::ShuttingDown
            | ToParent::Exiting { .. }
    )
}

/// Rate-limits the high memory warning: once per cooldown, or sooner when usage grew.
#[derive(Default)]
pub struct WarnGate {
    last: Option<(Duration, f64)>,
}

impl WarnGate {
    pub fn should_warn(&mut self, memory_mb: f64, now: Duration) -> bool {
        let emit = match self.last {
            None => true,
            Some((at, mb)) => {
                now.saturating_sub(at) >= MEMORY_WARN_COOLDOWN
                    || memory_mb - mb >= MEMORY_WARN_RESET_DELTA_MB
            }
        };
        if emit {
            self.last = Some((now, memory_mb));
        }
        emit
    }
}

/// What the supervisor's diagnostics name and whom it tells the runner is released.
pub trait SupervisorCallbacks: Send + Sync + 'static {
    fn runner(&self) -> RunnerFields {
        RunnerFields::default()
    }
    /// The runner ended and its supervision is over: the host's handle can go.
    fn release(&self) {}
}

const QUEUE: usize = 16;

enum Command {
    Supervise,
    Initialize(Hello, oneshot::Sender<Result<Duration, InitError>>),
    StartJob(Box<StartJob>, oneshot::Sender<Result<(), StartJobError>>),
}

/// Whether a driver input still applies once the runner exited.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AfterExit {
    /// What the runner did or said: still read after its exit.
    Read,
    /// A request to the runner: moot once it exited.
    Dropped,
}

#[derive(Clone, Copy, Default)]
struct Control {
    shutdown: bool,
    kill: bool,
}

#[derive(Clone, Default)]
struct Status {
    /// Whether a runner was launched, once the launch settled, with why not.
    launched: Option<Result<(), HostError>>,
    failure: Option<lk_agents_abi::worker::ipc::IpcFailure>,
    /// How the runner ended, once it did.
    exit: Option<Exit>,
    incomplete: bool,
    cleaned: bool,
}

/// Owns one supervisor actor. Control is coalesced independently of bounded work submission.
#[derive(Clone)]
pub struct RunnerSupervisor {
    commands: mpsc::Sender<Command>,
    control: watch::Sender<Control>,
    status: watch::Receiver<Status>,
    /// The actor's owner; dropping the supervisor does not cancel it.
    _driver: Arc<TaskSet>,
}

impl RunnerSupervisor {
    /// Launches a runner through `executor` and supervises it. Needs a runtime context.
    pub fn new<H: SupervisorCallbacks>(
        executor: Arc<dyn RunnerExecutor>,
        host: H,
        opts: JobRunnerOptions,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
    ) -> Self {
        let spawner = TokioSpawner::current();
        let (commands, commands_rx) = mpsc::channel(QUEUE);
        let (control, control_rx) = watch::channel(Control::default());
        let (status_tx, status) = watch::channel(Status::default());
        let host: Arc<dyn SupervisorCallbacks> = Arc::new(host);
        // Self-owned: it ends once the runner ended and its channel settled.
        let driving = Arc::new(TaskSet::default());
        let run = async move {
            match executor.launch().await {
                Ok((runner, channel)) => {
                    drop(executor);
                    status_tx.send_modify(|status| status.launched = Some(Ok(())));
                    let driver = Driver::new(opts, host, blocking_io, runner, channel, status_tx);
                    driver.run(commands_rx, control_rx).await;
                }
                Err(cause) => refuse(cause, host, commands_rx, status_tx).await,
            }
        };
        driving.spawn(&spawner, "job runner supervisor", run);
        Self { commands, control, status, _driver: driving }
    }

    /// Once the launch settled: whether a runner was launched, with why not.
    pub async fn launched(&self) -> Result<(), HostError> {
        let mut status = self.status.clone();
        match status.wait_for(|status| status.launched.is_some()).await {
            Ok(status) => status.launched.clone().unwrap_or(Ok(())),
            Err(_) => Err(HostError::new("the supervisor ended before the launch settled")),
        }
    }

    /// Whether `stop()` killed the runner.
    pub fn killed(&self) -> bool {
        self.status.borrow().exit.is_some_and(|exit| exit.killed)
    }

    pub fn completion(&self) -> lk_agents_abi::worker::ipc::JobCompletion {
        let status = self.status.borrow();
        lk_agents_abi::worker::ipc::JobCompletion {
            exit_code: status.exit.and_then(|exit| i32::try_from(exit.code).ok()),
            failure: status.failure.clone(),
        }
    }

    /// Until the supervision is over, or its cleanup was reported incomplete.
    async fn settled(&self) {
        let mut status = self.status.clone();
        let _ = status.wait_for(|s| s.cleaned || s.incomplete).await;
    }

    pub async fn initialize(&self, hello: Hello) -> Result<Duration, InitError> {
        let (tx, rx) = oneshot::channel();
        self.commands.send(Command::Initialize(hello, tx)).await.map_err(|_| InitError::Closed)?;
        rx.await.unwrap_or(Err(InitError::Closed))
    }

    pub async fn supervise(&self) {
        let _ = self.commands.send(Command::Supervise).await;
    }

    pub async fn start_job(&self, job: StartJob) -> Result<(), StartJobError> {
        validate_start_job(&job)?;
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::StartJob(Box::new(job), reply))
            .await
            .map_err(|_| StartJobError::Closed)?;
        result.await.unwrap_or(Err(StartJobError::Closed))
    }

    /// Asks the runner to shut down, stops it when late, and waits for its cleanup.
    pub async fn shutdown(&self) {
        self.control.send_modify(|c| c.shutdown = true);
        self.settled().await;
    }

    /// Asks the runner to shut down and stops it at once, then waits for its cleanup.
    pub async fn kill(&self) {
        self.control.send_modify(|c| c.kill = true);
        self.settled().await;
    }

    /// Whether a launched runner ended (`false`: none was launched).
    pub async fn wait_exit(&self) -> bool {
        let mut status = self.status.clone();
        let _ = status.wait_for(|state| state.exit.is_some() || state.cleaned).await;

        status.borrow().exit.is_some()
    }

    /// Waits for the supervision's end: whether a launched runner ended.
    pub async fn wait_cleaned(&self) -> bool {
        let mut status = self.status.clone();
        let _ = status.wait_for(|state| state.cleaned).await;

        status.borrow().exit.is_some()
    }
}

/// A launch that started nothing: whatever was asked of the runner fails with why.
async fn refuse(
    cause: HostError,
    host: Arc<dyn SupervisorCallbacks>,
    mut commands: mpsc::Receiver<Command>,
    status: watch::Sender<Status>,
) {
    runner_log!(host.runner(), log::Level::Error; "launch failed before starting a runner: {cause}");
    status.send_modify(|status| status.launched = Some(Err(cause.clone())));
    commands.close();
    while let Some(command) = commands.recv().await {
        match command {
            Command::Initialize(_, reply) => {
                let _ = reply.send(Err(InitError::Failed(cause.to_string())));
            }
            Command::StartJob(_, reply) => {
                let _ = reply.send(Err(StartJobError::Closed));
            }
            Command::Supervise => {}
        }
    }
    host.release();
    status.send_modify(|status| status.cleaned = true);
}

/// The host's memory sample, off the runtime.
type Sampling = Option<lk_agents_net::blocking_io::Job<Option<f64>>>;

/// The supervisor's actor: the runner's state and every deadline, changed only here.
struct Driver {
    opts: JobRunnerOptions,
    origin: Instant,
    host: Arc<dyn SupervisorCallbacks>,
    blocking_io: lk_agents_net::blocking_io::BlockingIo,
    runner: Arc<dyn Runner>,
    channel: Arc<Connection>,
    init_reply: Option<oneshot::Sender<Result<Duration, InitError>>>,
    job_reply: Option<(u64, oneshot::Sender<Result<(), StartJobError>>)>,
    next_job_id: u64,
    status: watch::Sender<Status>,
    sampling: Sampling,
    lifecycle: Lifecycle,
    baseline_mb: Option<f64>,
    gate: WarnGate,
    protocol_failure: Option<Arc<ProtocolError>>,
}

impl Driver {
    fn new(
        opts: JobRunnerOptions,
        host: Arc<dyn SupervisorCallbacks>,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
        runner: Box<dyn Runner>,
        channel: Connection,
        status: watch::Sender<Status>,
    ) -> Self {
        let origin = Instant::now();
        Self {
            lifecycle: Lifecycle::Active {
                phase: Active::Starting { deadline: secs(opts.initialize_timeout) },
                health: Health::Idle,
            },
            opts,
            origin,
            host,
            blocking_io,
            runner: runner.into(),
            channel: Arc::new(channel),
            init_reply: None,
            job_reply: None,
            next_job_id: 0,
            status,
            sampling: None,
            baseline_mb: None,
            gate: WarnGate::default(),
            protocol_failure: None,
        }
    }

    async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        mut control: watch::Receiver<Control>,
    ) {
        let channel = self.channel.clone();
        let runner = self.runner.clone();
        let mut reader_open = true;
        let mut commands_open = true;
        let mut control_open = true;
        let mut seen = Control::default();
        while !self.cleaned() {
            if self.wake_at().is_some_and(|at| at <= self.now()) {
                self.tick();
                self.publish();
                continue;
            }
            let deadline = self.wake_at().map(|at| self.origin + at);
            tokio::select! {
                biased;
                () = async { match deadline {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                } } => continue,
                exit = runner.finished(), if !self.exited() => self.input(AfterExit::Read, |d| d.on_finished(exit)),
                changed = control.changed(), if control_open => match changed {
                    Ok(()) => self.on_control(*control.borrow_and_update(), &mut seen),
                    Err(_) => {
                        control_open = false;
                        self.input(AfterExit::Dropped, Self::end);
                    }
                },
                msg = channel.recv(), if reader_open => match msg {
                    Ok(msg) => self.input(AfterExit::Read, |d| d.on_message(msg)),
                    Err(reason) => {
                        reader_open = false;
                        self.input(AfterExit::Read, |d| d.on_reader_closed(reason));
                    }
                },
                command = commands.recv(), if commands_open => match command {
                    Some(command) => self.on_command(command),
                    None => {
                        commands_open = false;
                        self.input(AfterExit::Dropped, Self::end);
                    }
                },
                sample = async { match self.sampling.as_mut() {
                    Some(sample) => sample.await,
                    None => std::future::pending().await,
                } } => {
                    self.sampling = None;
                    self.input(AfterExit::Dropped, |d| d.on_sample(sample.ok().flatten()));
                }
            }
            self.publish();
        }
        // Started blocking samples cannot be aborted: retain the host until they settle.
        if let Some(mut sample) = self.sampling.take()
            && tokio::time::timeout(secs(self.opts.close_timeout), &mut sample).await.is_err()
        {
            runner_log!(self.host.runner(), log::Level::Error; "memory sample did not settle; retaining the runner's host");
            self.status.send_modify(|s| s.incomplete = true);
            let _ = sample.await;
        }
        if let Some(init) = self.init_reply.take() {
            let _ = init.send(Err(InitError::Closed));
        }
        if let Some((_, job)) = self.job_reply.take() {
            let _ = job.send(Err(StartJobError::Closed));
        }
        self.host.release();
        self.status.send_modify(|status| status.cleaned = true);
    }

    /// The handle's control changed to `next`: each of kill and shutdown acts once.
    fn on_control(&mut self, next: Control, seen: &mut Control) {
        if next.kill && !seen.kill {
            self.input(AfterExit::Dropped, Self::end);
        }
        if next.shutdown && !seen.shutdown {
            self.input(AfterExit::Dropped, Self::shutdown);
        }
        *seen = next;
    }

    fn on_command(&mut self, command: Command) {
        match command {
            Command::Supervise => self.input(AfterExit::Dropped, Self::supervise),
            Command::Initialize(hello, reply) => self.initialize_requested(hello, reply),
            Command::StartJob(job, reply) => self.job_requested(*job, reply),
        }
    }

    /// One initialize at a time, and none once the runner is stopping.
    fn initialize_requested(
        &mut self,
        hello: Hello,
        reply: oneshot::Sender<Result<Duration, InitError>>,
    ) {
        if self.stopping() {
            let _ = reply.send(Err(InitError::Closed));
            return;
        }
        if self.init_reply.is_some() {
            let _ = reply.send(Err(InitError::AlreadyInitializing));
            return;
        }
        self.init_reply = Some(reply);
        self.input(AfterExit::Dropped, |d| d.initialize(hello));
    }

    /// One job submission at a time, each with its own request id.
    fn job_requested(&mut self, job: StartJob, reply: oneshot::Sender<Result<(), StartJobError>>) {
        if self.job_reply.is_some() {
            let _ = reply.send(Err(StartJobError::AlreadyRunning));
            return;
        }
        self.next_job_id += 1;
        let request_id = self.next_job_id;
        self.job_reply = Some((request_id, reply));
        self.input(AfterExit::Dropped, |d| d.start_job(request_id, job));
    }

    fn now(&self) -> Duration {
        Instant::now().duration_since(self.origin)
    }

    fn exited(&self) -> bool {
        self.lifecycle.exit().is_some()
    }

    /// One input: what is due first (a reply at a deadline cannot revive an expired phase), then
    /// the input, which after the exit only [`AfterExit::Read`] inputs still reach.
    fn input(&mut self, after_exit: AfterExit, f: impl FnOnce(&mut Self)) {
        if self.cleaned() {
            return;
        }
        self.tick();
        if self.exited() && after_exit == AfterExit::Dropped {
            return;
        }
        f(self);
    }

    fn publish(&self) {
        let failure = self.protocol_failure.as_deref().map(Into::into);
        let exit = self.lifecycle.exit();
        self.status.send_modify(|s| {
            s.failure = failure;
            s.exit = exit;
        });
    }

    /// Closing or ended: the memory monitor's failures are no longer worth a log.
    fn stopping(&self) -> bool {
        !matches!(self.lifecycle, Lifecycle::Active { .. })
    }

    fn cleaned(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Exited { reader: Drain::Closed, .. })
    }

    fn wake_at(&self) -> Option<Duration> {
        let init = self.lifecycle.initialization().map(|init| init.deadline);
        let job = match self.lifecycle.job() {
            Job::Pending { deadline, .. } => Some(deadline),
            _ => None,
        };
        match self.lifecycle {
            Lifecycle::Active { phase, health } => {
                let startup = match phase {
                    Active::Starting { deadline } => Some(deadline),
                    _ => None,
                };
                let (ping, pong, sample) = match health {
                    Health::Idle => (None, None, None),
                    Health::Monitoring { next_ping, pong_deadline, next_sample, .. } => (
                        Some(next_ping),
                        pong_deadline,
                        next_sample.filter(|_| self.sampling.is_none()),
                    ),
                };
                [startup, init, job, ping, pong, sample].into_iter().flatten().min()
            }
            Lifecycle::Closing { phase, cleanup, .. } => {
                let cleanup = match cleanup {
                    Cleanup::At(at) => Some(at),
                    Cleanup::Reported => None,
                };
                [init, job, phase.deadline(), cleanup].into_iter().flatten().min()
            }
            Lifecycle::Exited { reader: Drain::Until(at), .. } => Some(at),
            Lifecycle::Exited { reader: Drain::Closed, .. } => None,
        }
    }

    fn reply_init(&mut self, result: Result<Duration, InitError>) {
        if let Some(init) = self.init_reply.take() {
            let _ = init.send(result);
        }
    }

    fn reply_job(&mut self, request_id: u64, result: Result<(), StartJobError>) {
        if self.job_reply.as_ref().is_some_and(|(pending, _)| *pending == request_id)
            && let Some((_, reply)) = self.job_reply.take()
        {
            let _ = reply.send(result);
        }
    }

    fn initialize(&mut self, hello: Hello) {
        let Lifecycle::Active { phase: Active::Starting { deadline }, health } = self.lifecycle
        else {
            self.reply_init(Err(InitError::AlreadyInitializing));
            return;
        };
        self.lifecycle = Lifecycle::Active {
            phase: Active::Initializing(Initialization { started: self.now(), deadline }),
            health,
        };
        runner_log!(self.host.runner(), log::Level::Info; "initializing {}", runner_name(&self.opts));
        self.send(ToChild::Hello(hello));
    }

    fn supervise(&mut self) {
        let Lifecycle::Active { health: Health::Idle, phase } = self.lifecycle else {
            return;
        };
        let now = self.now();
        self.lifecycle = Lifecycle::Active {
            phase,
            health: Health::Monitoring {
                next_ping: now,
                ping: Ping::Ready,
                pong_deadline: (self.opts.ping_timeout > 0.0)
                    .then(|| now + secs(self.opts.ping_timeout)),
                next_sample: (self.opts.memory_limit_mb > 0.0 || self.opts.memory_warn_mb > 0.0)
                    .then_some(now),
            },
        };
    }

    fn start_job(&mut self, request_id: u64, job: StartJob) {
        let failure = if let Some(failure) = &self.protocol_failure {
            Some(StartJobError::Protocol(failure.clone()))
        } else if self.stopping() {
            Some(StartJobError::Closed)
        } else if !matches!(self.lifecycle, Lifecycle::Active { phase: Active::Ready(_), .. }) {
            Some(StartJobError::NotReady)
        } else if !matches!(self.lifecycle.job(), Job::Idle) {
            Some(StartJobError::AlreadyRunning)
        } else {
            None
        };
        if let Some(error) = failure {
            self.reply_job(request_id, Err(error));
            return;
        }
        let message = ToChild::StartJob { request_id, job: Box::new(job) };
        match self.channel.try_send(message) {
            Err(SendError::TooLarge(size)) => self.reply_job(
                request_id,
                Err(StartJobError::InvalidFields(format!("a {size}-byte job is too large"))),
            ),
            sent => {
                let deadline = self.now() + secs(self.opts.initialize_timeout);
                self.set_job(Job::Pending { request_id, deadline });
                self.sent(sent);
            }
        }
    }

    /// The peer finishing after it said it was shutting down is the clean end; anything else is
    /// a protocol failure.
    fn on_reader_closed(&mut self, reason: ChannelError) {
        let shutting_down = match self.lifecycle {
            Lifecycle::Closing { phase, .. } => phase.child() == Child::ShuttingDown,
            Lifecycle::Exited { .. } => {
                matches!(self.lifecycle_reader_child(), Child::ShuttingDown)
            }
            _ => false,
        };
        if shutting_down && matches!(reason, ChannelError::Closed) {
            let deadline = self.now() + secs(self.opts.close_timeout);
            match &mut self.lifecycle {
                Lifecycle::Closing { phase, reader, .. } => {
                    *reader = Reader::Closed;
                    *phase = phase.acknowledged(deadline);
                }
                Lifecycle::Exited { reader, .. } => *reader = Drain::Closed,
                _ => {}
            }
        } else {
            self.fail_protocol(Arc::new(reason.into()));
        }
    }

    fn on_finished(&mut self, exit: Exit) {
        let now = self.now();
        let work = self.lifecycle.work();
        let reader = if matches!(work, Work::Uninitialized) || !self.lifecycle.reader_open() {
            Drain::Closed
        } else {
            Drain::Until(now + secs(self.opts.close_timeout))
        };
        let child = self.lifecycle_reader_child();
        self.lifecycle = Lifecycle::Exited { exit, reader, work, child };
        if matches!(reader, Drain::Closed) && self.take_initialization().is_some() {
            self.reply_init(Err(InitError::Closed));
        }
        if exit.code != 0 && !exit.killed {
            runner_log!(self.host.runner(), log::Level::Error; "{} exited with non-zero exit code {}", runner_name(&self.opts), exit.code);
        }
    }

    /// What is due now.
    fn tick(&mut self) {
        if self.cleaned() {
            return;
        }
        let now = self.now();
        if matches!(self.lifecycle, Lifecycle::Exited { reader: Drain::Until(at), .. } if at <= now)
        {
            self.fail_protocol(Arc::new(ProtocolError::Channel(ChannelError::Io {
                kind: std::io::ErrorKind::TimedOut,
                os_error: None,
                message: "IPC reader did not settle after the runner ended".into(),
            })));
        }
        if self.exited() {
            return;
        }
        if let Lifecycle::Closing { cleanup, .. } = &mut self.lifecycle
            && matches!(*cleanup, Cleanup::At(at) if at <= now)
        {
            *cleanup = Cleanup::Reported;
            runner_log!(self.host.runner(), log::Level::Error; "incomplete {} cleanup; retaining ownership", runner_name(&self.opts));
            self.status.send_modify(|s| s.incomplete = true);
        }
        if matches!(self.lifecycle, Lifecycle::Active { phase: Active::Starting { deadline }, .. } if deadline <= now)
        {
            runner_log!(self.host.runner(), log::Level::Error; "{} launch did not enter initialization before its deadline", runner_name(&self.opts));
            self.end();
        }
        if let Some(init) = self.lifecycle.initialization()
            && init.deadline <= now
        {
            self.take_initialization();
            self.reply_init(Err(InitError::TimedOut));
            self.end();
        }
        if let Job::Pending { request_id, deadline } = self.lifecycle.job()
            && deadline <= now
        {
            self.set_job(Job::Idle);
            self.reply_job(request_id, Err(StartJobError::Timeout));
            self.end();
        }
        if let Lifecycle::Active { health: Health::Monitoring { next_ping, ping, .. }, .. } =
            &mut self.lifecycle
            && *next_ping <= now
        {
            let send = matches!(*ping, Ping::Ready);
            *ping = Ping::Unanswered;
            *next_ping = now + secs(self.opts.ping_interval).max(Duration::from_millis(1));
            if send {
                self.send(ToChild::Ping { sent_ms: unix_now_ms().max(0) as u64 });
            }
        }
        if matches!(self.lifecycle, Lifecycle::Active { health: Health::Monitoring { pong_deadline: Some(at), .. }, .. } if at <= now)
        {
            runner_log!(self.host.runner(), log::Level::Error; "{} is unresponsive, stopping it", runner_name(&self.opts));
            self.end();
        }
        if let Lifecycle::Active { health: Health::Monitoring { next_sample, .. }, .. } =
            &mut self.lifecycle
            && next_sample.is_some_and(|at| at <= now)
            && self.sampling.is_none()
        {
            *next_sample = None;
            let runner = self.runner.clone();
            // Admission before any task: a saturated sampler retries at its next cadence.
            match self.blocking_io.try_spawn((), move || runner.memory_mb()) {
                Ok(receipt) => self.sampling = Some(receipt),
                Err(_) => *next_sample = Some(now + MEMORY_MONITOR_INTERVAL),
            }
        }
        match self.lifecycle {
            Lifecycle::Closing { phase: Close::Ack { deadline, .. }, .. } if deadline <= now => {
                let action = if self.stop() { "stopping it" } else { "waiting for it" };
                runner_log!(self.host.runner(), log::Level::Error; "{} did not ack shutdown in time, {action}", runner_name(&self.opts));
                if let Lifecycle::Closing { phase, .. } = &mut self.lifecycle {
                    *phase = phase.acknowledged(now + secs(self.opts.close_timeout));
                }
            }
            Lifecycle::Closing { phase, .. } if phase.deadline().is_some_and(|at| at <= now) => {
                let action = if self.stop() { "stopping it" } else { "waiting for it" };
                runner_log!(self.host.runner(), log::Level::Error; "{} did not exit in time, {action}", runner_name(&self.opts));
                if let Lifecycle::Closing { phase: close, .. } = &mut self.lifecycle {
                    *close = Close::Waiting(phase.child());
                }
            }
            _ => {}
        }
    }

    fn on_message(&mut self, msg: ToParent) {
        if self.exited() && !read_after_exit(&msg) {
            return;
        }
        let now = self.now();
        if let Some(init) = self.take_initialization() {
            self.on_init_reply(init.started, now, msg);
            return;
        }
        match msg {
            ToParent::Pong { sent_ms } => self.on_pong(sent_ms, now),
            ToParent::ShutdownAck => {
                if let Lifecycle::Closing { phase: phase @ Close::Ack { .. }, .. } =
                    &mut self.lifecycle
                {
                    *phase = phase.acknowledged(now + secs(self.opts.close_timeout));
                }
            }
            ToParent::ShuttingDown => self.on_shutting_down(now),
            ToParent::Exiting { reason } => {
                runner_log!(self.host.runner(), log::Level::Info, reason = reason.as_str(); "{} exiting", runner_name(&self.opts));
            }
            ToParent::JobAccepted { request_id, error } => self.on_job_accepted(request_id, error),
            ToParent::Ready { .. } => {
                self.fail_protocol(Arc::new(ProtocolError::Unexpected("unsolicited Ready".into())))
            }
        }
    }

    /// The runner's answer to the initialize: a refusal or anything but `Ready` fails it.
    fn on_init_reply(&mut self, started: Duration, now: Duration, msg: ToParent) {
        let result = match msg {
            ToParent::Ready { error: Some(error) } => Err(InitError::Failed(error)),
            ToParent::Ready { error: None } => {
                let elapsed = now.saturating_sub(started);
                runner_log!(self.host.runner(), log::Level::Info, elapsed_time = round(elapsed.as_secs_f64(), 2); "{} initialized", runner_name(&self.opts));
                Ok(elapsed)
            }
            _ => Err(InitError::Unexpected),
        };
        let refusal = match &result {
            Err(error) => Some(ProtocolError::Unexpected(error.to_string())),
            Ok(_) => None,
        };
        self.reply_init(result);
        if let Some(refusal) = refusal {
            self.fail_protocol(Arc::new(refusal));
        }
    }

    fn on_pong(&mut self, sent_ms: u64, now: Duration) {
        if let Lifecycle::Active {
            health: Health::Monitoring { ping, pong_deadline, .. }, ..
        } = &mut self.lifecycle
        {
            *ping = Ping::Ready;
            if pong_deadline.is_some() {
                *pong_deadline = Some(now + secs(self.opts.ping_timeout));
            }
        }
        let delay = unix_now_ms().saturating_sub(sent_ms as i64);
        if delay as f64 > self.opts.high_ping_threshold * MILLIS_PER_SECOND {
            runner_log!(self.host.runner(), log::Level::Warn, delay = delay; "{} is unresponsive", runner_name(&self.opts));
        }
    }

    /// The runner is ending on its own: it has the close timeout to exit.
    fn on_shutting_down(&mut self, now: Duration) {
        let deadline = now + secs(self.opts.close_timeout);
        match &mut self.lifecycle {
            Lifecycle::Active { .. } => self.begin_close(Close::Exit { deadline }, deadline),
            Lifecycle::Closing { phase, .. } => *phase = phase.shutting_down(deadline),
            Lifecycle::Exited { child, .. } => *child = Child::ShuttingDown,
        }
    }

    /// The runner took the pending job, or refused it, which ends the runner.
    fn on_job_accepted(&mut self, request_id: u64, error: Option<String>) {
        if !matches!(self.lifecycle.job(), Job::Pending { request_id: pending, .. } if pending == request_id)
        {
            self.fail_protocol(Arc::new(ProtocolError::Unexpected(format!(
                "job acknowledgment {request_id} has no pending submission"
            ))));
            return;
        }
        self.set_job(Job::Idle);
        let Some(error) = error else {
            self.set_job(Job::Running);
            self.reply_job(request_id, Ok(()));
            return;
        };
        self.reply_job(request_id, Err(StartJobError::Rejected(error)));
        self.end();
    }

    /// The first protocol failure remains authoritative even when the runner exits 0.
    fn fail_protocol(&mut self, failure: Arc<ProtocolError>) {
        if self.protocol_failure.is_some() {
            return;
        }
        self.protocol_failure = Some(failure.clone());
        runner_log!(self.host.runner(), log::Level::Error; "invalid IPC message: {failure}");
        if self.take_initialization().is_some() {
            let error = match *failure {
                ProtocolError::Channel(ChannelError::Version(theirs)) => InitError::Version(theirs),
                ProtocolError::Channel(ChannelError::Closed) => InitError::Closed,
                _ => InitError::Protocol(failure.clone()),
            };
            self.reply_init(Err(error));
        }
        if let Job::Pending { request_id, .. } = self.lifecycle.job() {
            self.set_job(Job::Idle);
            self.reply_job(request_id, Err(StartJobError::Protocol(failure)));
        }
        let deadline = self.now() + secs(self.opts.close_timeout);
        self.begin_close(Close::Exit { deadline }, deadline);
        match &mut self.lifecycle {
            Lifecycle::Closing { reader, phase, .. } => {
                *reader = Reader::Closed;
                *phase = Close::Exit { deadline };
            }
            Lifecycle::Exited { reader, .. } => *reader = Drain::Closed,
            _ => {}
        }
        // nothing more can be asked over the channel
        self.stop();
    }

    /// Asks the runner to shut down, with a deadline per phase.
    fn shutdown(&mut self) {
        if self.stopping() {
            return;
        }
        let now = self.now();
        if let Job::Pending { request_id, .. } = self.lifecycle.job() {
            self.set_job(Job::Idle);
            self.reply_job(request_id, Err(StartJobError::Closed));
        }
        self.begin_close(
            Close::Ack { deadline: now + secs(self.opts.close_timeout), child: Child::Running },
            now + secs(self.opts.close_timeout) * 3,
        );
        self.send(ToChild::Shutdown { reason: String::new() });
    }

    fn begin_close(&mut self, phase: Close, deadline: Duration) {
        if matches!(self.lifecycle, Lifecycle::Active { .. }) {
            self.lifecycle = Lifecycle::Closing {
                phase,
                cleanup: Cleanup::At(deadline),
                stop: Stop::NotRequested,
                reader: Reader::Open,
                work: self.lifecycle.work(),
            };
        }
    }

    fn take_initialization(&mut self) -> Option<Initialization> {
        match &mut self.lifecycle {
            Lifecycle::Active { phase: phase @ Active::Initializing(_), .. } => {
                let Active::Initializing(init) = *phase else { return None };
                *phase = Active::Ready(Job::Idle);
                Some(init)
            }
            Lifecycle::Closing { work: work @ Work::Initializing(_), .. }
            | Lifecycle::Exited { work: work @ Work::Initializing(_), .. } => {
                let Work::Initializing(init) = *work else { return None };
                *work = Work::Ready(Job::Idle);
                Some(init)
            }
            _ => None,
        }
    }

    fn set_job(&mut self, next: Job) {
        match &mut self.lifecycle {
            Lifecycle::Active { phase: Active::Ready(job), .. }
            | Lifecycle::Closing { work: Work::Ready(job), .. }
            | Lifecycle::Exited { work: Work::Ready(job), .. } => *job = next,
            _ => {}
        }
    }

    fn lifecycle_reader_child(&self) -> Child {
        match self.lifecycle {
            Lifecycle::Closing { phase, .. } => phase.child(),
            Lifecycle::Exited { child, .. } => child,
            _ => Child::Running,
        }
    }

    /// Ends the runner now: asks it to shut down and stops it (a process is killed; a thread
    /// finishes what it was asked).
    fn end(&mut self) {
        self.shutdown();
        self.stop();
    }

    /// Whether the runner's end was forced (or it already ended); false while it is waited for.
    fn stop(&mut self) -> bool {
        if self.exited() {
            return true;
        }
        let deadline = self.now() + secs(self.opts.close_timeout);
        self.begin_close(Close::Waiting(Child::Running), deadline);
        let forced = self.runner.stop();
        if let Lifecycle::Closing { stop: stop @ Stop::NotRequested, .. } = &mut self.lifecycle {
            *stop = Stop::Requested;
            let runner = runner_name(&self.opts);
            if forced {
                runner_log!(self.host.runner(), log::Level::Info; "stopping {runner}");
            } else {
                runner_log!(self.host.runner(), log::Level::Info; "waiting for {runner} to end");
            }
        }
        forced
    }

    fn send(&mut self, message: ToChild) {
        if !self.lifecycle.reader_open() || self.exited() {
            return;
        }
        let sent = self.channel.try_send(message);
        self.sent(sent);
    }

    /// A closed channel is reported by its reader; a full one means the child stopped reading.
    fn sent(&mut self, sent: Result<(), SendError>) {
        if let Err(error @ (SendError::Full | SendError::TooLarge(_))) = sent {
            self.fail_protocol(Arc::new(ProtocolError::Unexpected(error.to_string())));
        }
    }

    /// `None`: the runner has no footprint of its own, or it could not be read; sampling stops.
    fn on_sample(&mut self, sample: Option<f64>) {
        let now = self.now();
        let Some(memory_mb) = sample else { return };
        let baseline = *self.baseline_mb.get_or_insert(memory_mb);
        let (warn_mb, limit_mb) = (self.opts.memory_warn_mb, self.opts.memory_limit_mb);
        let over_limit = limit_mb > 0.0 && memory_mb > limit_mb;
        let warning = if over_limit {
            Some((
                log::Level::Error,
                format!("{} exceeded memory limit, stopping it", runner_name(&self.opts)),
            ))
        } else if warn_mb > 0.0 && memory_mb > warn_mb && self.gate.should_warn(memory_mb, now) {
            // without a hard limit nothing is stopped: say so
            let advisory = match limit_mb <= 0.0 {
                true => {
                    format!(" (advisory only, the {} will not be stopped)", runner_name(&self.opts))
                }
                false => String::new(),
            };
            let message = format!(
                "{} memory usage is above the warning threshold{advisory}",
                runner_name(&self.opts)
            );
            Some((log::Level::Warn, message))
        } else {
            None
        };
        if let Some((level, message)) = warning {
            runner_log!(
                self.host.runner(),
                level,
                memory_usage_mb = round(memory_mb, 1),
                memory_warn_mb = warn_mb,
                memory_limit_mb = limit_mb,
                uptime = round(now.as_secs_f64(), 1),
                has_running_job = matches!(self.lifecycle.job(), Job::Running),
                baseline_memory_mb = round(baseline, 1),
                growth_memory_mb = round(memory_mb - baseline, 1);
                "{message}"
            );
        }
        if over_limit {
            self.end();
        }
        if let Lifecycle::Active { health: Health::Monitoring { next_sample, .. }, .. } =
            &mut self.lifecycle
        {
            *next_sample = Some(now + MEMORY_MONITOR_INTERVAL);
        }
    }
}

#[cfg(test)]
#[path = "supervisor_tests.rs"]
mod tests;

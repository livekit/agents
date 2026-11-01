//! One job runner's supervisor: an actor that launches it through its executor, then owns the
//! channel, the initialization, pings, memory samples, the shutdown and stop deadlines and the
//! runner's cleanup. It never asks whether the runner is a process or a thread.

use lk_agents_net::clock::unix_now_ms;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::executor::{Connection, Exit, Runner, RunnerExecutor};
use super::ipc_channel::{ChannelError, SendError, VERSION};
use super::wire::{Hello, StartJob, ToChild, ToParent};

/// A job's dispatch timeline, as the host's root span records it.
pub struct JobTimeline;

impl lk_agents_abi::worker::job::JobTrace for JobTimeline {
    fn job_timeline(
        received_at: f64,
        accepted_at: f64,
        assigned_at: f64,
        launched_at: f64,
        entrypoint_started_at: f64,
        server_started_at: i64,
    ) -> (Vec<(String, i64)>, Vec<(String, f64)>) {
        let ns = |ts: f64| (ts * 1e9) as i64;
        let stages = [
            ("job_received", received_at),
            ("job_accepted", accepted_at),
            ("job_assigned", assigned_at),
            ("process_assigned", launched_at),
            ("entrypoint_started", entrypoint_started_at),
        ];
        let mut events: Vec<_> = stages
            .iter()
            .filter(|(_, ts)| *ts != 0.0)
            .map(|(name, ts)| ((*name).to_owned(), ns(*ts)))
            .collect();
        let gaps = [
            ("lk.job.accept_latency", received_at, accepted_at),
            ("lk.job.assignment_latency", accepted_at, assigned_at),
            ("lk.job.launch_latency", assigned_at, launched_at),
            ("lk.job.entrypoint_latency", launched_at, entrypoint_started_at),
            ("lk.job.dispatch_latency", received_at, entrypoint_started_at),
        ];
        let attributes = gaps
            .iter()
            .filter(|(_, start, end)| *start != 0.0 && *end != 0.0)
            .map(|(attr, start, end)| ((*attr).to_owned(), (end - start).max(0.0)))
            .collect();
        if server_started_at > 0 {
            let v = server_started_at as f64;
            let started = if v > 1e17 {
                v / 1e9
            } else if v > 1e11 {
                v / 1e3
            } else {
                v
            };
            events.push(("job_started_on_server".to_owned(), ns(started)));
        }
        (events, attributes)
    }
}

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExtraValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// A log record's extra fields; the host adds the process's own after them.
pub type Extra = Vec<(&'static str, ExtraValue)>;

/// The fields as the host's log record takes them.
pub fn extra_fields(extra: Extra) -> serde_json::Map<String, serde_json::Value> {
    extra
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                ExtraValue::Int(v) => v.into(),
                ExtraValue::Float(v) => v.into(),
                ExtraValue::Str(v) => v.into(),
                ExtraValue::Bool(v) => v.into(),
            };
            (key.to_owned(), value)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct SupervisorOptions {
    /// What the logs call the runner: "process" or "job thread".
    pub runner: &'static str,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub ping_interval: f64,
    pub ping_timeout: f64,
    pub high_ping_threshold: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("initialization already requested")]
    AlreadyInitializing,
    #[error("duplex closed")]
    Closed,
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
                Self::Io { kind: "UnexpectedEof".into(), os_error: None, message }
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
    use prost::Message;
    livekit_protocol::Job::decode(job.job.as_slice())
        .map_err(|error| StartJobError::InvalidFields(format!("job protobuf: {error}")))?;
    for (field, value) in [
        ("received_at", job.received_at),
        ("accepted_at", job.accepted_at),
        ("assigned_at", job.assigned_at),
        ("launched_at", job.launched_at),
    ] {
        if !value.is_finite() || value < 0.0 {
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

/// Where a graceful close is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Close {
    None,
    Ack(Duration),
    ShuttingDown(Duration),
    Exit(Duration),
    Waiting,
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

/// Where the supervisor's diagnostics go: the executor's logger in the host language.
pub trait SupervisorCallbacks: Send + Sync + 'static {
    fn log(&self, level: Level, msg: &str, extra: Extra);
    /// The runner ended and its supervision is over: the host's handle can go.
    fn release(&self) {}
}

const QUEUE: usize = 16;

enum Command {
    Supervise,
    Initialize(Hello, oneshot::Sender<Result<Option<Duration>, InitError>>),
    StartJob(StartJob, oneshot::Sender<Result<(), StartJobError>>),
}

#[derive(Clone, Copy, Default)]
struct Control {
    shutdown: bool,
    kill: bool,
}

#[derive(Clone, Default)]
struct Status {
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
        opts: SupervisorOptions,
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
                    let driver = Driver::new(opts, host, blocking_io, runner, channel, status_tx);
                    driver.run(commands_rx, control_rx).await;
                }
                Err(cause) => refuse(cause, host, commands_rx, status_tx).await,
            }
        };
        driving.spawn(&spawner, "job runner supervisor", run);
        Self { commands, control, status, _driver: driving }
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

    pub async fn initialize(&self, hello: Hello) -> Result<Option<Duration>, InitError> {
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
            .send(Command::StartJob(job, reply))
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
    cause: String,
    host: Arc<dyn SupervisorCallbacks>,
    mut commands: mpsc::Receiver<Command>,
    status: watch::Sender<Status>,
) {
    host.log(Level::Error, &format!("launch failed before starting a runner: {cause}"), vec![]);
    commands.close();
    while let Some(command) = commands.recv().await {
        match command {
            Command::Initialize(_, reply) => {
                let _ = reply.send(Err(InitError::Failed(cause.clone())));
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
    opts: SupervisorOptions,
    origin: Instant,
    host: Arc<dyn SupervisorCallbacks>,
    blocking_io: lk_agents_net::blocking_io::BlockingIo,
    runner: Arc<dyn Runner>,
    channel: Arc<Connection>,
    init_reply: Option<oneshot::Sender<Result<Option<Duration>, InitError>>>,
    job_reply: Option<(u64, oneshot::Sender<Result<(), StartJobError>>)>,
    next_job_id: u64,
    status: watch::Sender<Status>,
    sampling: Sampling,
    /// Waiting for `Ready`: since when, and until when.
    init: Option<(Duration, Duration)>,
    initialized_requested: bool,
    initialized: bool,
    pending_job: Option<(u64, Duration)>,
    startup_at: Option<Duration>,
    supervising: bool,
    next_ping: Option<Duration>,
    /// A ping is unanswered: the next waits for its pong, so a stalled child never fills the
    /// channel with pings.
    ping_unanswered: bool,
    pong_deadline: Option<Duration>,
    next_sample: Option<Duration>,
    baseline_mb: Option<f64>,
    gate: WarnGate,
    close: Close,
    acked: bool,
    shutting_down: bool,
    closing: bool,
    /// `stop()` was called on the runner.
    stopped: bool,
    exit: Option<Exit>,
    running_job: bool,
    reader_closed: bool,
    reader_deadline: Option<Duration>,
    cleanup_at: Option<Duration>,
    protocol_failure: Option<Arc<ProtocolError>>,
}

impl Driver {
    fn new(
        opts: SupervisorOptions,
        host: Arc<dyn SupervisorCallbacks>,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
        runner: Box<dyn Runner>,
        channel: Connection,
        status: watch::Sender<Status>,
    ) -> Self {
        let origin = Instant::now();
        Self {
            startup_at: Some(secs(opts.initialize_timeout)),
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
            init: None,
            initialized_requested: false,
            initialized: false,
            pending_job: None,
            supervising: false,
            next_ping: None,
            ping_unanswered: false,
            pong_deadline: None,
            next_sample: None,
            baseline_mb: None,
            gate: WarnGate::default(),
            close: Close::None,
            acked: false,
            shutting_down: false,
            closing: false,
            stopped: false,
            exit: None,
            running_job: false,
            reader_closed: false,
            reader_deadline: None,
            cleanup_at: None,
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
                () = async { tokio::time::sleep_until(deadline.unwrap()).await }, if deadline.is_some() => continue,
                exit = runner.finished(), if self.exit.is_none() => self.input(true, |d| d.on_finished(exit)),
                changed = control.changed(), if control_open => {
                    if changed.is_err() {
                        control_open = false;
                        self.input(false, Self::end);
                    } else {
                        let next = *control.borrow_and_update();
                        if next.kill && !seen.kill {
                            self.input(false, Self::end);
                        }
                        if next.shutdown && !seen.shutdown {
                            self.input(false, Self::shutdown);
                        }
                        seen = next;
                    }
                },
                msg = channel.recv(), if reader_open => match msg {
                    Ok(msg) => self.input(true, |d| d.on_message(msg)),
                    Err(reason) => {
                        reader_open = false;
                        self.input(true, |d| d.on_reader_closed(reason));
                    }
                },
                command = commands.recv(), if commands_open => match command {
                    Some(Command::Supervise) => self.input(false, Self::supervise),
                    Some(Command::Initialize(hello, reply)) => {
                        if self.stopping() {
                            let _ = reply.send(Err(InitError::Closed));
                        } else if self.init_reply.is_some() {
                            let _ = reply.send(Err(InitError::AlreadyInitializing));
                        } else {
                            self.init_reply = Some(reply);
                            self.input(false, |d| d.initialize(hello));
                        }
                    }
                    Some(Command::StartJob(job, reply)) => {
                        if self.job_reply.is_some() {
                            let _ = reply.send(Err(StartJobError::AlreadyRunning));
                        } else {
                            self.next_job_id += 1;
                            let request_id = self.next_job_id;
                            self.job_reply = Some((request_id, reply));
                            self.input(false, |d| d.start_job(request_id, job));
                        }
                    }
                    None => {
                        commands_open = false;
                        self.input(false, Self::end);
                    }
                },
                sample = async { self.sampling.as_mut().unwrap().await }, if self.sampling.is_some() => {
                    self.sampling = None;
                    self.input(false, |d| d.on_sample(sample.ok().flatten()));
                }
            }
            self.publish();
        }
        // Started blocking samples cannot be aborted: retain the host until they settle.
        if let Some(mut sample) = self.sampling.take()
            && tokio::time::timeout(secs(self.opts.close_timeout), &mut sample).await.is_err()
        {
            self.log(Level::Error, "memory sample did not settle; retaining the runner's host");
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

    fn now(&self) -> Duration {
        Instant::now().duration_since(self.origin)
    }

    fn exited(&self) -> bool {
        self.exit.is_some()
    }

    /// One input: what is due first (a reply at a deadline cannot revive an expired phase), then
    /// the input, which after the exit only `after_exit` inputs still reach.
    fn input(&mut self, after_exit: bool, f: impl FnOnce(&mut Self)) {
        if self.cleaned() {
            return;
        }
        self.tick();
        if self.exited() && !after_exit {
            return;
        }
        f(self);
    }

    fn publish(&self) {
        let failure = self.protocol_failure.as_deref().map(Into::into);
        let exit = self.exit;
        self.status.send_modify(|s| {
            s.failure = failure;
            s.exit = exit;
        });
    }

    /// Closing or ended: the memory monitor's failures are no longer worth a log.
    fn stopping(&self) -> bool {
        self.closing || self.stopped || self.exited()
    }

    fn cleaned(&self) -> bool {
        self.exited() && (!self.initialized_requested || self.reader_closed)
    }

    fn wake_at(&self) -> Option<Duration> {
        if self.cleaned() {
            return None;
        }
        if self.exited() {
            return self.reader_deadline;
        }
        let close = match self.close {
            Close::Ack(at) | Close::ShuttingDown(at) | Close::Exit(at) => Some(at),
            _ => None,
        };
        let sample = self.next_sample.filter(|_| self.sampling.is_none() && !self.stopping());
        [
            self.startup_at,
            self.init.map(|(_, at)| at),
            self.pending_job.map(|(_, at)| at),
            self.next_ping,
            self.pong_deadline,
            sample,
            close,
            self.cleanup_at,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    fn log(&self, level: Level, msg: &str) {
        self.host.log(level, msg, Vec::new());
    }

    fn reply_init(&mut self, result: Result<Option<Duration>, InitError>) {
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
        if self.initialized_requested || self.closing {
            self.reply_init(Err(InitError::AlreadyInitializing));
            return;
        }
        self.initialized_requested = true;
        self.log(Level::Info, &format!("initializing {}", self.opts.runner));
        self.send(ToChild::Hello(hello));
        if self.reader_closed {
            self.reply_init(Err(InitError::Closed));
        } else {
            let now = self.now();
            let until = self.startup_at.take().unwrap_or(now + secs(self.opts.initialize_timeout));
            self.init = Some((now, until));
        }
    }

    fn supervise(&mut self) {
        if self.supervising {
            return;
        }
        let now = self.now();
        self.supervising = true;
        self.next_ping = Some(now);
        // `ping_timeout` 0: a late pong is only reported
        if self.opts.ping_timeout > 0.0 {
            self.pong_deadline = Some(now + secs(self.opts.ping_timeout));
        }
        if self.opts.memory_limit_mb > 0.0 || self.opts.memory_warn_mb > 0.0 {
            self.next_sample = Some(now);
        }
    }

    fn start_job(&mut self, request_id: u64, job: StartJob) {
        let failure = if let Some(failure) = &self.protocol_failure {
            Some(StartJobError::Protocol(failure.clone()))
        } else if self.stopping() {
            Some(StartJobError::Closed)
        } else if !self.initialized {
            Some(StartJobError::NotReady)
        } else if self.pending_job.is_some() || self.running_job {
            Some(StartJobError::AlreadyRunning)
        } else {
            None
        };
        match failure {
            Some(error) => self.reply_job(request_id, Err(error)),
            None => {
                let message = ToChild::StartJob { request_id, job: Box::new(job) };
                match self.channel.try_send(message) {
                    Err(SendError::TooLarge(size)) => self.reply_job(
                        request_id,
                        Err(StartJobError::InvalidFields(format!(
                            "a {size}-byte job is too large"
                        ))),
                    ),
                    sent => {
                        self.pending_job =
                            Some((request_id, self.now() + secs(self.opts.initialize_timeout)));
                        self.sent(sent);
                    }
                }
            }
        }
    }

    /// The peer finishing after it said it was shutting down is the clean end; anything else is
    /// a protocol failure.
    fn on_reader_closed(&mut self, reason: ChannelError) {
        if self.shutting_down && matches!(reason, ChannelError::Closed) {
            self.reader_closed = true;
            self.reader_deadline = None;
            self.next_ping = None;
            self.acked = true;
            self.advance_close();
        } else {
            self.fail_protocol(Arc::new(reason.into()));
        }
    }

    fn on_finished(&mut self, exit: Exit) {
        let now = self.now();
        self.exit = Some(exit);
        if self.initialized_requested && !self.reader_closed {
            self.reader_deadline = Some(now + secs(self.opts.close_timeout));
        }
        if self.reader_closed && self.init.take().is_some() {
            self.reply_init(Err(InitError::Closed));
        }
        (self.startup_at, self.cleanup_at) = (None, None);
        (self.next_ping, self.pong_deadline, self.next_sample) = (None, None, None);
        self.close = Close::None;
        if exit.code != 0 && !exit.killed {
            let message =
                format!("{} exited with non-zero exit code {}", self.opts.runner, exit.code);
            self.log(Level::Error, &message);
        }
    }

    /// What is due now.
    fn tick(&mut self) {
        if self.cleaned() {
            return;
        }
        let now = self.now();
        if self.reader_deadline.is_some_and(|at| at <= now) {
            self.reader_deadline = None;
            self.fail_protocol(Arc::new(ProtocolError::Channel(ChannelError::Io {
                kind: std::io::ErrorKind::TimedOut,
                os_error: None,
                message: "IPC reader did not settle after the runner ended".into(),
            })));
        }
        if self.exited() {
            return;
        }
        if self.cleanup_at.is_some_and(|at| at <= now) {
            self.cleanup_at = None;
            let message = format!("incomplete {} cleanup; retaining ownership", self.opts.runner);
            self.log(Level::Error, &message);
            self.status.send_modify(|s| s.incomplete = true);
        }
        if self.startup_at.is_some_and(|at| at <= now) {
            self.startup_at = None;
            let message = format!(
                "{} launch did not enter initialization before its deadline",
                self.opts.runner
            );
            self.log(Level::Error, &message);
            self.end();
        }
        if let Some((_, at)) = self.init
            && at <= now
        {
            self.init = None;
            self.reply_init(Ok(None));
            self.end();
        }
        if let Some((request_id, deadline)) = self.pending_job
            && deadline <= now
        {
            self.pending_job = None;
            self.reply_job(request_id, Err(StartJobError::Timeout));
            self.end();
        }
        if let Some(at) = self.next_ping
            && at <= now
        {
            if !self.ping_unanswered {
                self.ping_unanswered = true;
                self.send(ToChild::Ping { sent_ms: unix_now_ms().max(0) as u64 });
            }
            self.next_ping =
                Some(now + secs(self.opts.ping_interval).max(Duration::from_millis(1)));
        }
        if let Some(at) = self.pong_deadline
            && at <= now
        {
            self.pong_deadline = None;
            self.log(Level::Error, &format!("{} is unresponsive, stopping it", self.opts.runner));
            self.end();
        }
        if let Some(at) = self.next_sample
            && at <= now
            && self.sampling.is_none()
            && !self.stopping()
        {
            self.next_sample = None;
            let runner = self.runner.clone();
            // Admission before any task: a saturated sampler retries at its next cadence.
            match self.blocking_io.try_spawn((), move || runner.memory_mb()) {
                Ok(receipt) => self.sampling = Some(receipt),
                Err(_) => self.next_sample = Some(now + MEMORY_MONITOR_INTERVAL),
            }
        }
        match self.close {
            Close::Ack(at) if at <= now => {
                let action = if self.stop() { "stopping it" } else { "waiting for it" };
                let message =
                    format!("{} did not ack shutdown in time, {action}", self.opts.runner);
                self.log(Level::Error, &message);
                self.close = Close::ShuttingDown(now + secs(self.opts.close_timeout));
                self.advance_close();
            }
            Close::ShuttingDown(at) | Close::Exit(at) if at <= now => {
                let action = if self.stop() { "stopping it" } else { "waiting for it" };
                let message = format!("{} did not exit in time, {action}", self.opts.runner);
                self.log(Level::Error, &message);
                self.close = Close::Waiting;
            }
            _ => {}
        }
    }

    fn on_message(&mut self, msg: ToParent) {
        if self.exited() {
            match &msg {
                ToParent::Ready { error: Some(_) } => {}
                ToParent::JobAccepted { error: Some(_), .. }
                | ToParent::ShuttingDown
                | ToParent::Exiting { .. } => {}
                _ => return,
            }
        }
        let now = self.now();
        if let Some((started, _)) = self.init.take() {
            let result = match msg {
                ToParent::Ready { error: Some(error) } => Err(InitError::Failed(error)),
                ToParent::Ready { error: None } => {
                    self.initialized = true;
                    let elapsed = now.saturating_sub(started);
                    let extra =
                        vec![("elapsed_time", ExtraValue::Float(round(elapsed.as_secs_f64(), 2)))];
                    let message = format!("{} initialized", self.opts.runner);
                    self.host.log(Level::Info, &message, extra);
                    Ok(Some(elapsed))
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
            return;
        }
        match msg {
            ToParent::Pong { sent_ms } => {
                self.ping_unanswered = false;
                let delay = unix_now_ms().saturating_sub(sent_ms as i64);
                if delay as f64 > self.opts.high_ping_threshold * 1000.0 {
                    let message = format!("{} is unresponsive", self.opts.runner);
                    self.host.log(
                        Level::Warning,
                        &message,
                        vec![("delay", ExtraValue::Int(delay))],
                    );
                }
                // a late pong does not revive a deadline that fired
                if self.pong_deadline.is_some() {
                    self.pong_deadline = Some(now + secs(self.opts.ping_timeout));
                }
            }
            ToParent::ShutdownAck => {
                self.acked = true;
                self.advance_close();
            }
            ToParent::ShuttingDown => {
                self.shutting_down = true;
                self.next_ping = None;
                self.pong_deadline = None;
                if self.close == Close::None {
                    self.closing = true;
                    self.close = Close::Exit(now + secs(self.opts.close_timeout));
                    self.cleanup_at.get_or_insert(now + secs(self.opts.close_timeout));
                }
                self.advance_close();
            }
            ToParent::Exiting { reason } => {
                let extra = vec![("reason", ExtraValue::Str(reason))];
                self.host.log(Level::Info, &format!("{} exiting", self.opts.runner), extra);
            }
            ToParent::JobAccepted { request_id, error } => {
                if self.pending_job.is_some_and(|(pending, _)| pending == request_id) {
                    self.pending_job = None;
                    let result = match error {
                        Some(error) => Err(StartJobError::Rejected(error)),
                        None => {
                            self.running_job = true;
                            Ok(())
                        }
                    };
                    let refused = result.is_err();
                    self.reply_job(request_id, result);
                    if refused {
                        self.end();
                    }
                } else {
                    self.fail_protocol(Arc::new(ProtocolError::Unexpected(format!(
                        "job acknowledgment {request_id} has no pending submission"
                    ))));
                }
            }
            ToParent::Ready { .. } => {
                self.fail_protocol(Arc::new(ProtocolError::Unexpected("unsolicited Ready".into())))
            }
        }
    }

    /// The first protocol failure remains authoritative even when the runner exits 0.
    fn fail_protocol(&mut self, failure: Arc<ProtocolError>) {
        if self.protocol_failure.is_some() {
            return;
        }
        self.protocol_failure = Some(failure.clone());
        self.reader_closed = true;
        self.reader_deadline = None;
        self.initialized = false;
        self.next_ping = None;
        self.pong_deadline = None;
        self.closing = true;
        self.log(Level::Error, &format!("invalid IPC message: {failure}"));
        if self.init.take().is_some() {
            let error = match *failure {
                ProtocolError::Channel(ChannelError::Version(theirs)) => InitError::Version(theirs),
                ProtocolError::Channel(ChannelError::Closed) => InitError::Closed,
                _ => InitError::Protocol(failure.clone()),
            };
            self.reply_init(Err(error));
        }
        if let Some((request_id, _)) = self.pending_job.take() {
            self.reply_job(request_id, Err(StartJobError::Protocol(failure)));
        }
        self.acked = true;
        self.shutting_down = true;
        self.advance_close();
        // nothing more can be asked over the channel
        self.stop();
    }

    /// Asks the runner to shut down, with a deadline per phase.
    fn shutdown(&mut self) {
        if self.close != Close::None {
            return;
        }
        let now = self.now();
        self.closing = true;
        if let Some((request_id, _)) = self.pending_job.take() {
            self.reply_job(request_id, Err(StartJobError::Closed));
        }
        self.send(ToChild::Shutdown { reason: String::new() });
        self.cleanup_at = Some(now + secs(self.opts.close_timeout) * 3);
        self.close = Close::Ack(now + secs(self.opts.close_timeout));
        self.advance_close();
    }

    fn advance_close(&mut self) {
        let now = self.now();
        if matches!(self.close, Close::Ack(_)) && self.acked {
            self.close = Close::ShuttingDown(now + secs(self.opts.close_timeout));
        }
        if matches!(self.close, Close::ShuttingDown(_)) && self.shutting_down {
            self.close = Close::Exit(now + secs(self.opts.close_timeout));
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
        self.startup_at = None;
        self.closing = true;
        let forced = self.runner.stop();
        if !std::mem::replace(&mut self.stopped, true) {
            let runner = &self.opts.runner;
            let message = if forced {
                format!("stopping {runner}")
            } else {
                format!("waiting for {runner} to end")
            };
            self.log(Level::Info, &message);
        }
        self.cleanup_at.get_or_insert(self.now() + secs(self.opts.close_timeout));
        forced
    }

    fn send(&mut self, message: ToChild) {
        if self.reader_closed || self.exited() {
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
        if limit_mb > 0.0 && memory_mb > limit_mb {
            let extra = self.memory_extra(now, memory_mb, baseline);
            let message = format!("{} exceeded memory limit, stopping it", self.opts.runner);
            self.host.log(Level::Error, &message, extra);
            self.end();
        } else if warn_mb > 0.0 && memory_mb > warn_mb && self.gate.should_warn(memory_mb, now) {
            // without a hard limit nothing is stopped: say so
            let advisory = match limit_mb <= 0.0 {
                true => format!(" (advisory only, the {} will not be stopped)", self.opts.runner),
                false => String::new(),
            };
            let msg = format!(
                "{} memory usage is above the warning threshold{advisory}",
                self.opts.runner
            );
            let extra = self.memory_extra(now, memory_mb, baseline);
            self.host.log(Level::Warning, &msg, extra);
        }
        if !self.stopping() {
            self.next_sample = Some(now + MEMORY_MONITOR_INTERVAL);
        }
    }

    fn memory_extra(&self, now: Duration, memory_mb: f64, baseline_mb: f64) -> Extra {
        vec![
            ("memory_usage_mb", ExtraValue::Float(round(memory_mb, 1))),
            ("memory_warn_mb", ExtraValue::Float(self.opts.memory_warn_mb)),
            ("memory_limit_mb", ExtraValue::Float(self.opts.memory_limit_mb)),
            ("uptime", ExtraValue::Float(round(now.as_secs_f64(), 1))),
            ("has_running_job", ExtraValue::Bool(self.running_job)),
            ("baseline_memory_mb", ExtraValue::Float(round(baseline_mb, 1))),
            ("growth_memory_mb", ExtraValue::Float(round(memory_mb - baseline_mb, 1))),
        ]
    }
}

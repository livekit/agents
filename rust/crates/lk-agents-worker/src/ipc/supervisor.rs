//! One job process's supervisor: an actor owning the IPC connection, the initialization, pings,
//! memory samples, shutdown and kill deadlines and the cleanup ownership of the process.

use lk_agents_abi::unix_now_ms;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use lk_agents_abi::errors::HostError;

use super::channel::{Channel, ChannelError, SendError};
use super::wire::{Hello, StartJob, ToChild, ToParent};

mod timeline;
mod types;
pub use timeline::JobTimeline;
pub use types::*;
use types::{Close, round};

/// What the supervisor needs from the host language: logging and the process handle.
pub trait ProcessCallbacks: Send + Sync + 'static {
    fn log(&self, level: Level, msg: &str, extra: Extra);
    /// An error log with the failure's traceback.
    fn log_exception(&self, msg: &str, err: HostError);
    fn is_alive(&self) -> bool;
    /// SIGKILL (`Process.kill`; a terminate on Windows).
    fn kill(&self) -> Result<(), String>;
    /// The process's footprint in MB and the metric it reflects (pss, uss or rss). Blocking: the
    /// supervisor calls it off the runtime.
    fn sample_memory(&self) -> Result<(f64, String), MemoryError>;
    /// The process exited and was reaped: the host's handle can go.
    fn release(&self) {}
    fn incomplete_cleanup(&self, _: CleanupObservation) {}
    fn signal_failed(&self, _: SignalFailure) {}
}

/// Why a runtime memory sampler failed. Host errors never enter supervision state.
pub enum MemoryError {
    /// The process is gone or inaccessible; stop sampling after a diagnostic.
    Unavailable(String),
    /// A transient failure, retaining the host error for its diagnostic.
    Failed(HostError),
    /// A process-control interruption; stop sampling without a diagnostic.
    Interrupted,
}

const QUEUE: usize = 16;

enum Command {
    Supervise,
    Initialize(Hello, oneshot::Sender<Result<Option<Duration>, InitError>>),
    StartJob(StartJob, oneshot::Sender<Result<(), StartJobError>>),
}

#[derive(Clone, Default)]
struct Control {
    shutdown: bool,
    kill: bool,
    exited: Option<Option<i64>>,
    reaped: Option<i64>,
    pid: Option<u32>,
    launch_failed: Option<String>,
    launch_rejected: Option<String>,
}

#[derive(Clone, Default)]
struct Status {
    failure: Option<lk_agents_abi::worker::ipc::IpcFailure>,
    killed: bool,
    kill_settled: bool,
    exited: bool,
    reaped: bool,
    incomplete: bool,
    cleaned: bool,
}

/// Owns one supervisor actor. Control is coalesced independently of bounded work submission.
#[derive(Clone)]
pub struct ProcSupervisor {
    commands: mpsc::Sender<Command>,
    control: watch::Sender<Control>,
    status: watch::Receiver<Status>,
    /// The actor's owner; dropping the supervisor does not cancel it.
    _driver: Arc<TaskSet>,
}

impl ProcSupervisor {
    /// Supervises the process at the other end of `channel`. Needs a runtime context.
    pub fn new<H: ProcessCallbacks>(
        channel: Channel<ToChild, ToParent>,
        host: H,
        opts: SupervisorOptions,
        blocking_io: lk_agents_net::blocking_io::BlockingIo,
    ) -> Self {
        let channel = Arc::new(channel);
        let spawner = TokioSpawner::current();
        let (commands, commands_rx) = mpsc::channel(QUEUE);
        let (control, control_rx) = watch::channel(Control::default());
        let (status_tx, status) = watch::channel(Status::default());
        let driver = Driver {
            opts,
            origin: Instant::now(),
            host: Arc::new(host),
            blocking_io,
            channel: channel.clone(),
            init_reply: None,
            job_reply: None,
            next_job_id: 0,
            status: status_tx,
            sampling: None,
            init: None,
            initialized_requested: false,
            initialized: false,
            pending_job: None,
            startup_at: None,
            supervising: false,
            next_ping: None,
            pong_deadline: None,
            next_sample: None,
            baseline_mb: None,
            gate: WarnGate::default(),
            close: Close::None,
            acked: false,
            shutting_down: false,
            closing: false,
            killed: false,
            kill_ended: false,
            exited: false,
            running_job: false,
            reader_closed: false,
            reader_deadline: None,
            reaped: false,
            launch_rejected: false,
            exitcode: None,
            cleanup_at: None,
            protocol_failure: None,
        };
        // Self-owned: it ends once the supervisor's channels close, after shutting the child down.
        let driving = Arc::new(TaskSet::default());
        let run = driver.run(commands_rx, control_rx, channel);
        driving.spawn(&spawner, "process supervisor", run);
        Self { commands, control, status, _driver: driving }
    }

    pub fn killed(&self) -> bool {
        self.status.borrow().killed
    }

    pub fn completion(
        &self,
        exit_code: Option<i32>,
    ) -> lk_agents_abi::worker::ipc::ProcessCompletion {
        lk_agents_abi::worker::ipc::ProcessCompletion {
            exit_code,
            failure: self.status.borrow().failure.clone(),
        }
    }

    async fn until(&self, done: impl Fn(&Status) -> bool) {
        let mut status = self.status.clone();
        let _ = status.wait_for(|s| s.cleaned || s.incomplete || done(s)).await;
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

    pub async fn launched(&self, pid: u32) {
        self.control.send_modify(|c| c.pid = Some(pid));
    }

    pub async fn not_launched(&self, cause: String) {
        self.control.send_modify(|control| control.launch_rejected = Some(cause));
        self.until(|status| status.cleaned).await;
    }

    pub async fn launch_failed(&self, cause: String) {
        self.control.send_modify(|control| control.launch_failed = Some(cause));
    }

    pub async fn shutdown(&self) {
        self.control.send_modify(|c| c.shutdown = true);
        self.until(|_| false).await;
    }

    pub async fn kill(&self) {
        self.control.send_modify(|c| c.kill = true);
        self.until(|s| s.kill_settled).await;
    }

    /// OS exit is separate from the start method's reaping receipt.
    pub async fn exited(&self, exitcode: Option<i64>) {
        self.control.send_modify(|c| c.exited = Some(exitcode));
        self.until(|s| s.exited).await;
    }

    /// Observes OS exit without waiting for the separate reaping receipt.
    pub async fn wait_exit(&self) -> bool {
        let mut status = self.status.clone();
        let _ = status.wait_for(|state| state.exited || state.cleaned).await;
        let exited = status.borrow().exited;
        exited
    }

    /// Waits for the owner's real cleanup receipt, including a no-process launch rejection.
    pub async fn wait_cleaned(&self) -> bool {
        let mut status = self.status.clone();
        let _ = status.wait_for(|state| state.cleaned).await;
        let reaped = status.borrow().reaped;
        reaped
    }

    pub async fn wait_cleanup(&self) {
        self.until(|_| false).await;
    }

    pub async fn reaped(&self, exitcode: i64) {
        self.control.send_modify(|c| c.reaped = Some(exitcode));
        let mut status = self.status.clone();
        let _ = status.wait_for(|s| s.reaped).await;
    }
}

/// What the host's memory sampler answered.
enum Sample {
    Mb(f64, String),
    Unavailable(String),
    /// Logged when it arrived, with the host's error.
    Failed,
    Interrupted,
}

type Sampling =
    Option<lk_agents_net::blocking_io::BlockingIoTask<Result<(f64, String), MemoryError>>>;

/// The supervisor's actor: the process's state and every deadline, changed only here.
struct Driver {
    opts: SupervisorOptions,
    origin: Instant,
    host: Arc<dyn ProcessCallbacks>,
    blocking_io: lk_agents_net::blocking_io::BlockingIo,
    channel: Arc<Channel<ToChild, ToParent>>,
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
    pong_deadline: Option<Duration>,
    next_sample: Option<Duration>,
    baseline_mb: Option<f64>,
    gate: WarnGate,
    close: Close,
    acked: bool,
    shutting_down: bool,
    closing: bool,
    killed: bool,
    kill_ended: bool,
    exited: bool,
    running_job: bool,
    reader_closed: bool,
    reader_deadline: Option<Duration>,
    reaped: bool,
    launch_rejected: bool,
    exitcode: Option<i64>,
    cleanup_at: Option<Duration>,
    protocol_failure: Option<Arc<ProtocolError>>,
}

impl Driver {
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        mut control: watch::Receiver<Control>,
        channel: Arc<Channel<ToChild, ToParent>>,
    ) {
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
                changed = control.changed(), if control_open => {
                    if changed.is_err() {
                        control_open = false;
                        self.input(false, Self::kill);
                    } else {
                        let next = control.borrow_and_update().clone();
                        if next.pid != seen.pid && let Some(pid) = next.pid {
                            self.input(false, |d| d.launched(pid));
                        }
                        if next.exited != seen.exited && let Some(code) = next.exited {
                            self.input(false, |d| d.on_exited(code));
                        }
                        if next.reaped != seen.reaped && let Some(code) = next.reaped {
                            self.input(true, |d| d.on_reaped(code));
                        }
                        if next.launch_failed != seen.launch_failed && let Some(cause) = &next.launch_failed {
                            self.input(true, |d| d.launch_failed(cause.clone()));
                        }
                        if next.launch_rejected != seen.launch_rejected && let Some(cause) = &next.launch_rejected {
                            self.input(false, |d| d.launch_rejected(cause.clone()));
                        }
                        if next.kill && !seen.kill {
                            self.input(false, Self::kill);
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
                        if self.stopping() || self.exited {
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
                        self.input(false, Self::kill);
                    }
                },
                sample = async { self.sampling.as_mut().unwrap().await }, if self.sampling.is_some() => {
                    self.sampling = None;
                    let sample = match sample {
                        Ok(Ok((mb, metric))) => Sample::Mb(mb, metric),
                        Ok(Err(MemoryError::Unavailable(e))) => Sample::Unavailable(e),
                        Ok(Err(MemoryError::Failed(e))) => {
                            if !self.stopping() { self.host.log_exception("Error in memory monitoring task", e); }
                            Sample::Failed
                        }
                        Ok(Err(MemoryError::Interrupted)) | Err(_) => Sample::Interrupted,
                    };
                    self.input(false, |d| d.on_sample(sample));
                }
            }
            self.publish();
        }
        // Started blocking samples cannot be aborted: retain the host until they settle.
        if let Some(mut sample) = self.sampling.take() {
            if tokio::time::timeout(secs(self.opts.close_timeout), &mut sample).await.is_err() {
                let observation =
                    CleanupObservation { pid: self.opts.pid, exited: true, reaped: true };
                self.host.log(
                    Level::Error,
                    "memory sample did not settle; retaining process host ownership",
                    Vec::new(),
                );
                self.host.incomplete_cleanup(observation);
                self.status.send_modify(|s| s.incomplete = true);
                let _ = sample.await;
            }
        }
        if let Some(init) = self.init_reply.take() {
            let _ = init.send(Err(InitError::Closed));
        }
        if let Some((_, job)) = self.job_reply.take() {
            let _ = job.send(Err(StartJobError::Closed));
        }
        self.host.release();
        let reaped = self.exited && self.reaped;
        self.status.send_modify(|status| {
            status.reaped = reaped;
            status.cleaned = true;
        });
    }

    fn now(&self) -> Duration {
        Instant::now().duration_since(self.origin)
    }

    /// One input: what is due first (a reply at a deadline cannot revive an expired phase), then
    /// the input, which after the exit only `after_exit` inputs still reach.
    fn input(&mut self, after_exit: bool, f: impl FnOnce(&mut Self)) {
        if self.cleaned() {
            return;
        }
        self.tick();
        if self.exited && !after_exit {
            return;
        }
        f(self);
    }

    fn publish(&self) {
        let failure = self.protocol_failure.as_deref().map(Into::into);
        let (killed, kill_settled, exited) =
            (self.killed, self.exited || self.kill_ended, self.exited);
        self.status.send_modify(|s| {
            s.failure = failure;
            s.killed = killed;
            s.kill_settled = kill_settled;
            s.exited = exited;
        });
    }

    /// Closing or killed: the memory monitor's failures are no longer worth a log.
    fn stopping(&self) -> bool {
        self.closing || self.killed || self.exited
    }

    fn cleaned(&self) -> bool {
        self.launch_rejected
            || (self.exited && self.reaped && (!self.initialized_requested || self.reader_closed))
    }

    fn wake_at(&self) -> Option<Duration> {
        if self.cleaned() {
            return None;
        }
        if self.exited {
            return [self.cleanup_at, self.reader_deadline].into_iter().flatten().min();
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

    fn launched(&mut self, pid: u32) {
        self.opts.pid = pid;
        if !self.initialized_requested && !self.closing {
            self.startup_at = Some(self.now() + secs(self.opts.initialize_timeout));
        }
        if self.closing && !self.exited {
            self.kill_ended = false;
            self.begin_kill();
        }
    }

    fn launch_rejected(&mut self, cause: String) {
        self.launch_rejected = true;
        self.startup_at = None;
        self.init = None;
        self.cleanup_at = None;
        self.reply_init(Err(InitError::Failed(cause.clone())));
        self.log(
            Level::Error,
            &format!("process launch failed before creating a process: {cause}"),
        );
    }

    fn launch_failed(&mut self, cause: String) {
        self.startup_at = None;
        self.closing = true;
        if self.init.take().is_some() {
            self.reply_init(Err(InitError::Failed(cause.clone())));
        }
        self.log(Level::Error, &format!("process launch failed: {cause}"));
        self.begin_kill();
        if !(self.exited && self.reaped) {
            self.cleanup_at.get_or_insert(self.now() + self.termination_timeout());
        }
    }

    fn on_reaped(&mut self, code: i64) {
        self.observe_exitcode(Some(code));
        self.reaped = true;
        if self.exited {
            self.cleanup_at = None;
        }
    }

    fn initialize(&mut self, hello: Hello) {
        if self.initialized_requested || self.closing || self.exited {
            self.reply_init(Err(InitError::AlreadyInitializing));
            return;
        }
        self.initialized_requested = true;
        self.log(Level::Info, &format!("initializing {}", self.opts.runner()));
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
        // a job thread has no pong deadline (`ping_timeout` 0): it is only reported
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
                        Err(StartJobError::InvalidFields(format!("a {size}-byte job is too large"))),
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

    fn kill(&mut self) {
        if !self.opts.killable {
            self.shutdown();
            return;
        }
        self.closing = true;
        self.begin_kill();
        self.cleanup_at.get_or_insert(self.now() + self.termination_timeout());
    }

    fn on_exited(&mut self, code: Option<i64>) {
        let now = self.now();
        self.exited = true;
        if self.initialized_requested && !self.reader_closed {
            self.reader_deadline = Some(now + secs(self.opts.close_timeout));
        }
        self.startup_at = None;
        self.cleanup_at.get_or_insert(now + secs(self.opts.close_timeout));
        if self.reader_closed && self.init.take().is_some() {
            self.reply_init(Err(InitError::Closed));
        }
        if self.reaped {
            self.cleanup_at = None;
        }
        (self.next_ping, self.pong_deadline, self.next_sample) = (None, None, None);
        self.close = Close::None;
        self.observe_exitcode(code);
    }

    /// What is due now.
    fn tick(&mut self) {
        if self.cleaned() {
            return;
        }
        let now = self.now();
        if self.cleanup_at.is_some_and(|at| at <= now) {
            self.cleanup_at = None;
            self.log(Level::Error, "incomplete process cleanup; retaining ownership until reaping");
            self.host.incomplete_cleanup(CleanupObservation {
                pid: self.opts.pid,
                exited: self.exited,
                reaped: self.reaped,
            });
            self.status.send_modify(|s| s.incomplete = true);
        }
        if self.reader_deadline.is_some_and(|at| at <= now) {
            self.reader_deadline = None;
            self.fail_protocol(Arc::new(ProtocolError::Channel(ChannelError::Io {
                kind: std::io::ErrorKind::TimedOut,
                os_error: None,
                message: "IPC reader did not settle after process exit".into(),
            })));
        }
        if self.exited {
            return;
        }
        if self.startup_at.is_some_and(|at| at <= now) {
            self.startup_at = None;
            self.closing = true;
            self.cleanup_at.get_or_insert(now + self.termination_timeout());
            self.log(
                Level::Error,
                "process launch did not enter initialization before its deadline",
            );
            self.begin_kill();
        }
        if let Some((_, at)) = self.init
            && at <= now
        {
            self.init = None;
            self.reply_init(Ok(None));
            if self.opts.killable {
                self.begin_kill();
            }
        }
        if let Some((request_id, deadline)) = self.pending_job
            && deadline <= now
        {
            self.pending_job = None;
            self.reply_job(request_id, Err(StartJobError::Timeout));
            self.closing = true;
            self.begin_kill();
            self.cleanup_at.get_or_insert(now + self.termination_timeout());
        }
        if let Some(at) = self.next_ping
            && at <= now
        {
            self.send(ToChild::Ping { sent_ms: unix_now_ms().max(0) as u64 });
            self.next_ping =
                Some(now + secs(self.opts.ping_interval).max(Duration::from_millis(1)));
        }
        if let Some(at) = self.pong_deadline
            && at <= now
        {
            self.pong_deadline = None;
            self.log(Level::Error, "process is unresponsive, killing process");
            self.begin_kill();
        }
        if let Some(at) = self.next_sample
            && at <= now
            && self.sampling.is_none()
            && !self.stopping()
        {
            self.next_sample = None;
            let host = self.host.clone();
            // Admission before any task: a saturated sampler retries at its next cadence.
            match self.blocking_io.try_spawn(move || host.sample_memory()) {
                Ok(receipt) => self.sampling = Some(receipt),
                Err(_) => self.on_sample(Sample::Failed),
            }
        }
        match self.close {
            Close::Ack(at) if at <= now => {
                if self.opts.killable {
                    self.log(Level::Error, "process did not ack shutdown in time, killing process");
                    self.begin_kill();
                } else {
                    self.log(Level::Error, "job did not ack shutdown in time");
                }
                self.close = Close::ShuttingDown(now + secs(self.opts.close_timeout));
                self.advance_close();
            }
            Close::ShuttingDown(at) | Close::Exit(at) if at <= now => {
                if self.opts.killable {
                    self.log(Level::Error, "process did not exit in time, killing process");
                    self.begin_kill();
                } else {
                    self.log(Level::Error, "job shutdown is taking too much time..");
                }
                self.close = Close::Waiting;
            }
            _ => {}
        }
    }

    fn on_message(&mut self, msg: ToParent) {
        if self.exited {
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
                    self.host.log(
                        Level::Info,
                        &format!("{} initialized", self.opts.runner()),
                        extra,
                    );
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
                let delay = unix_now_ms().saturating_sub(sent_ms as i64);
                if delay as f64 > self.opts.high_ping_threshold * 1000.0 {
                    let msg = if self.opts.killable {
                        "process is unresponsive"
                    } else {
                        "job executor is unresponsive"
                    };
                    self.host.log(Level::Warning, msg, vec![("delay", ExtraValue::Int(delay))]);
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
                    self.cleanup_at.get_or_insert(now + self.termination_timeout());
                }
                self.advance_close();
            }
            ToParent::Exiting { reason } => {
                let extra = vec![("reason", ExtraValue::Str(reason))];
                match self.opts.killable {
                    true => self.host.log(Level::Info, "process exiting", extra),
                    false => self.host.log(Level::Debug, "job exiting", extra),
                }
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
                        self.closing = true;
                        self.begin_kill();
                        self.cleanup_at.get_or_insert(now + self.termination_timeout());
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

    /// The first protocol failure remains authoritative even when OS exit reports zero.
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
        self.begin_kill();
        if !(self.exited && self.reaped) {
            self.cleanup_at.get_or_insert(self.now() + self.termination_timeout());
        }
    }

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

    /// Kills the process unless it is already dead; a failed kill can be retried.
    fn begin_kill(&mut self) {
        if !self.opts.killable || self.killed || self.exited {
            return;
        }
        if !self.host.is_alive() {
            self.kill_ended = true;
            return;
        }
        self.log(Level::Info, "killing process");
        match self.host.kill() {
            Ok(()) => {
                self.killed = true;
                self.kill_ended = true;
            }
            Err(cause) => {
                self.log(Level::Error, &format!("process termination failed: {cause}"));
                self.host.signal_failed(SignalFailure { pid: self.opts.pid, cause });
            }
        }
        self.cleanup_at.get_or_insert(self.now() + self.termination_timeout());
    }

    fn observe_exitcode(&mut self, code: Option<i64>) {
        if self.exitcode.is_none()
            && let Some(code) = code
        {
            self.exitcode = Some(code);
            if code != 0 && !self.killed {
                self.log(Level::Error, &format!("process exited with non-zero exit code {code}"));
            }
        }
    }

    fn termination_timeout(&self) -> Duration {
        secs(self.opts.close_timeout)
    }

    fn send(&mut self, message: ToChild) {
        if self.reader_closed || self.exited {
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

    fn on_sample(&mut self, sample: Sample) {
        let now = self.now();
        match sample {
            Sample::Mb(memory_mb, metric) => {
                let baseline = *self.baseline_mb.get_or_insert(memory_mb);
                let (warn_mb, limit_mb) = (self.opts.memory_warn_mb, self.opts.memory_limit_mb);
                if limit_mb > 0.0 && memory_mb > limit_mb {
                    let extra = self.memory_extra(now, memory_mb, &metric, baseline);
                    self.host.log(
                        Level::Error,
                        "job process exceeded memory limit, killing it",
                        extra,
                    );
                    self.begin_kill();
                } else if warn_mb > 0.0
                    && memory_mb > warn_mb
                    && self.gate.should_warn(memory_mb, now)
                {
                    // without a hard limit nothing is terminated: say so
                    let advisory = match limit_mb <= 0.0 {
                        true => " (advisory only, the process will not be terminated)",
                        false => "",
                    };
                    let msg = format!(
                        "job process memory usage is above the warning threshold{advisory}"
                    );
                    let extra = self.memory_extra(now, memory_mb, &metric, baseline);
                    self.host.log(Level::Warning, &msg, extra);
                }
            }
            Sample::Unavailable(e) => {
                if !self.stopping() {
                    self.log(
                        Level::Warning,
                        &format!("failed to get memory info for process: {e}"),
                    );
                }
                return;
            }
            Sample::Interrupted => return,
            Sample::Failed => {}
        }
        if !self.stopping() {
            self.next_sample = Some(now + MEMORY_MONITOR_INTERVAL);
        }
    }

    fn memory_extra(&self, now: Duration, memory_mb: f64, metric: &str, baseline_mb: f64) -> Extra {
        vec![
            ("memory_usage_mb", ExtraValue::Float(round(memory_mb, 1))),
            ("memory_metric", ExtraValue::Str(metric.to_owned())),
            ("memory_warn_mb", ExtraValue::Float(self.opts.memory_warn_mb)),
            ("memory_limit_mb", ExtraValue::Float(self.opts.memory_limit_mb)),
            ("uptime", ExtraValue::Float(round(now.as_secs_f64(), 1))),
            ("has_running_job", ExtraValue::Bool(self.running_job)),
            ("baseline_memory_mb", ExtraValue::Float(round(baseline_mb, 1))),
            ("growth_memory_mb", ExtraValue::Float(round(memory_mb - baseline_mb, 1))),
        ]
    }
}

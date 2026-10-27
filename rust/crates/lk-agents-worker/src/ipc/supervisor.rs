//! The worker's side of one job process: its initialization, pings, memory, shutdown and kill.
//!
//! [`Supervision`] is the decisions, sans I/O: it takes the [`Input`]s (what the host asked, what
//! the child sent, a sample, the exit) at an instant, says what to do as [`Effect`]s, and keeps
//! every timeout as a deadline ([`Supervision::next_deadline`], acted on by
//! [`Supervision::tick`]). [`ProcSupervisor`] is the thin async driver around it: one task owning
//! the socket's writing half, the state, and a reader task's messages.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::wire::{self, Hello, PROTOCOL_VERSION, StartJob, ToChild, ToParent, WireError};

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;
/// Between SIGUSR1 (the child dumps its stacks) and SIGKILL.
pub const KILL_GRACE: Duration = Duration::from_millis(500);
/// The bound of the driver's queues.
const QUEUE: usize = 16;

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

/// Why a memory sample failed.
pub enum MemoryError<E> {
    /// The process is gone or not ours to read (`psutil.NoSuchProcess`, `psutil.AccessDenied`):
    /// warn once and stop monitoring.
    Unavailable(String),
    /// Anything else: log it and try again on the next sample.
    Failed(E),
    /// An interruption rather than an error (a `BaseException` such as `SystemExit`): stop
    /// monitoring without a log.
    Interrupted,
}

/// What the supervisor needs from the host language: logging and the process handle.
pub trait ProcHost: Send + Sync + 'static {
    type Error: Send + 'static;

    fn log(&self, level: Level, msg: &str, extra: Extra);
    /// An error log with the failure's traceback.
    fn log_exception(&self, msg: &str, err: Self::Error);
    fn is_alive(&self) -> bool;
    /// SIGKILL (`Process.kill`; a terminate on Windows).
    fn kill(&self);
    /// The process's footprint in MB and the metric it reflects (pss, uss or rss). Blocking: the
    /// supervisor calls it off the runtime.
    fn sample_memory(&self) -> Result<(f64, String), MemoryError<Self::Error>>;
    /// The process exited and was reaped: the host's handle can go.
    fn release(&self) {}
}

#[derive(Clone, Debug)]
pub struct SupervisorOptions {
    pub pid: u32,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub ping_interval: f64,
    pub ping_timeout: f64,
    pub high_ping_threshold: f64,
    /// A process can be killed; a job thread cannot: its late initialization, ack or exit is
    /// logged and waited for instead, and `kill` asks it to shut down.
    pub killable: bool,
}

impl SupervisorOptions {
    /// What the logs call the supervised runner.
    fn runner(&self) -> &'static str {
        if self.killable { "process" } else { "job runner" }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("duplex closed")]
    Closed,
    #[error("first message must be Ready")]
    Unexpected,
    #[error("process initialization failed: {0}")]
    Failed(String),
    #[error("the job process speaks protocol version {0}, the worker {PROTOCOL_VERSION}")]
    Version(u32),
}

/// Seconds as a `Duration`: zero when negative or NaN.
pub(crate) fn secs(s: f64) -> Duration {
    if s.is_nan() || s <= 0.0 { Duration::ZERO } else { Duration::from_secs_f64(s.min(1e9)) }
}

/// `round(x, digits)`: Python rounds the exact binary value half to even, as formatting does.
fn round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// `utils.time_ms()`: wall-clock milliseconds, rounded.
pub fn time_ms() -> i64 {
    (lk_agents_abi::unix_now() * 1000.0 + 0.5) as i64
}

// ------------------------------------------------------------------------------- the decisions

#[derive(Debug)]
pub enum Input {
    /// Send `Hello` and wait for `Ready`, within the initialize timeout.
    Initialize(Hello),
    /// The initialization settled: pings, the pong deadline and the memory monitor start.
    Supervise,
    StartJob(Box<StartJob>),
    Message(ToParent),
    /// The child's end closed, or (`Some`) sent a frame this side cannot read.
    ReaderClosed(Option<String>),
    /// The graceful close: `Shutdown`, then the ack and the exit, each within the close timeout.
    Shutdown,
    Kill,
    /// The kill sequence's first step ran: whether a signal went out (the stack dump), `None`
    /// when the process was already dead.
    KillBegun(Option<bool>),
    Sampled(Sample),
    /// The process exited and was reaped.
    Exited(Option<i64>),
}

#[derive(Debug)]
pub enum Sample {
    Mb(f64, String),
    Unavailable(String),
    /// Logged by the driver, which holds the host's error.
    Failed,
    Interrupted,
}

#[derive(Debug)]
pub enum Effect {
    Send(ToChild),
    Log(Level, String, Extra),
    /// The kill sequence's first step: unless dead, SIGUSR1 (the child dumps its stacks; on
    /// Windows a `DumpStacks`), then `KillBegun`.
    BeginKill,
    /// SIGKILL, unless already dead.
    Kill,
    SampleMemory,
    /// The initialization settled: how long it took, `None` when it timed out.
    Initialized(Result<Option<Duration>, InitError>),
}

/// Where a graceful close is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Close {
    None,
    Ack(Instant),
    ShuttingDown,
    Exit(Instant),
    Waiting,
}

/// Rate-limits the high memory warning: once per cooldown, or sooner when usage grew.
#[derive(Default)]
pub(crate) struct WarnGate {
    last: Option<(Instant, f64)>,
}

impl WarnGate {
    pub(crate) fn should_warn(&mut self, memory_mb: f64, now: Instant) -> bool {
        let emit = match self.last {
            None => true,
            Some((at, mb)) => {
                now.duration_since(at) >= MEMORY_WARN_COOLDOWN
                    || memory_mb - mb >= MEMORY_WARN_RESET_DELTA_MB
            }
        };
        if emit {
            self.last = Some((now, memory_mb));
        }
        emit
    }
}

/// One process's supervision, sans I/O.
pub struct Supervision {
    opts: SupervisorOptions,
    spawned: Instant,
    /// Waiting for `Ready`: since when, and until when.
    init: Option<(Instant, Instant)>,
    supervising: bool,
    next_ping: Option<Instant>,
    pong_deadline: Option<Instant>,
    next_sample: Option<Instant>,
    sampling: bool,
    baseline_mb: Option<f64>,
    gate: WarnGate,
    close: Close,
    killing: bool,
    kill_at: Option<Instant>,
    acked: bool,
    shutting_down: bool,
    closing: bool,
    killed: bool,
    kill_ended: bool,
    exited: bool,
    running_job: bool,
    reader_closed: bool,
}

impl Supervision {
    pub fn new(opts: SupervisorOptions, now: Instant) -> Self {
        Self {
            opts,
            spawned: now,
            init: None,
            supervising: false,
            next_ping: None,
            pong_deadline: None,
            next_sample: None,
            sampling: false,
            baseline_mb: None,
            gate: WarnGate::default(),
            close: Close::None,
            killing: false,
            kill_at: None,
            acked: false,
            shutting_down: false,
            closing: false,
            killed: false,
            kill_ended: false,
            exited: false,
            running_job: false,
            reader_closed: false,
        }
    }

    /// The kill sequence ran on a live process.
    pub fn killed(&self) -> bool {
        self.killed
    }

    pub fn exited(&self) -> bool {
        self.exited
    }

    /// Closing or killed: the memory monitor's failures are no longer worth a log.
    pub fn stopping(&self) -> bool {
        self.closing || self.killed
    }

    /// When [`tick`](Self::tick) has something to do.
    pub fn next_deadline(&self) -> Option<Instant> {
        let close = match self.close {
            Close::Ack(at) | Close::Exit(at) => Some(at),
            _ => None,
        };
        let sample = self.next_sample.filter(|_| !self.sampling && !self.stopping());
        [
            self.init.map(|(_, at)| at),
            self.next_ping,
            self.pong_deadline,
            sample,
            close,
            self.kill_at,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    pub fn handle(&mut self, now: Instant, input: Input) -> Vec<Effect> {
        let mut out = Vec::new();
        if self.exited && !matches!(input, Input::KillBegun(_)) {
            return out;
        }
        match input {
            Input::Initialize(hello) => {
                let msg = format!("initializing {}", self.opts.runner());
                out.push(Effect::Log(Level::Info, msg, Vec::new()));
                out.push(Effect::Send(ToChild::Hello(hello)));
                match self.reader_closed {
                    true => out.push(Effect::Initialized(Err(InitError::Closed))),
                    false => self.init = Some((now, now + secs(self.opts.initialize_timeout))),
                }
            }
            Input::Supervise => {
                if !self.supervising {
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
            }
            Input::StartJob(job) => {
                self.running_job = true;
                out.push(Effect::Send(ToChild::StartJob(job)));
            }
            Input::Message(msg) => self.on_message(now, msg, &mut out),
            Input::ReaderClosed(reason) => {
                self.reader_closed = true;
                if let Some(reason) = reason {
                    out.push(Effect::Log(
                        Level::Error,
                        format!("invalid IPC message: {reason}"),
                        Vec::new(),
                    ));
                }
                if self.init.take().is_some() {
                    out.push(Effect::Initialized(Err(InitError::Closed)));
                }
                // a close never waits for an ack that cannot arrive
                self.acked = true;
                self.shutting_down = true;
                self.advance_close(now);
            }
            Input::Shutdown => self.shutdown(now, &mut out),
            Input::Kill if !self.opts.killable => self.shutdown(now, &mut out),
            Input::Kill => {
                self.closing = true;
                self.begin_kill(&mut out);
            }
            Input::KillBegun(Some(true)) => self.kill_at = Some(now + KILL_GRACE),
            Input::KillBegun(Some(false)) => self.finish_kill(&mut out),
            Input::KillBegun(None) => {
                self.killing = false;
                self.kill_ended = true;
            }
            Input::Sampled(sample) => self.on_sample(now, sample, &mut out),
            Input::Exited(code) => {
                self.exited = true;
                self.init = None;
                (self.next_ping, self.pong_deadline, self.next_sample, self.kill_at) =
                    (None, None, None, None);
                self.close = Close::None;
                self.acked = true;
                self.shutting_down = true;
                if code != Some(0) && !self.killed {
                    let code = code.map_or("None".to_owned(), |c| c.to_string());
                    let msg = format!("process exited with non-zero exit code {code}");
                    out.push(Effect::Log(Level::Error, msg, Vec::new()));
                }
            }
        }
        out
    }

    /// What is due at `now`.
    pub fn tick(&mut self, now: Instant) -> Vec<Effect> {
        let mut out = Vec::new();
        if self.exited {
            return out;
        }
        if let Some((_, at)) = self.init
            && at <= now
        {
            self.init = None;
            out.push(Effect::Initialized(Ok(None)));
            if self.opts.killable {
                self.begin_kill(&mut out);
            }
        }
        if let Some(at) = self.next_ping
            && at <= now
        {
            out.push(Effect::Send(ToChild::Ping { sent_ms: time_ms().max(0) as u64 }));
            self.next_ping =
                Some(now + secs(self.opts.ping_interval).max(Duration::from_millis(1)));
        }
        if let Some(at) = self.pong_deadline
            && at <= now
        {
            self.pong_deadline = None;
            let msg = "process is unresponsive, killing process".to_owned();
            out.push(Effect::Log(Level::Error, msg, Vec::new()));
            self.begin_kill(&mut out);
        }
        if let Some(at) = self.next_sample
            && at <= now
            && !self.sampling
            && !self.stopping()
        {
            self.sampling = true;
            self.next_sample = None;
            out.push(Effect::SampleMemory);
        }
        match self.close {
            Close::Ack(at) if at <= now => {
                if self.opts.killable {
                    let msg = "process did not ack shutdown in time, killing process".to_owned();
                    out.push(Effect::Log(Level::Error, msg, Vec::new()));
                    self.begin_kill(&mut out);
                } else {
                    out.push(Effect::Log(
                        Level::Error,
                        "job did not ack shutdown in time".into(),
                        Vec::new(),
                    ));
                }
                self.close = Close::ShuttingDown;
                self.advance_close(now);
            }
            Close::Exit(at) if at <= now => {
                if self.opts.killable {
                    let msg = "process did not exit in time, killing process".to_owned();
                    out.push(Effect::Log(Level::Error, msg, Vec::new()));
                    self.begin_kill(&mut out);
                } else {
                    let msg = "job shutdown is taking too much time..".to_owned();
                    out.push(Effect::Log(Level::Error, msg, Vec::new()));
                }
                self.close = Close::Waiting;
            }
            _ => {}
        }
        if let Some(at) = self.kill_at
            && at <= now
        {
            self.kill_at = None;
            self.finish_kill(&mut out);
        }
        out
    }

    /// Whether the kill sequence asked of it is over: the process was killed, found dead, or
    /// exited.
    pub fn kill_settled(&self) -> bool {
        self.exited || self.kill_ended
    }

    fn on_message(&mut self, now: Instant, msg: ToParent, out: &mut Vec<Effect>) {
        if let Some((started, _)) = self.init.take() {
            let result = match msg {
                ToParent::Ready { version, .. } if version != PROTOCOL_VERSION => {
                    Err(InitError::Version(version))
                }
                ToParent::Ready { error: Some(error), .. } => Err(InitError::Failed(error)),
                ToParent::Ready { error: None, .. } => {
                    let elapsed = now.duration_since(started);
                    let extra =
                        vec![("elapsed_time", ExtraValue::Float(round(elapsed.as_secs_f64(), 2)))];
                    out.push(Effect::Log(
                        Level::Info,
                        format!("{} initialized", self.opts.runner()),
                        extra,
                    ));
                    Ok(Some(elapsed))
                }
                _ => Err(InitError::Unexpected),
            };
            out.push(Effect::Initialized(result));
            return;
        }
        match msg {
            ToParent::Pong { sent_ms } => {
                let delay = time_ms() - sent_ms as i64;
                if delay as f64 > self.opts.high_ping_threshold * 1000.0 {
                    let msg = if self.opts.killable {
                        "process is unresponsive"
                    } else {
                        "job executor is unresponsive"
                    };
                    out.push(Effect::Log(
                        Level::Warning,
                        msg.into(),
                        vec![("delay", ExtraValue::Int(delay))],
                    ));
                }
                // a late pong does not revive a deadline that fired
                if self.pong_deadline.is_some() {
                    self.pong_deadline = Some(now + secs(self.opts.ping_timeout));
                }
            }
            ToParent::ShutdownAck => {
                self.acked = true;
                self.advance_close(now);
            }
            ToParent::ShuttingDown => {
                self.shutting_down = true;
                self.advance_close(now);
            }
            ToParent::Exiting { reason } => {
                let extra = vec![("reason", ExtraValue::Str(reason))];
                match self.opts.killable {
                    true => out.push(Effect::Log(Level::Info, "process exiting".into(), extra)),
                    false => out.push(Effect::Log(Level::Debug, "job exiting".into(), extra)),
                }
            }
            ToParent::Ready { .. } => {}
        }
    }

    fn shutdown(&mut self, now: Instant, out: &mut Vec<Effect>) {
        if self.close != Close::None {
            return;
        }
        self.closing = true;
        out.push(Effect::Send(ToChild::Shutdown { reason: String::new() }));
        self.close = Close::Ack(now + secs(self.opts.close_timeout));
        self.advance_close(now);
    }

    fn advance_close(&mut self, now: Instant) {
        if matches!(self.close, Close::Ack(_)) && self.acked {
            self.close = Close::ShuttingDown;
        }
        if self.close == Close::ShuttingDown && self.shutting_down {
            self.close = Close::Exit(now + secs(self.opts.close_timeout));
        }
    }

    fn begin_kill(&mut self, out: &mut Vec<Effect>) {
        if !self.killing {
            self.killing = true;
            out.push(Effect::BeginKill);
        }
    }

    fn finish_kill(&mut self, out: &mut Vec<Effect>) {
        out.push(Effect::Kill);
        self.killing = false;
        self.killed = true;
        self.kill_ended = true;
    }

    fn on_sample(&mut self, now: Instant, sample: Sample, out: &mut Vec<Effect>) {
        self.sampling = false;
        match sample {
            Sample::Mb(memory_mb, metric) => {
                let baseline = *self.baseline_mb.get_or_insert(memory_mb);
                let (warn_mb, limit_mb) = (self.opts.memory_warn_mb, self.opts.memory_limit_mb);
                if limit_mb > 0.0 && memory_mb > limit_mb {
                    let extra = self.memory_extra(now, memory_mb, &metric, baseline);
                    let msg = "job process exceeded memory limit, killing it".to_owned();
                    out.push(Effect::Log(Level::Error, msg, extra));
                    self.begin_kill(out);
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
                    out.push(Effect::Log(Level::Warning, msg, extra));
                }
            }
            Sample::Unavailable(e) => {
                if !self.stopping() {
                    let msg = format!("failed to get memory info for process: {e}");
                    out.push(Effect::Log(Level::Warning, msg, Vec::new()));
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

    fn memory_extra(&self, now: Instant, memory_mb: f64, metric: &str, baseline_mb: f64) -> Extra {
        vec![
            ("memory_usage_mb", ExtraValue::Float(round(memory_mb, 1))),
            ("memory_metric", ExtraValue::Str(metric.to_owned())),
            ("memory_warn_mb", ExtraValue::Float(self.opts.memory_warn_mb)),
            ("memory_limit_mb", ExtraValue::Float(self.opts.memory_limit_mb)),
            ("uptime", ExtraValue::Float(round(now.duration_since(self.spawned).as_secs_f64(), 1))),
            ("has_running_job", ExtraValue::Bool(self.running_job)),
            ("baseline_memory_mb", ExtraValue::Float(round(baseline_mb, 1))),
            ("growth_memory_mb", ExtraValue::Float(round(memory_mb - baseline_mb, 1))),
        ]
    }
}

// ---------------------------------------------------------------------------------- the driver

enum Command {
    Input(Input),
    Initialize(Hello, oneshot::Sender<Result<Option<Duration>, InitError>>),
}

/// What the driver publishes.
#[derive(Clone, Copy, Default)]
struct Status {
    killed: bool,
    kill_settled: bool,
    exited: bool,
}

/// The worker's side of one job process. The host spawns the process, hands over its end of the
/// socket, calls [`initialize`](Self::initialize) then [`supervise`](Self::supervise), and reports
/// the exit with [`exited`](Self::exited). Needs a runtime context to be made.
pub struct ProcSupervisor {
    commands: mpsc::Sender<Command>,
    status: watch::Receiver<Status>,
}

impl Clone for ProcSupervisor {
    fn clone(&self) -> Self {
        Self { commands: self.commands.clone(), status: self.status.clone() }
    }
}

impl ProcSupervisor {
    pub fn new<S, H>(stream: S, host: H, opts: SupervisorOptions) -> Self
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
        H: ProcHost,
    {
        let (mut reader, writer) = tokio::io::split(stream);
        let (messages_tx, messages) = mpsc::channel(QUEUE);
        let reading = tokio::spawn(async move {
            loop {
                let read = wire::read::<ToParent, _>(&mut reader).await;
                // the socket's end closes the reader quietly; a frame it cannot read, with why
                let end = match read {
                    Ok(msg) => {
                        if messages_tx.send(Ok(msg)).await.is_err() {
                            return;
                        }
                        continue;
                    }
                    Err(WireError::Io(_)) => return,
                    Err(e) => e,
                };
                let _ = messages_tx.send(Err(end)).await;
                return;
            }
        });
        let (commands, commands_rx) = mpsc::channel(QUEUE);
        let (status_tx, status) = watch::channel(Status::default());
        let state = Supervision::new(opts.clone(), Instant::now());
        let driver = Driver {
            state,
            opts,
            host: Arc::new(host),
            writer: Some(Box::new(writer)),
            init: None,
            status: status_tx,
        };
        tokio::spawn(driver.run(commands_rx, messages, reading));
        Self { commands, status }
    }

    /// The kill sequence ran on a live process.
    pub fn killed(&self) -> bool {
        self.status.borrow().killed
    }

    async fn input(&self, input: Input) {
        let _ = self.commands.send(Command::Input(input)).await;
    }

    async fn until(&self, done: impl Fn(&Status) -> bool) {
        let mut status = self.status.clone();
        let _ = status.wait_for(|s| s.exited || done(s)).await;
    }

    /// Sends `hello` and waits for the process's `Ready`: how long it took, `None` when it timed
    /// out (a process is then killed).
    pub async fn initialize(&self, hello: Hello) -> Result<Option<Duration>, InitError> {
        let (tx, rx) = oneshot::channel();
        if self.commands.send(Command::Initialize(hello, tx)).await.is_err() {
            return Err(InitError::Closed);
        }
        rx.await.unwrap_or(Err(InitError::Closed))
    }

    /// Starts the pings and (with a memory threshold) the memory monitor. Called once the
    /// initialization settled, whatever its outcome.
    pub async fn supervise(&self) {
        self.input(Input::Supervise).await;
    }

    /// Hands the process its job.
    pub async fn start_job(&self, job: StartJob) {
        self.input(Input::StartJob(Box::new(job))).await;
    }

    /// The graceful close: `Shutdown`, then the ack and the exit within the close timeout each,
    /// killing the process when either is late. Returns once it exited or was killed.
    pub async fn shutdown(&self) {
        self.input(Input::Shutdown).await;
        self.until(|s| s.killed).await;
    }

    /// The kill sequence; a runner that cannot be killed is asked to shut down instead. Returns
    /// once it is over.
    pub async fn kill(&self) {
        self.input(Input::Kill).await;
        self.until(|s| s.kill_settled).await;
    }

    /// The process exited and was reaped with `exitcode`: the supervision ends and the socket
    /// closes.
    pub async fn exited(&self, exitcode: Option<i64>) {
        self.input(Input::Exited(exitcode)).await;
        self.until(|_| false).await;
    }
}

struct Driver<H: ProcHost> {
    state: Supervision,
    opts: SupervisorOptions,
    host: Arc<H>,
    writer: Option<Box<dyn AsyncWrite + Send + Unpin>>,
    init: Option<oneshot::Sender<Result<Option<Duration>, InitError>>>,
    status: watch::Sender<Status>,
}

type Messages = mpsc::Receiver<Result<ToParent, WireError>>;

impl<H: ProcHost> Driver<H> {
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        mut messages: Messages,
        reading: tokio::task::JoinHandle<()>,
    ) {
        let mut reader_open = true;
        let mut sampling: Option<
            tokio::task::JoinHandle<Result<(f64, String), MemoryError<H::Error>>>,
        > = None;
        let far = || Instant::now() + Duration::from_secs(1 << 30);
        while !self.state.exited() {
            let deadline = self.state.next_deadline();
            let input = tokio::select! {
                command = commands.recv() => match command {
                    Some(Command::Input(input)) => input,
                    Some(Command::Initialize(hello, reply)) => {
                        self.init = Some(reply);
                        Input::Initialize(hello)
                    }
                    None => break,
                },
                msg = messages.recv(), if reader_open => match msg {
                    Some(Ok(msg)) => Input::Message(msg),
                    Some(Err(e)) => {
                        reader_open = false;
                        Input::ReaderClosed(Some(e.to_string()))
                    }
                    None => {
                        reader_open = false;
                        Input::ReaderClosed(None)
                    }
                },
                sample = async { sampling.as_mut().expect("sampling").await }, if sampling.is_some() => {
                    sampling = None;
                    Input::Sampled(match sample {
                        Ok(Ok((mb, metric))) => Sample::Mb(mb, metric),
                        Ok(Err(MemoryError::Unavailable(e))) => Sample::Unavailable(e),
                        Ok(Err(MemoryError::Failed(e))) => {
                            if !self.state.stopping() {
                                self.host.log_exception("Error in memory monitoring task", e);
                            }
                            Sample::Failed
                        }
                        Ok(Err(MemoryError::Interrupted)) | Err(_) => Sample::Interrupted,
                    })
                }
                () = tokio::time::sleep_until(deadline.unwrap_or_else(far)), if deadline.is_some() => {
                    let effects = self.state.tick(Instant::now());
                    self.execute(effects, &mut sampling).await;
                    self.publish();
                    continue;
                }
            };
            let effects = self.state.handle(Instant::now(), input);
            self.execute(effects, &mut sampling).await;
            self.publish();
        }
        reading.abort();
        if let Some(sampling) = sampling {
            sampling.abort();
        }
        if let Some(mut writer) = self.writer.take() {
            let _ = writer.shutdown().await;
        }
        if let Some(init) = self.init.take() {
            let _ = init.send(Err(InitError::Closed));
        }
        self.host.release();
        self.status.send_modify(|s| s.exited = true);
    }

    fn publish(&self) {
        let status = Status {
            killed: self.state.killed(),
            kill_settled: self.state.kill_settled(),
            exited: false,
        };
        self.status.send_if_modified(|s| {
            let changed = (s.killed, s.kill_settled) != (status.killed, status.kill_settled);
            (s.killed, s.kill_settled) = (status.killed, status.kill_settled);
            changed
        });
    }

    async fn execute(
        &mut self,
        effects: Vec<Effect>,
        sampling: &mut Option<
            tokio::task::JoinHandle<Result<(f64, String), MemoryError<H::Error>>>,
        >,
    ) {
        let mut queue = std::collections::VecDeque::from(effects);
        while let Some(effect) = queue.pop_front() {
            match effect {
                Effect::Send(msg) => self.send(&msg).await,
                Effect::Log(level, msg, extra) => self.host.log(level, &msg, extra),
                Effect::BeginKill => {
                    let begun = self.begin_kill().await;
                    queue.extend(self.state.handle(Instant::now(), Input::KillBegun(begun)));
                }
                Effect::Kill => {
                    if self.host.is_alive() {
                        self.host.kill();
                    }
                }
                Effect::SampleMemory => {
                    let host = self.host.clone();
                    *sampling = Some(tokio::task::spawn_blocking(move || host.sample_memory()));
                }
                Effect::Initialized(result) => {
                    if let Some(init) = self.init.take() {
                        let _ = init.send(result);
                    }
                }
            }
        }
    }

    async fn send(&mut self, msg: &ToChild) {
        if let Some(writer) = self.writer.as_mut()
            && wire::write(writer, msg).await.is_err()
        {
            self.writer = None;
        }
    }

    /// The kill sequence's first step: whether a signal went out, `None` when the process was
    /// already dead.
    async fn begin_kill(&mut self) -> Option<bool> {
        let mut signalled = false;
        // where SIGUSR1 does not exist, the child is asked over the socket
        let dump = std::env::var("LK_DUMP_STACK_TRACES")
            .is_ok_and(|v| !matches!(v.to_lowercase().as_str(), "0" | "false" | "no"));
        if cfg!(windows) && dump {
            self.host.log(
                Level::Info,
                "sending DumpStackTraceRequest message to process",
                Vec::new(),
            );
            self.send(&ToChild::DumpStacks).await;
            signalled = self.writer.is_some();
        }
        if !self.host.is_alive() {
            return None;
        }
        self.host.log(Level::Info, "killing process", Vec::new());
        #[cfg(unix)]
        if let Ok(pid) = libc::pid_t::try_from(self.opts.pid)
            && pid > 0
        {
            self.host.log(Level::Info, "sending SIGUSR1 signal to process", Vec::new());
            // SAFETY: kill(2) takes plain integers.
            signalled |= unsafe { libc::kill(pid, libc::SIGUSR1) } == 0;
        }
        Some(signalled)
    }
}

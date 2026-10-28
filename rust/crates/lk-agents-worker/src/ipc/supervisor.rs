//! The worker's side of one job process: its initialization, pings, memory, shutdown and kill.
//!
//! [`Supervision`] is the decisions, sans I/O: it takes the [`Input`]s (what the host asked, what
//! the child sent, a sample, the exit) at an instant, says what to do as [`Effect`]s, and keeps
//! every timeout as a deadline ([`Supervision::wake_at`], acted on by
//! [`Supervision::tick`]). Socket, timer and process execution live in `supervisor_driver`.

use std::collections::BTreeSet;
use std::time::Duration;

use super::wire::{Hello, PROTOCOL_VERSION, StartJob, ToChild, ToParent};

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;
/// Between SIGUSR1 (the child dumps its stacks) and SIGKILL.
pub const KILL_GRACE: Duration = Duration::from_millis(500);

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
    fn kill(&self) -> Result<(), String>;
    fn request_stack_dump(&self) -> Result<bool, String> {
        Ok(false)
    }
    /// The process's footprint in MB and the metric it reflects (pss, uss or rss). Blocking: the
    /// supervisor calls it off the runtime.
    fn sample_memory(&self) -> Result<(f64, String), MemoryError<Self::Error>>;
    /// The process exited and was reaped: the host's handle can go.
    fn release(&self) {}
    fn incomplete_cleanup(&self, _: CleanupObservation) {}
    fn signal_failed(&self, _: SignalFailure) {}
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
    #[error("initialization already requested")]
    AlreadyInitializing,
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

// ------------------------------------------------------------------------------- the decisions

#[derive(Debug)]
pub enum Input {
    Launched(u32),
    LaunchFailed(String),
    LaunchRejected(String),
    Reaped(i64),
    KillFinished(Result<(), String>),
    Written {
        id: u64,
        result: Result<(), String>,
    },
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
    Send {
        id: u64,
        message: ToChild,
    },
    IncompleteCleanup(CleanupObservation),
    SignalFailed(SignalFailure),
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

/// A native termination request failed; exit/reaping ownership remains unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalFailure {
    pub pid: u32,
    pub cause: String,
}

/// Cleanup missed its deadline; ownership stays with the driver until a reaping receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanupObservation {
    pub pid: u32,
    pub exited: bool,
    pub reaped: bool,
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
pub(crate) struct WarnGate {
    last: Option<(Duration, f64)>,
}

impl WarnGate {
    pub(crate) fn should_warn(&mut self, memory_mb: f64, now: Duration) -> bool {
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

/// One process's supervision, sans I/O.
pub struct Supervision {
    opts: SupervisorOptions,
    spawned: Duration,
    /// Waiting for `Ready`: since when, and until when.
    init: Option<(Duration, Duration)>,
    initialized_requested: bool,
    startup_at: Option<Duration>,
    supervising: bool,
    next_ping: Option<Duration>,
    pong_deadline: Option<Duration>,
    next_sample: Option<Duration>,
    sampling: bool,
    baseline_mb: Option<f64>,
    gate: WarnGate,
    close: Close,
    killing: bool,
    kill_at: Option<Duration>,
    acked: bool,
    shutting_down: bool,
    closing: bool,
    killed: bool,
    kill_ended: bool,
    exited: bool,
    running_job: bool,
    reader_closed: bool,
    reaped: bool,
    launch_rejected: bool,
    exitcode: Option<i64>,
    cleanup_at: Option<Duration>,
    next_write: u64,
    pending_writes: BTreeSet<u64>,
    protocol_failure: Option<String>,
}

impl Supervision {
    pub fn new(opts: SupervisorOptions, now: Duration) -> Self {
        Self {
            opts,
            spawned: now,
            init: None,
            initialized_requested: false,
            startup_at: None,
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
            reaped: false,
            launch_rejected: false,
            exitcode: None,
            cleanup_at: None,
            next_write: 0,
            pending_writes: BTreeSet::new(),
            protocol_failure: None,
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
        self.closing || self.killing || self.killed || self.exited
    }

    /// When [`tick`](Self::tick) has something to do.
    pub fn wake_at(&self) -> Option<Duration> {
        if self.cleaned() {
            return None;
        }
        if self.exited {
            return self.cleanup_at;
        }
        let close = match self.close {
            Close::Ack(at) | Close::ShuttingDown(at) | Close::Exit(at) => Some(at),
            _ => None,
        };
        let sample = self.next_sample.filter(|_| !self.sampling && !self.stopping());
        [
            self.startup_at,
            self.init.map(|(_, at)| at),
            self.next_ping,
            self.pong_deadline,
            sample,
            close,
            self.kill_at,
            self.cleanup_at,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    pub fn handle(&mut self, now: Duration, wall_ms: i64, input: Input) -> Vec<Effect> {
        let mut out = Vec::new();
        if self.cleaned() {
            return out;
        }
        // A reply received at a deadline cannot revive an expired phase.
        let mut due = self.tick(now, wall_ms);
        out.append(&mut due);
        if self.exited && !matches!(input, Input::Reaped(_) | Input::LaunchFailed(_)) {
            return out;
        }
        match input {
            Input::Launched(pid) => {
                self.opts.pid = pid;
                if !self.initialized_requested && !self.closing {
                    self.startup_at = Some(now + secs(self.opts.initialize_timeout));
                }
                if self.closing && !self.exited {
                    self.kill_ended = false;
                    self.begin_kill(&mut out);
                }
            }
            Input::LaunchRejected(cause) => {
                self.launch_rejected = true;
                self.startup_at = None;
                self.init = None;
                self.cleanup_at = None;
                out.push(Effect::Initialized(Err(InitError::Failed(cause.clone()))));
                out.push(Effect::Log(
                    Level::Error,
                    format!("process launch failed before creating a process: {cause}"),
                    Vec::new(),
                ));
            }
            Input::LaunchFailed(cause) => {
                self.startup_at = None;
                self.closing = true;
                if self.init.take().is_some() {
                    out.push(Effect::Initialized(Err(InitError::Failed(cause.clone()))));
                }
                out.push(Effect::Log(
                    Level::Error,
                    format!("process launch failed: {cause}"),
                    Vec::new(),
                ));
                self.begin_kill(&mut out);
                if !self.reaped() {
                    self.cleanup_at.get_or_insert(now + self.termination_timeout());
                }
            }
            Input::Reaped(code) => {
                self.observe_exitcode(Some(code), &mut out);
                self.reaped = true;
                if self.exited {
                    self.cleanup_at = None;
                }
            }
            Input::KillFinished(result) => {
                self.killing = false;
                match result {
                    Ok(()) => {
                        self.killed = true;
                        self.kill_ended = true;
                    }
                    Err(cause) => {
                        out.push(Effect::SignalFailed(SignalFailure { pid: self.opts.pid, cause }))
                    }
                }
            }
            Input::Written { id, result } => {
                if self.pending_writes.remove(&id)
                    && let Err(reason) = result
                {
                    out.extend(self.handle(now, wall_ms, Input::ReaderClosed(Some(reason))));
                }
            }
            Input::Initialize(hello) => {
                if self.initialized_requested || self.closing || self.exited {
                    out.push(Effect::Initialized(Err(InitError::AlreadyInitializing)));
                    return out;
                }
                self.initialized_requested = true;
                let msg = format!("initializing {}", self.opts.runner());
                out.push(Effect::Log(Level::Info, msg, Vec::new()));
                self.send(ToChild::Hello(hello), &mut out);
                match self.reader_closed {
                    true => out.push(Effect::Initialized(Err(InitError::Closed))),
                    false => {
                        self.init = Some((
                            now,
                            self.startup_at
                                .take()
                                .unwrap_or(now + secs(self.opts.initialize_timeout)),
                        ))
                    }
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
                self.send(ToChild::StartJob(job), &mut out);
            }
            Input::Message(msg) => self.on_message(now, wall_ms, msg, &mut out),
            Input::ReaderClosed(reason) => {
                if self.reader_closed {
                    return out;
                }
                let orderly_close = self.closing && self.shutting_down;
                self.reader_closed = true;
                self.next_ping = None;
                self.protocol_failure = reason.clone();
                if let Some(reason) = reason.filter(|_| !self.closing || !self.shutting_down) {
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
                if !self.exited && !orderly_close {
                    self.closing = true;
                    self.begin_kill(&mut out);
                    self.cleanup_at.get_or_insert(now + self.termination_timeout());
                }
            }
            Input::Shutdown => self.shutdown(now, &mut out),
            Input::Kill if !self.opts.killable => self.shutdown(now, &mut out),
            Input::Kill => {
                self.closing = true;
                self.begin_kill(&mut out);
                self.cleanup_at.get_or_insert(now + self.termination_timeout());
            }
            Input::KillBegun(Some(true)) => {
                self.kill_at = Some(now + KILL_GRACE);
                self.cleanup_at.get_or_insert(now + self.termination_timeout());
            }
            Input::KillBegun(Some(false)) => {
                self.finish_kill(&mut out);
                self.cleanup_at.get_or_insert(now + self.termination_timeout());
            }
            Input::KillBegun(None) => {
                self.killing = false;
                self.kill_ended = true;
            }
            Input::Sampled(sample) => self.on_sample(now, sample, &mut out),
            Input::Exited(code) => {
                self.exited = true;
                self.startup_at = None;
                self.cleanup_at.get_or_insert(now + secs(self.opts.close_timeout));
                if self.init.take().is_some() {
                    out.push(Effect::Initialized(Err(InitError::Closed)));
                }
                if self.reaped {
                    self.cleanup_at = None;
                }
                (self.next_ping, self.pong_deadline, self.next_sample, self.kill_at) =
                    (None, None, None, None);
                self.close = Close::None;
                self.acked = true;
                self.shutting_down = true;
                self.observe_exitcode(code, &mut out);
            }
        }
        out
    }

    /// What is due at `now`.
    pub fn tick(&mut self, now: Duration, wall_ms: i64) -> Vec<Effect> {
        let mut out = Vec::new();
        if self.cleaned() {
            return out;
        }
        if self.cleanup_at.is_some_and(|at| at <= now) {
            self.cleanup_at = None;
            out.push(Effect::IncompleteCleanup(CleanupObservation {
                pid: self.opts.pid,
                exited: self.exited,
                reaped: self.reaped,
            }));
        }
        if self.exited {
            return out;
        }
        if self.startup_at.is_some_and(|at| at <= now) {
            self.startup_at = None;
            self.closing = true;
            self.cleanup_at.get_or_insert(now + self.termination_timeout());
            out.push(Effect::Log(
                Level::Error,
                "process launch did not enter initialization before its deadline".into(),
                Vec::new(),
            ));
            self.begin_kill(&mut out);
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
            self.send(ToChild::Ping { sent_ms: wall_ms.max(0) as u64 }, &mut out);
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
                self.close = Close::ShuttingDown(now + secs(self.opts.close_timeout));
                self.advance_close(now);
            }
            Close::ShuttingDown(at) | Close::Exit(at) if at <= now => {
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

    fn on_message(&mut self, now: Duration, wall_ms: i64, msg: ToParent, out: &mut Vec<Effect>) {
        if let Some((started, _)) = self.init.take() {
            let result = match msg {
                ToParent::Ready { version, .. } if version != PROTOCOL_VERSION => {
                    Err(InitError::Version(version))
                }
                ToParent::Ready { error: Some(error), .. } => Err(InitError::Failed(error)),
                ToParent::Ready { error: None, .. } => {
                    let elapsed = now.saturating_sub(started);
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
                let delay = wall_ms.saturating_sub(sent_ms as i64);
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

    fn shutdown(&mut self, now: Duration, out: &mut Vec<Effect>) {
        if self.close != Close::None {
            return;
        }
        self.closing = true;
        self.send(ToChild::Shutdown { reason: String::new() }, out);
        self.cleanup_at = Some(now + secs(self.opts.close_timeout) * 3 + self.kill_grace());
        self.close = Close::Ack(now + secs(self.opts.close_timeout));
        self.advance_close(now);
    }

    fn advance_close(&mut self, now: Duration) {
        if matches!(self.close, Close::Ack(_)) && self.acked {
            self.close = Close::ShuttingDown(now + secs(self.opts.close_timeout));
        }
        if matches!(self.close, Close::ShuttingDown(_)) && self.shutting_down {
            self.close = Close::Exit(now + secs(self.opts.close_timeout));
        }
    }

    fn begin_kill(&mut self, out: &mut Vec<Effect>) {
        if self.opts.killable && !self.killing && !self.killed && !self.exited {
            self.killing = true;
            out.push(Effect::BeginKill);
        }
    }

    fn finish_kill(&mut self, out: &mut Vec<Effect>) {
        out.push(Effect::Kill);
    }

    fn observe_exitcode(&mut self, code: Option<i64>, out: &mut Vec<Effect>) {
        if self.exitcode.is_none()
            && let Some(code) = code
        {
            self.exitcode = Some(code);
            if code != 0 && !self.killed {
                out.push(Effect::Log(
                    Level::Error,
                    format!("process exited with non-zero exit code {code}"),
                    Vec::new(),
                ));
            }
        }
    }

    fn kill_grace(&self) -> Duration {
        if self.opts.killable { KILL_GRACE } else { Duration::ZERO }
    }

    fn termination_timeout(&self) -> Duration {
        self.kill_grace() + secs(self.opts.close_timeout)
    }

    pub fn cleaned(&self) -> bool {
        self.launch_rejected || self.reaped()
    }

    pub fn reaped(&self) -> bool {
        self.exited && self.reaped
    }

    pub(crate) fn close_timeout(&self) -> f64 {
        self.opts.close_timeout
    }

    pub fn pid(&self) -> u32 {
        self.opts.pid
    }

    fn send(&mut self, message: ToChild, out: &mut Vec<Effect>) {
        if self.reader_closed || self.exited {
            return;
        }
        self.next_write += 1;
        let id = self.next_write;
        self.pending_writes.insert(id);
        out.push(Effect::Send { id, message });
    }

    fn on_sample(&mut self, now: Duration, sample: Sample, out: &mut Vec<Effect>) {
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

    fn memory_extra(&self, now: Duration, memory_mb: f64, metric: &str, baseline_mb: f64) -> Extra {
        vec![
            ("memory_usage_mb", ExtraValue::Float(round(memory_mb, 1))),
            ("memory_metric", ExtraValue::Str(metric.to_owned())),
            ("memory_warn_mb", ExtraValue::Float(self.opts.memory_warn_mb)),
            ("memory_limit_mb", ExtraValue::Float(self.opts.memory_limit_mb)),
            ("uptime", ExtraValue::Float(round(now.saturating_sub(self.spawned).as_secs_f64(), 1))),
            ("has_running_job", ExtraValue::Bool(self.running_job)),
            ("baseline_memory_mb", ExtraValue::Float(round(baseline_mb, 1))),
            ("growth_memory_mb", ExtraValue::Float(round(memory_mb - baseline_mb, 1))),
        ]
    }
}

#[cfg(test)]
#[path = "supervisor_tests.rs"]
mod tests;

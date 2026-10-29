//! One job process's IPC connection, deadlines, health sampling and cleanup ownership.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use lk_agents_abi::errors::HostError;

use super::wire::{self, Hello, PROTOCOL_VERSION, StartJob, ToChild, ToParent, WireError};

mod timeline;
mod types;
pub use timeline::JobTimeline;
pub use types::*;
use types::{Close, WritePurpose, round};

/// One process's supervision, sans I/O.
pub struct Supervision {
    opts: SupervisorOptions,
    spawned: Duration,
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
    reader_deadline: Option<Duration>,
    reaped: bool,
    launch_rejected: bool,
    exitcode: Option<i64>,
    cleanup_at: Option<Duration>,
    next_write: u64,
    pending_writes: BTreeMap<u64, WritePurpose>,
    protocol_failure: Option<Arc<WireError>>,
}

impl Supervision {
    pub fn new(opts: SupervisorOptions, now: Duration) -> Self {
        Self {
            opts,
            spawned: now,
            init: None,
            initialized_requested: false,
            initialized: false,
            pending_job: None,
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
            reader_deadline: None,
            reaped: false,
            launch_rejected: false,
            exitcode: None,
            cleanup_at: None,
            next_write: 0,
            pending_writes: BTreeMap::new(),
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
            return [self.cleanup_at, self.reader_deadline].into_iter().flatten().min();
        }
        let close = match self.close {
            Close::Ack(at) | Close::ShuttingDown(at) | Close::Exit(at) => Some(at),
            _ => None,
        };
        let sample = self.next_sample.filter(|_| !self.sampling && !self.stopping());
        [
            self.startup_at,
            self.init.map(|(_, at)| at),
            self.pending_job.map(|(_, at)| at),
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
        if self.exited
            && !matches!(
                input,
                Input::Reaped(_)
                    | Input::LaunchFailed(_)
                    | Input::ReaderClosed(_)
                    | Input::Message(_)
            )
        {
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
                if let (Some(purpose), Err(reason)) = (self.pending_writes.remove(&id), result) {
                    let expected_ping_close = purpose == WritePurpose::Ping
                        && self.shutting_down
                        && matches!(reason.as_ref(), WireError::Io { kind, .. } if matches!(kind,
                            std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                            | std::io::ErrorKind::ConnectionAborted | std::io::ErrorKind::NotConnected));
                    if !expected_ping_close {
                        out.extend(self.handle(now, wall_ms, Input::ReaderClosed(Some(reason))));
                    }
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
            Input::StartJob { request_id, job } => {
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
                if let Some(error) = failure {
                    out.push(Effect::JobStarted { request_id, result: Err(error) });
                } else {
                    self.pending_job = Some((request_id, now + secs(self.opts.initialize_timeout)));
                    self.send(ToChild::StartJob { request_id, job }, &mut out);
                }
            }
            Input::Message(msg) => self.on_message(now, wall_ms, msg, &mut out),
            Input::ReaderClosed(reason) => {
                let reason = reason.unwrap_or_else(|| Arc::new(WireError::Eof));
                if self.shutting_down && matches!(*reason, WireError::Eof) {
                    self.reader_closed = true;
                    self.reader_deadline = None;
                    self.next_ping = None;
                    self.acked = true;
                    self.advance_close(now);
                } else {
                    self.fail_protocol(now, reason, &mut out);
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
                if self.initialized_requested && !self.reader_closed {
                    self.reader_deadline = Some(now + secs(self.opts.close_timeout));
                }
                self.startup_at = None;
                self.cleanup_at.get_or_insert(now + secs(self.opts.close_timeout));
                if self.reader_closed && self.init.take().is_some() {
                    out.push(Effect::Initialized(Err(InitError::Closed)));
                }
                if self.reaped {
                    self.cleanup_at = None;
                }
                (self.next_ping, self.pong_deadline, self.next_sample, self.kill_at) =
                    (None, None, None, None);
                self.close = Close::None;
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
        if self.reader_deadline.is_some_and(|at| at <= now) {
            self.reader_deadline = None;
            self.fail_protocol(
                now,
                Arc::new(WireError::Io {
                    kind: std::io::ErrorKind::TimedOut,
                    os_error: None,
                    message: "IPC reader did not settle after process exit".into(),
                }),
                &mut out,
            );
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
        if let Some((request_id, deadline)) = self.pending_job
            && deadline <= now
        {
            self.pending_job = None;
            out.push(Effect::JobStarted { request_id, result: Err(StartJobError::Timeout) });
            self.closing = true;
            self.begin_kill(&mut out);
            self.cleanup_at.get_or_insert(now + self.termination_timeout());
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
        if self.exited {
            match &msg {
                ToParent::Ready { version, error }
                    if *version != PROTOCOL_VERSION || error.is_some() => {}
                ToParent::JobAccepted { error: Some(_), .. }
                | ToParent::ShuttingDown
                | ToParent::Exiting { .. } => {}
                _ => return,
            }
        }
        if let Some((started, _)) = self.init.take() {
            let result = match msg {
                ToParent::Ready { version, .. } if version != PROTOCOL_VERSION => {
                    Err(InitError::Version(version))
                }
                ToParent::Ready { error: Some(error), .. } => Err(InitError::Failed(error)),
                ToParent::Ready { error: None, .. } => {
                    self.initialized = true;
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
            let refusal = match &result {
                Err(InitError::Version(theirs)) => Some(WireError::Version { theirs: *theirs }),
                Err(error) => Some(WireError::Unexpected(error.to_string())),
                Ok(_) => None,
            };
            out.push(Effect::Initialized(result));
            if let Some(refusal) = refusal {
                self.fail_protocol(now, Arc::new(refusal), out);
            }
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
                self.next_ping = None;
                self.pong_deadline = None;
                if self.close == Close::None {
                    self.closing = true;
                    self.close = Close::Exit(now + secs(self.opts.close_timeout));
                    self.cleanup_at.get_or_insert(now + self.termination_timeout());
                }
                self.advance_close(now);
            }
            ToParent::Exiting { reason } => {
                let extra = vec![("reason", ExtraValue::Str(reason))];
                match self.opts.killable {
                    true => out.push(Effect::Log(Level::Info, "process exiting".into(), extra)),
                    false => out.push(Effect::Log(Level::Debug, "job exiting".into(), extra)),
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
                    out.push(Effect::JobStarted { request_id, result });
                    if refused {
                        self.closing = true;
                        self.begin_kill(out);
                        self.cleanup_at.get_or_insert(now + self.termination_timeout());
                    }
                } else {
                    self.fail_protocol(
                        now,
                        Arc::new(WireError::Unexpected(format!(
                            "job acknowledgment {request_id} has no pending submission"
                        ))),
                        out,
                    );
                }
            }
            ToParent::Ready { .. } => self.fail_protocol(
                now,
                Arc::new(WireError::Unexpected("unsolicited Ready".into())),
                out,
            ),
        }
    }

    /// The first protocol failure remains authoritative even when OS exit reports zero.
    pub fn protocol_failure(&self) -> Option<&WireError> {
        self.protocol_failure.as_deref()
    }

    fn fail_protocol(&mut self, now: Duration, failure: Arc<WireError>, out: &mut Vec<Effect>) {
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
        out.push(Effect::Log(Level::Error, format!("invalid IPC message: {failure}"), Vec::new()));
        if self.init.take().is_some() {
            out.push(Effect::Initialized(Err(InitError::Protocol(failure.clone()))));
        }
        if let Some((request_id, _)) = self.pending_job.take() {
            out.push(Effect::JobStarted {
                request_id,
                result: Err(StartJobError::Protocol(failure)),
            });
        }
        self.acked = true;
        self.shutting_down = true;
        self.advance_close(now);
        self.begin_kill(out);
        if !self.reaped() {
            self.cleanup_at.get_or_insert(now + self.termination_timeout());
        }
    }

    fn shutdown(&mut self, now: Duration, out: &mut Vec<Effect>) {
        if self.close != Close::None {
            return;
        }
        self.closing = true;
        if let Some((request_id, _)) = self.pending_job.take() {
            out.push(Effect::JobStarted { request_id, result: Err(StartJobError::Closed) });
        }
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
        self.launch_rejected
            || (self.reaped() && (!self.initialized_requested || self.reader_closed))
    }

    pub fn reaped(&self) -> bool {
        self.exited && self.reaped
    }

    pub fn close_timeout(&self) -> f64 {
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
        let purpose = if matches!(message, ToChild::Ping { .. }) {
            WritePurpose::Ping
        } else {
            WritePurpose::Protocol
        };
        self.pending_writes.insert(id, purpose);
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

/// What the supervisor needs from the host language: logging and the process handle.
pub trait ProcHost: Send + Sync + 'static {
    fn log(&self, level: Level, msg: &str, extra: Extra);
    /// An error log with the failure's traceback.
    fn log_exception(&self, msg: &str, err: HostError);
    fn is_alive(&self) -> bool;
    /// SIGKILL (`Process.kill`; a terminate on Windows).
    fn kill(&self) -> Result<(), String>;
    fn request_stack_dump(&self) -> Result<bool, String> {
        Ok(false)
    }
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

pub fn time_ms() -> i64 {
    (lk_agents_abi::unix_now() * 1000.0 + 0.5) as i64
}

enum Command {
    Input(Input),
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
}

impl ProcSupervisor {
    pub fn new<S: AsyncRead + AsyncWrite + Send + 'static, H: ProcHost>(
        stream: S,
        host: H,
        opts: SupervisorOptions,
        workers: crate::io::WorkerIo,
    ) -> Self {
        let (mut reader, mut writer) = tokio::io::split(stream);
        let (messages_tx, messages) = mpsc::channel(QUEUE);
        let reading = tokio::spawn(async move {
            loop {
                let read = wire::read::<ToParent, _>(&mut reader).await;
                let failed = read.is_err();
                if messages_tx.send(read.map_err(Arc::new)).await.is_err() || failed {
                    break;
                }
            }
        });
        let (writes, mut queued) = mpsc::channel::<(u64, ToChild)>(QUEUE);
        let (written_tx, written) = mpsc::channel(QUEUE);
        let writing = tokio::spawn(async move {
            while let Some((id, message)) = queued.recv().await {
                let result = wire::write(&mut writer, &message)
                    .await
                    .map_err(|error| Arc::new(wire::io_failure(error)));
                let failed = result.is_err();
                if written_tx.send((id, result)).await.is_err() || failed {
                    break;
                }
            }
        });
        let (commands, commands_rx) = mpsc::channel(QUEUE);
        let (control, control_rx) = watch::channel(Control::default());
        let (status_tx, status) = watch::channel(Status::default());
        let driver = Driver {
            state: Supervision::new(opts, Duration::ZERO),
            origin: Instant::now(),
            host: Arc::new(host),
            workers,
            writes,
            init: None,
            job: None,
            next_job_id: 0,
            status: status_tx,
        };
        tokio::spawn(driver.run(commands_rx, control_rx, messages, written, reading, writing));
        Self { commands, control, status }
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
        let _ = self.commands.send(Command::Input(Input::Supervise)).await;
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

struct Driver {
    state: Supervision,
    origin: Instant,
    host: Arc<dyn ProcHost>,
    workers: crate::io::WorkerIo,
    writes: mpsc::Sender<(u64, ToChild)>,
    init: Option<oneshot::Sender<Result<Option<Duration>, InitError>>>,
    job: Option<(u64, oneshot::Sender<Result<(), StartJobError>>)>,
    next_job_id: u64,
    status: watch::Sender<Status>,
}

type Sampling = Option<crate::io::WorkerIoTask<Result<(f64, String), MemoryError>>>;

impl Driver {
    #[allow(clippy::too_many_arguments)]
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        mut control: watch::Receiver<Control>,
        mut messages: mpsc::Receiver<Result<ToParent, Arc<WireError>>>,
        mut written: mpsc::Receiver<(u64, Result<(), Arc<WireError>>)>,
        reading: tokio::task::JoinHandle<()>,
        writing: tokio::task::JoinHandle<()>,
    ) {
        let mut reader_open = true;
        let mut commands_open = true;
        let mut control_open = true;
        let mut writer_open = true;
        let mut sampling: Sampling = None;
        let mut seen = Control::default();
        while !self.state.cleaned() {
            let now = Instant::now();
            if self.state.wake_at().is_some_and(|at| at <= now.duration_since(self.origin)) {
                let effects = self.state.tick(now.duration_since(self.origin), time_ms());
                self.execute(effects, &mut sampling);
                self.publish();
                continue;
            }
            let deadline = self.state.wake_at().map(|at| self.origin + at);
            let input = tokio::select! {
                biased;
                () = async { tokio::time::sleep_until(deadline.unwrap()).await }, if deadline.is_some() => continue,
                changed = control.changed(), if control_open => {
                    if changed.is_err() { control_open = false; Input::Kill }
                    else {
                        let next = control.borrow_and_update().clone();
                        let mut inputs = Vec::new();
                        if next.pid != seen.pid && let Some(pid) = next.pid { inputs.push(Input::Launched(pid)); }
                        if next.exited != seen.exited && let Some(code) = next.exited { inputs.push(Input::Exited(code)); }
                        if next.reaped != seen.reaped && let Some(code) = next.reaped { inputs.push(Input::Reaped(code)); }
                        if next.launch_failed != seen.launch_failed && let Some(cause) = &next.launch_failed { inputs.push(Input::LaunchFailed(cause.clone())); }
                        if next.launch_rejected != seen.launch_rejected && let Some(cause) = &next.launch_rejected { inputs.push(Input::LaunchRejected(cause.clone())); }
                        if next.kill && !seen.kill { inputs.push(Input::Kill); }
                        if next.shutdown && !seen.shutdown { inputs.push(Input::Shutdown); }
                        seen = next;
                        for input in inputs { self.apply(input, &mut sampling); }
                        self.publish();
                        continue;
                    }
                },
                msg = messages.recv(), if reader_open => match msg {
                    Some(Ok(msg)) => Input::Message(msg),
                    Some(Err(reason)) => { reader_open = false; Input::ReaderClosed(Some(reason)) },
                    None => { reader_open = false; Input::ReaderClosed(None) }
                },
                result = written.recv(), if writer_open => match result {
                    Some((id, result)) => {
                        // A failed write is the writer's final correlated outcome.
                        writer_open = result.is_ok();
                        Input::Written { id, result }
                    },
                    None => { writer_open = false; Input::ReaderClosed(Some(Arc::new(WireError::Unexpected("IPC writer closed".into())))) }
                },
                command = commands.recv(), if commands_open => match command {
                    Some(Command::Input(input)) => input,
                    Some(Command::Initialize(hello, reply)) => {
                        if self.state.stopping() || self.state.exited() {
                            let _ = reply.send(Err(InitError::Closed));
                            continue;
                        }
                        if self.init.is_some() {
                            let _ = reply.send(Err(InitError::AlreadyInitializing));
                            continue;
                        }
                        self.init = Some(reply);
                        Input::Initialize(hello)
                    }
                    Some(Command::StartJob(job, reply)) => {
                        if self.job.is_some() {
                            let _ = reply.send(Err(StartJobError::AlreadyRunning));
                            continue;
                        }
                        self.next_job_id += 1;
                        let request_id = self.next_job_id;
                        self.job = Some((request_id, reply));
                        Input::StartJob { request_id, job: Box::new(job) }
                    }
                    None => { commands_open = false; Input::Kill }
                },
                sample = async { sampling.as_mut().unwrap().await }, if sampling.is_some() => {
                    sampling = None;
                    Input::Sampled(match sample {
                        Ok(Ok((mb, metric))) => Sample::Mb(mb, metric),
                        Ok(Err(MemoryError::Unavailable(e))) => Sample::Unavailable(e),
                        Ok(Err(MemoryError::Failed(e))) => {
                            if !self.state.stopping() { self.host.log_exception("Error in memory monitoring task", e); }
                            Sample::Failed
                        }
                        Ok(Err(MemoryError::Interrupted)) | Err(_) => Sample::Interrupted,
                    })
                }
            };
            self.apply(input, &mut sampling);
            self.publish();
        }
        reading.abort();
        writing.abort();
        let _ = reading.await;
        let _ = writing.await;
        // Started blocking samples cannot be aborted: retain the host until they settle.
        if let Some(mut sample) = sampling {
            if tokio::time::timeout(secs(self.state.close_timeout()), &mut sample).await.is_err() {
                let observation =
                    CleanupObservation { pid: self.state.pid(), exited: true, reaped: true };
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
        if let Some(init) = self.init.take() {
            let _ = init.send(Err(InitError::Closed));
        }
        if let Some((_, job)) = self.job.take() {
            let _ = job.send(Err(StartJobError::Closed));
        }
        self.host.release();
        self.status.send_modify(|status| {
            status.reaped = self.state.reaped();
            status.cleaned = true;
        });
    }

    fn apply(&mut self, input: Input, sampling: &mut Sampling) {
        let effects =
            self.state.handle(Instant::now().duration_since(self.origin), time_ms(), input);
        self.execute(effects, sampling);
    }

    fn publish(&self) {
        self.status.send_modify(|s| {
            s.failure = self.state.protocol_failure().map(Into::into);
            s.killed = self.state.killed();
            s.kill_settled = self.state.kill_settled();
            s.exited = self.state.exited();
        });
    }

    fn execute(&mut self, effects: Vec<Effect>, sampling: &mut Sampling) {
        let mut queue = std::collections::VecDeque::from(effects);
        while let Some(effect) = queue.pop_front() {
            match effect {
                Effect::Send { id, message } => {
                    if let Err(error) = self.writes.try_send((id, message)) {
                        queue.extend(self.state.handle(
                            Instant::now().duration_since(self.origin),
                            time_ms(),
                            Input::Written {
                                id,
                                result: Err(Arc::new(WireError::Unexpected(format!(
                                    "IPC write submission failed: {error}"
                                )))),
                            },
                        ));
                    }
                }
                Effect::Log(level, msg, extra) => self.host.log(level, &msg, extra),
                Effect::BeginKill => {
                    let begun = self.begin_kill();
                    queue.extend(self.state.handle(
                        Instant::now().duration_since(self.origin),
                        time_ms(),
                        Input::KillBegun(begun),
                    ));
                }
                Effect::Kill => {
                    let result = if self.host.is_alive() { self.host.kill() } else { Ok(()) };
                    queue.extend(self.state.handle(
                        Instant::now().duration_since(self.origin),
                        time_ms(),
                        Input::KillFinished(result),
                    ));
                }
                Effect::SignalFailed(failure) => {
                    self.host.log(
                        Level::Error,
                        &format!("process termination failed: {}", failure.cause),
                        Vec::new(),
                    );
                    self.host.signal_failed(failure);
                }
                Effect::SampleMemory => {
                    let host = self.host.clone();
                    // Try admission before creating any task; a saturated sampler retries at its next cadence.
                    match self.workers.try_spawn(move || host.sample_memory()) {
                        Ok(receipt) => *sampling = Some(receipt),
                        Err(_) => queue.extend(self.state.handle(
                            Instant::now().duration_since(self.origin),
                            time_ms(),
                            Input::Sampled(Sample::Failed),
                        )),
                    }
                }
                Effect::Initialized(result) => {
                    if let Some(init) = self.init.take() {
                        let _ = init.send(result);
                    }
                }
                Effect::JobStarted { request_id, result } => {
                    if self.job.as_ref().is_some_and(|(pending, _)| *pending == request_id)
                        && let Some((_, reply)) = self.job.take()
                    {
                        let _ = reply.send(result);
                    }
                }
                Effect::IncompleteCleanup(observation) => {
                    self.host.log(
                        Level::Error,
                        "incomplete process cleanup; retaining ownership until reaping",
                        Vec::new(),
                    );
                    self.host.incomplete_cleanup(observation);
                    self.status.send_modify(|s| s.incomplete = true);
                }
            }
        }
    }

    fn begin_kill(&mut self) -> Option<bool> {
        if !self.host.is_alive() {
            return None;
        }
        self.host.log(Level::Info, "killing process", Vec::new());
        match self.host.request_stack_dump() {
            Ok(signalled) => Some(signalled),
            Err(cause) => {
                self.host.signal_failed(SignalFailure { pid: self.state.pid(), cause });
                Some(false)
            }
        }
    }
}

#[cfg(test)]
mod tests;

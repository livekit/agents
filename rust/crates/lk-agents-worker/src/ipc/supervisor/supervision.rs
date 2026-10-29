//! One process's supervision, sans I/O: messages, deadlines and samples in, effects out.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::super::wire::{PROTOCOL_VERSION, ToChild, ToParent, WireError};
use super::types::*;
use super::types::{Close, WritePurpose, round};

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

#[cfg(test)]
mod tests;

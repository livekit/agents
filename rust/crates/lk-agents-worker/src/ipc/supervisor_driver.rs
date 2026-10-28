//! Socket, timer and host execution for the synchronous process supervisor.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::supervisor::*;
use super::wire::{self, Hello, StartJob, ToChild, ToParent, WireError};

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
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    pub fn new<S: AsyncRead + AsyncWrite + Send + 'static, H: ProcHost>(
        stream: S,
        host: H,
        opts: SupervisorOptions,
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
                    .map_err(|error| Arc::new(WireError::Io(error)));
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

struct Driver<H: ProcHost> {
    state: Supervision,
    origin: Instant,
    host: Arc<H>,
    writes: mpsc::Sender<(u64, ToChild)>,
    init: Option<oneshot::Sender<Result<Option<Duration>, InitError>>>,
    job: Option<(u64, oneshot::Sender<Result<(), StartJobError>>)>,
    next_job_id: u64,
    status: watch::Sender<Status>,
}

type Sampling<E> = Option<tokio::task::JoinHandle<Result<(f64, String), MemoryError<E>>>>;

impl<H: ProcHost> Driver<H> {
    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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
        let mut sampling: Sampling<H::Error> = None;
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
                    Some((id, result)) => Input::Written { id, result },
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

    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    fn apply(&mut self, input: Input, sampling: &mut Sampling<H::Error>) {
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    fn execute(&mut self, effects: Vec<Effect>, sampling: &mut Sampling<H::Error>) {
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
                    match crate::worker::WORKER_IO_SLOTS.try_acquire() {
                        Ok(permit) => {
                            *sampling = Some(tokio::task::spawn_blocking(move || {
                                let _permit = permit;
                                host.sample_memory()
                            }))
                        }
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

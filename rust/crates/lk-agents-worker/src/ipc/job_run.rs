//! A job's run inside its process (`_core.JobRun`): what ends the job (the room, the user, the
//! parent, a crash, the simulator leaving; the first wins), the entrypoint's supervision, the
//! order and bounds of the shutdown (whose steps the host runs and opens the spans of), the
//! session report's outputs, the participant entrypoints, the room connected once, the
//! application calls in flight and the recording's setup.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use super::telemetry;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{
    JobCall, JobExecution, JobProcess, ParticipantEntrypoint, WorkerError,
};
use lk_agents_abi::worker::job::StartJobFields;
use lk_agents_abi::{ATTRIBUTE_SIMULATOR, HostRef};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::{TaskSet, TokioSpawner};
use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};

use super::ipc_channel::Channel;

use super::wire::{ToChild, bounded};
use super::{Extra, ExtraValue, Level, ToParent, extra_fields, secs};
use lk_agents_net::env::ProcessEnv;

/// How long the entrypoint may take to return once the job shuts down, before it is cancelled.
const ENTRYPOINT_EXIT_TIMEOUT: Duration = Duration::from_secs(15);

/// The entrypoint has this long to connect the room before a warning.
const CONNECT_WARNING_DELAY: Duration = Duration::from_secs(10);

#[cfg(test)]
#[path = "job_run_tests.rs"]
mod tests;

#[derive(Default)]
struct State {
    /// The first reason the job ends with.
    shutdown: Option<String>,
    /// What ended the job: the first cause, unless the application ended it too.
    cause: Option<telemetry::ShutdownCause>,
    connected: bool,
    /// `add_participant_entrypoint`'s, in order, each with the participant kinds it runs for.
    entrypoints: Vec<Entrypoint>,
    /// The participant entrypoints running, by participant and entrypoint, each its run's number.
    participant_runs: HashMap<(String, usize), u64>,
    runs: u64,
    /// The application calls (`delete_room`, ...) still running.
    tasks: Vec<OwnedTask>,
    /// Whether `init_recording` ran, and the redaction it resolved (the job's until then).
    recording_initialized: bool,
    redaction: bool,
    /// What the session records, from `init_recording`.
    recording: Option<telemetry::RecordingOptions>,
    /// `ctx.add_shutdown_callback`'s, in order.
    shutdown_callbacks: Vec<ShutdownCallback>,
}

struct Entrypoint {
    run: Arc<dyn ParticipantEntrypoint>,
    kinds: Vec<i32>,
    name: String,
}

struct ShutdownCallback {
    callback: HostRef,
    takes_reason: bool,
    /// Registered by livekit-agents: it runs in the shutdown's span, not one of its own.
    framework: bool,
    name: String,
}

pub struct JobRun {
    inner: Arc<Inner>,
}

struct Inner {
    /// The runner and the context its steps run in, from `run()`.
    host: OnceLock<(Arc<dyn JobProcess>, CallContext)>,
    /// The parent the job process reports its end to (none for a job run outside one).
    parent: OnceLock<Arc<Channel<ToParent, ToChild>>>,
    spawner: TokioSpawner,
    fake: bool,
    session_end_timeout: f64,
    /// The name of the door's `on_session_end`, when it has one.
    session_end: Option<String>,
    /// The bound on `VoiceSession.aclose()` during the shutdown: past it, the rest of the
    /// shutdown (on_session_end, ShuttingDown, room.disconnect, the callbacks) runs anyway.
    session_aclose_timeout: f64,
    /// The job's LiveKit Cloud observability endpoint, when it has one.
    observability_url: Option<String>,
    state: Mutex<State>,
    ended: watch::Sender<bool>,
    /// Held while the room connects, so a second `connect()` waits for the first.
    connecting: tokio::sync::Mutex<()>,
    /// Set once the job ends: the application calls still running are cancelled.
    tasks_closed: watch::Sender<bool>,
    /// The participant entrypoints, which run to their own end.
    participants: Arc<TaskSet>,
    tracer: telemetry::Tracer,
    /// Times the session report's outputs.
    clock: Arc<dyn lk_agents_abi::Clock>,
    /// Where the job's Cloud telemetry goes, signed with the server's credentials.
    cloud: Option<telemetry::CloudEndpoint>,
    /// The job's root span, from its start until its shutdown ends.
    root: Mutex<Option<telemetry::ScopedSpan<telemetry::JobEntrypoint>>>,
    /// The job the parent handed over; none for a job run outside one.
    job: OnceLock<telemetry::Job>,
}

impl JobExecution for JobRun {
    fn session_closed(&self, reason: String) {
        self.inner.end(reason, telemetry::ShutdownCause::SessionClosed);
    }
}

impl JobRun {
    /// The job ends with `reason`; the first reason given is the one kept.
    pub(crate) fn end(&self, reason: String, cause: telemetry::ShutdownCause) {
        self.inner.end(reason, cause);
    }

    pub(crate) fn report_to(&self, parent: Arc<Channel<ToParent, ToChild>>) {
        let _ = self.inner.parent.set(parent);
    }
}

fn cancelled(error: &HostError) -> bool {
    error.kind == "CancelledError"
}

impl Inner {
    fn end(&self, reason: String, cause: telemetry::ShutdownCause) {
        let mut st = self.state.lock();
        // The session's close can reach the job before the application's own `shutdown`.
        if st.cause.is_none() || cause == telemetry::ShutdownCause::Application {
            st.cause = Some(cause);
        }
        if st.shutdown.is_none() {
            st.shutdown = Some(reason);
            self.ended.send_replace(true);
        }
    }

    fn host(&self) -> Option<&Arc<dyn JobProcess>> {
        self.host.get().map(|(host, _)| host)
    }

    /// `message` on the runner's logger, with the failure's traceback and `extra`.
    fn log(&self, level: Level, message: &str, error: Option<HostError>, extra: Extra) {
        let Some(host) = self.host() else {
            log::log!(target: crate::LOG, log_level(level), "{message}");
            return;
        };
        if let Err(failure) = host.log(level as i32, message.to_owned(), extra_fields(extra), error)
        {
            log::warn!(target: crate::LOG, "{message} (not logged by the host: {failure})");
        }
    }

    /// One shutdown step in its span, bounded by `timeout`.
    async fn step<D: telemetry::SpanDecl<End = ()>>(
        &self,
        span: telemetry::ScopedSpan<D>,
        run: impl Future<Output = Result<(), HostError>>,
        timeout: Option<f64>,
        error_msg: &str,
    ) where
        (): telemetry::SpanEnd<D>,
    {
        let outcome = match timeout {
            Some(t) => tokio::time::timeout(secs(t), run).await.map_err(|_| ()),
            None => Ok(run.await),
        };
        match outcome {
            Ok(ran) => {
                telemetry::end(span, ran.as_ref().map(|_| ()));
                if let Err(error) = ran {
                    self.log(Level::Error, error_msg, Some(error), Vec::new());
                }
            }
            Err(()) => {
                telemetry::end(span, lk_agents_abi::Elapsed);
                let msg =
                    format!("{} timed out after {}s", D::NAME, timeout.unwrap_or_default() as i64);
                self.log(Level::Error, &msg, None, Vec::new());
            }
        }
    }

    /// The session report's outputs: the host's report after `on_session_end`, its records with
    /// the job's flush, then the recording upload. Nothing without a session or its recording.
    async fn session_end_upload(
        &self,
        host: &Arc<dyn JobProcess>,
        cx: &CallContext,
        span: &telemetry::Cx,
    ) -> Result<(), HostError> {
        let Some(report) = host.session_report(span, cx).await? else { return Ok(()) };
        let (Some(options), Some(job)) = (self.state.lock().recording, self.job.get()) else {
            return Ok(());
        };
        let exporter = self.tracer.exporter();
        let outputs = telemetry::ReportOutputs {
            exporter: exporter.as_deref(),
            cloud: self.cloud.as_ref().filter(|_| self.observability_url.is_some()),
            clock: &*self.clock,
        };
        let simulated = job.simulation.is_some();
        let limit = Duration::from_secs(telemetry::UPLOAD_TIMEOUT_SECONDS);
        telemetry::session_report::export(outputs, &job.key, &report, &options, simulated, limit)
            .await
            .map_err(|error| HostError::new(error.to_string()))
    }

    async fn run(self: Arc<Self>) -> Result<(), WorkerError> {
        // A job run outside a parent's handover has a root span of its own and no exporter job.
        let root = self.root.lock().take();
        let span = root.unwrap_or_else(|| {
            telemetry::start(&self.tracer.root(), telemetry::JobEntrypoint::default())
        });
        let result = self.clone().run_traced(&span).await;
        let failure = match &result {
            Ok(crash) => crash.clone(),
            Err(error) => Some(HostError::new(error.to_string())),
        };
        if let Some(error) = &failure {
            telemetry::end(span, error);
        } else {
            telemetry::end(span, ());
        }
        if let (Some(exporter), Some(job)) = (self.tracer.exporter(), self.job.get()) {
            exporter.close_job(&job.key, secs(telemetry::SHUTDOWN_TIMEOUT_SECONDS)).await;
        }
        result.map(drop)
    }

    /// Runs the job to its shutdown; `Ok` carries the entrypoint's unhandled exception, which its
    /// root span ends with.
    async fn run_traced(
        self: Arc<Self>,
        root: &telemetry::ScopedSpan<telemetry::JobEntrypoint>,
    ) -> Result<Option<HostError>, WorkerError> {
        let (host, cx) = self.host.get().expect("run() sets the host first");
        let job = root.cx();
        let entry = host.start_entrypoint(job, cx).await?;
        let wait_entry = host.wait(job, cx, entry.clone());
        tokio::pin!(wait_entry);
        let warn_at = tokio::time::sleep(CONNECT_WARNING_DELAY);
        tokio::pin!(warn_at);
        let mut ended = self.ended.subscribe();
        let (mut entry_done, mut warned) = (false, self.fake);
        let mut crash = None;
        loop {
            tokio::select! {
                r = &mut wait_entry, if !entry_done => {
                    entry_done = true;
                    crash = self.entrypoint_settled(root, r);
                }
                () = &mut warn_at, if !entry_done && !warned => {
                    warned = true;
                    self.warn_if_unconnected();
                }
                _ = ended.wait_for(|e| *e) => break,
            }
        }
        let (reason, cause) = {
            let st = self.state.lock();
            let cause = st.cause.unwrap_or(telemetry::ShutdownCause::ProcessShutdown);
            (st.shutdown.clone().unwrap_or_default(), cause)
        };
        let shutdown =
            telemetry::start(job, telemetry::JobShutdown { reason: reason.clone(), cause });

        // The entrypoint returns, or is cancelled.
        if !entry_done
            && tokio::time::timeout(ENTRYPOINT_EXIT_TIMEOUT, &mut wait_entry).await.is_err()
        {
            self.log(
                Level::Warning,
                "entrypoint did not exit in time, cancelling",
                None,
                Vec::new(),
            );
            let _ = host.cancel(shutdown.cx(), cx, entry).await;
            telemetry::event(root, telemetry::EntrypointCancelled);
        }
        self.close_session(host, cx, shutdown.cx()).await;
        self.run_session_end(host, cx, shutdown.cx()).await;
        self.announce_exit(&reason, cause).await;
        let disconnect = telemetry::start(shutdown.cx(), telemetry::RoomDisconnect);
        let disconnected = host.disconnect(disconnect.cx(), cx).await;
        telemetry::end(disconnect, disconnected.as_ref());
        disconnected?;
        self.run_shutdown_callbacks(host, cx, shutdown.cx(), &reason).await;
        telemetry::end(shutdown, ());
        Ok(crash)
    }

    /// The entrypoint returned or raised: its unhandled exception, which ends the job.
    fn entrypoint_settled(
        &self,
        root: &telemetry::ScopedSpan<telemetry::JobEntrypoint>,
        settled: Result<(), HostError>,
    ) -> Option<HostError> {
        match settled {
            Err(e) if cancelled(&e) => {
                telemetry::event(root, telemetry::EntrypointCancelled);
                None
            }
            Err(e) => {
                self.log(
                    Level::Error,
                    "unhandled exception while running the job task",
                    Some(e.clone()),
                    Vec::new(),
                );
                // A crash before ctx.connect() resolves nothing else: the shutdown
                // still runs, so the cleanup can flush the crash logs.
                self.end("job crashed".into(), telemetry::ShutdownCause::EntrypointCrashed);
                Some(e)
            }
            Ok(()) => {
                telemetry::event(root, telemetry::EntrypointReturned);
                let unfinished = {
                    let st = self.state.lock();
                    !st.connected
                        && st.cause != Some(telemetry::ShutdownCause::Application)
                        && !self.fake
                };
                if unfinished {
                    self.log(Level::Warning, "The job task completed without establishing a connection or performing a proper shutdown. Ensure that job_ctx.connect()/job_ctx.shutdown() is called and the job is correctly finalized.", None, Vec::new());
                }
                None
            }
        }
    }

    /// The entrypoint is still running past [`CONNECT_WARNING_DELAY`]: a warning unless the room
    /// connected or the application ended the job.
    fn warn_if_unconnected(&self) {
        let unconnected = {
            let st = self.state.lock();
            !st.connected && st.cause != Some(telemetry::ShutdownCause::Application)
        };
        if unconnected {
            self.log(Level::Warning, "The room connection was not established within 10 seconds after calling job_entry. This might mean that job_ctx.connect() was never invoked, or that no VoiceSession with an active RoomIO has been started.", None, Vec::new());
        }
    }

    /// `VoiceSession.aclose()` within its bound; the shutdown goes on whatever it does.
    async fn close_session(
        &self,
        host: &Arc<dyn JobProcess>,
        cx: &CallContext,
        under: &telemetry::Cx,
    ) {
        let aclose_timeout = self.session_aclose_timeout;
        match tokio::time::timeout(secs(aclose_timeout), host.close_session(under, cx)).await {
            Ok(Err(error)) => {
                self.log(Level::Error, "error while closing the session", Some(error), Vec::new())
            }
            Ok(Ok(())) => {}
            Err(_) => self.log(
                Level::Error,
                &format!(
                    "VoiceSession.aclose() timed out after {aclose_timeout:.1}s; proceeding with shutdown so registered callbacks still run."
                ),
                None,
                Vec::new(),
            ),
        }
    }

    /// The door's `on_session_end`, then the session report's upload, each in its span.
    async fn run_session_end(
        &self,
        host: &Arc<dyn JobProcess>,
        cx: &CallContext,
        under: &telemetry::Cx,
    ) {
        if let Some(session_end) = &self.session_end {
            let span =
                telemetry::start(under, telemetry::OnSessionEnd { callback: session_end.clone() });
            let under = span.cx().clone();
            self.step(
                span,
                host.session_end(&under, cx),
                Some(self.session_end_timeout),
                "error while executing the on_session_end callback",
            )
            .await;
        }
        let span = telemetry::start(under, telemetry::SessionEndUpload);
        let under = span.cx().clone();
        self.step(
            span,
            self.session_end_upload(host, cx, &under),
            Some(telemetry::UPLOAD_TIMEOUT_SECONDS as f64),
            "failed to upload the session report",
        )
        .await;
    }

    /// The parent learns the job is ending, then why.
    async fn announce_exit(&self, reason: &str, cause: telemetry::ShutdownCause) {
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::ShuttingDown).await;
        }
        let extra = vec![
            ("reason", ExtraValue::Str(reason.to_owned())),
            ("user_initiated", ExtraValue::Bool(cause == telemetry::ShutdownCause::Application)),
        ];
        self.log(Level::Debug, "shutting down job task", None, extra);
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::Exiting { reason: bounded(reason.to_owned()) }).await;
        }
    }

    /// The shutdown callbacks run together: all start before any is awaited.
    async fn run_shutdown_callbacks(
        &self,
        host: &Arc<dyn JobProcess>,
        cx: &CallContext,
        under: &telemetry::Cx,
        reason: &str,
    ) {
        let callbacks = std::mem::take(&mut self.state.lock().shutdown_callbacks);
        let spawner = TokioSpawner::new(self.spawner.handle().clone());
        let runs: Vec<_> = callbacks
            .into_iter()
            .map(|callback| {
                let reason = callback.takes_reason.then(|| reason.to_owned());
                let run =
                    shutdown_callback(host.clone(), cx.clone(), under.clone(), callback, reason);
                spawner.spawn(run)
            })
            .collect();
        for run in runs {
            if let Ok(Err(error)) = run.await {
                self.log(
                    Level::Error,
                    "error while shutting down the job",
                    Some(error),
                    Vec::new(),
                );
            }
        }
    }
}

/// One shutdown callback, in a span of its own unless livekit-agents registered it.
async fn shutdown_callback(
    host: Arc<dyn JobProcess>,
    cx: CallContext,
    parent: telemetry::Cx,
    ShutdownCallback { callback, framework, name, .. }: ShutdownCallback,
    reason: Option<String>,
) -> Result<(), HostError> {
    if framework {
        return host.shutdown_callback(&parent, &cx, callback, reason).await;
    }
    let span = telemetry::start(&parent, telemetry::ShutdownCallback { callback: name });
    let ran = host.shutdown_callback(span.cx(), &cx, callback, reason).await;
    telemetry::end(span, ran.as_ref());
    ran
}

fn log_level(level: Level) -> log::Level {
    match level {
        Level::Debug => log::Level::Debug,
        Level::Info => log::Level::Info,
        Level::Warning => log::Level::Warn,
        Level::Error => log::Level::Error,
    }
}

impl lk_agents_abi::worker::ipc_host::JobRun for JobRun {
    type Env = ProcessEnv;

    #[allow(clippy::too_many_arguments, reason = "Preserve the public host operation signature")]
    fn new(
        env: &ProcessEnv,
        fake_job: bool,
        session_end_timeout: f64,
        redaction: bool,
        url: String,
        session_end: Option<String>,
        session_aclose_timeout: f64,
        observability_url: Option<String>,
        api_key: String,
        api_secret: String,
    ) -> Result<Self, WorkerError> {
        let observability_url =
            lk_agents_net::strings::observability_url(&url, observability_url.as_deref());
        let cloud = observability_url
            .clone()
            .filter(|_| !fake_job && !api_key.is_empty())
            .map(|base_url| telemetry::CloudEndpoint { base_url, api_key, api_secret });
        Ok(Self {
            inner: Arc::new(Inner {
                host: OnceLock::new(),
                parent: OnceLock::new(),
                spawner: TokioSpawner::new((env.runtime)().map_err(WorkerError::Failed)?),
                fake: fake_job,
                session_end_timeout,
                session_end,
                session_aclose_timeout,
                observability_url,
                state: Mutex::new(State { redaction, ..State::default() }),
                ended: watch::channel(false).0,
                connecting: tokio::sync::Mutex::new(()),
                tasks_closed: watch::channel(false).0,
                participants: Arc::default(),
                tracer: env.tracer.clone(),
                clock: env.clock.clone(),
                cloud,
                root: Mutex::new(None),
                job: OnceLock::new(),
            }),
        })
    }

    fn add_shutdown_callback(
        &self,
        callback: HostRef,
        takes_reason: bool,
        framework: bool,
        name: String,
    ) {
        self.inner.state.lock().shutdown_callbacks.push(ShutdownCallback {
            callback,
            takes_reason,
            framework,
            name,
        });
    }

    fn start_trace(&self, fields: StartJobFields) {
        let inner = &self.inner;
        let telemetry::JobTrace { job, span } =
            telemetry::start_job(&inner.tracer, inner.cloud.as_ref(), &fields);
        let _ = inner.job.set(job);
        *inner.root.lock() = Some(span);
    }

    /// The host's `observability_url`, else the server's when it is LiveKit Cloud's.
    fn shutdown(&self, reason: String) {
        self.inner.end(reason, telemetry::ShutdownCause::Application);
    }

    fn room_disconnected(&self) {
        self.inner.end("room disconnected".into(), telemetry::ShutdownCause::RoomDisconnected);
    }

    fn connected(&self) {
        self.inner.state.lock().connected = true;
    }

    async fn connect(
        &self,
        connect: Arc<dyn JobCall>,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<(), WorkerError> {
        let inner = self.inner.clone();
        self.inner
            .spawner
            .run(async move {
                let _connecting = inner.connecting.lock().await;
                if inner.state.lock().connected {
                    return Ok(());
                }
                connect.call(&cx, &context).await?;
                inner.state.lock().connected = true;
                Ok(())
            })
            .await
    }

    fn add_participant_entrypoint(
        &self,
        entrypoint: Arc<dyn ParticipantEntrypoint>,
        kinds: Vec<i32>,
        name: String,
    ) {
        self.inner.state.lock().entrypoints.push(Entrypoint { run: entrypoint, kinds, name });
    }

    fn participant_joined(
        &self,
        job: HostRef,
        participant: HostRef,
        identity: String,
        kind: i32,
        context: CallContext,
        cx: telemetry::Cx,
    ) {
        let mut starts = Vec::new();
        let mut overlapping = false;
        {
            let mut st = self.inner.state.lock();
            let matching: Vec<(usize, Arc<dyn ParticipantEntrypoint>, String)> = st
                .entrypoints
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.kinds.contains(&kind))
                .map(|(i, entry)| (i, entry.run.clone(), entry.name.clone()))
                .collect();
            for (index, entry, name) in matching {
                let key = (identity.clone(), index);
                st.runs += 1;
                let run = st.runs;
                overlapping |= st.participant_runs.insert(key.clone(), run).is_some();
                starts.push((key, run, entry, name));
            }
        }
        if overlapping {
            self.inner.log(Level::Warning, &format!("a participant has joined before a prior participant task matching the same identity has finished: '{identity}'"), None, Vec::new());
        }
        let spawner = TokioSpawner::new(self.inner.spawner.handle().clone());
        for (key, run, entry, name) in starts {
            let inner = self.inner.clone();
            let (job, participant, context, cx) =
                (job.clone(), participant.clone(), context.clone(), cx.clone());
            self.inner.participants.spawn(&spawner, "participant entrypoint", async move {
                let ran = entry.run(&cx, &context, participant, job).await;
                {
                    let mut st = inner.state.lock();
                    if st.participant_runs.get(&key) == Some(&run) {
                        st.participant_runs.remove(&key);
                    }
                }
                if let Err(error) = ran
                    && !cancelled(&error)
                {
                    let msg = format!("error in participant entrypoint {name} for '{}'", key.0);
                    inner.log(Level::Error, &msg, Some(error), Vec::new());
                }
            });
        }
    }

    /// A simulator leaving ends the job.
    fn participant_left(&self, attributes: BTreeMap<String, String>) {
        if attributes.contains_key(ATTRIBUTE_SIMULATOR) {
            self.inner.log(
                Level::Debug,
                "simulator disconnected, shutting down the job",
                None,
                Vec::new(),
            );
            self.inner
                .end("simulation completed".into(), telemetry::ShutdownCause::SimulationCompleted);
        }
    }

    async fn api_call(
        &self,
        call: Arc<dyn JobCall>,
        name: String,
        console_result: HostRef,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<HostRef, WorkerError> {
        if self.inner.fake {
            self.inner.log(
                Level::Warning,
                &format!("job_ctx.{name}() is not executed while in console mode"),
                None,
                Vec::new(),
            );
            return Ok(console_result);
        }
        let (tx, rx) = oneshot::channel();
        let mut closed = self.inner.tasks_closed.subscribe();
        let inner = self.inner.clone();
        let task = TokioSpawner::new(self.inner.spawner.handle().clone()).spawn(async move {
            let called = tokio::select! {
                called = call.call(&cx, &context) => called.map_err(WorkerError::Host),
                _ = closed.wait_for(|closed| *closed) => Err(WorkerError::Cancelled),
            };
            if let Err(WorkerError::Host(error)) = &called
                && !cancelled(error)
            {
                inner.log(
                    Level::Error,
                    &format!("error in {name}"),
                    Some(error.clone()),
                    Vec::new(),
                );
            }
            let _ = tx.send(called);
        });
        {
            let mut st = self.inner.state.lock();
            st.tasks.retain(|t| !t.is_finished());
            st.tasks.push(task);
        }
        rx.await.unwrap_or(Err(WorkerError::Cancelled))
    }

    async fn close_tasks(&self) {
        self.inner.tasks_closed.send_replace(true);
        let tasks = std::mem::take(&mut self.inner.state.lock().tasks);
        for task in tasks {
            let _ = task.await;
        }
    }

    /// Audio without its transcript under redaction is refused.
    fn init_recording(&self, options: BTreeMap<String, bool>) -> Result<(), WorkerError> {
        let observability = self.inner.observability_url.is_some();
        let get = |key: &str, default: bool| options.get(key).copied().unwrap_or(default);
        let mut st = self.inner.state.lock();
        let redaction = st.redaction || get("redaction", false);
        if redaction && get("audio", true) && !get("transcript", true) {
            return Err(WorkerError::Invalid(
                "audio upload requires transcript upload when redaction is enabled".into(),
            ));
        }
        if st.recording_initialized {
            return Ok(());
        }
        st.recording_initialized = true;
        st.redaction = redaction;
        let options = telemetry::RecordingOptions {
            audio: get("audio", true),
            traces: get("traces", true),
            logs: get("logs", true),
            transcript: get("transcript", true),
            redaction,
        };
        st.recording = Some(options);
        let uploads = options.recording_enabled();
        drop(st);
        if let (Some(exporter), Some(job)) = (self.inner.tracer.exporter(), self.inner.job.get()) {
            let cloud = observability && self.inner.cloud.is_some();
            // Metrics follow the recording as a whole, as 1.8.10's process meter did; the only
            // logs are the session report's records.
            let recording = telemetry::CloudRecording {
                traces: cloud && options.traces,
                logs: cloud && options.logs,
                metrics: cloud && uploads,
                redaction,
                simulation: job.simulation.clone(),
            };
            exporter.register_job(&job.key, recording);
        }
        Ok(())
    }

    async fn run(
        &self,
        host: Arc<dyn JobProcess>,
        context: CallContext,
    ) -> Result<(), WorkerError> {
        if self.inner.host.set((host, context)).is_err() {
            return Err(WorkerError::Failed("the job already runs".into()));
        }
        let inner = self.inner.clone();
        self.inner.spawner.run(inner.run()).await
    }
}

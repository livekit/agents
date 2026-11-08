//! A job's run inside its process (`_core.JobRun`): what ends the job (the room, the user, the
//! parent, a crash, the simulator leaving; the first wins), the entrypoint's supervision, the
//! order and bounds of the shutdown (whose steps the host runs and opens the spans of), the
//! session report's outputs, the room connected once, the application calls in flight and the
//! recording's setup.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use super::telemetry;
use lk_agents_abi::errors::{HostError, HostErrorKind};
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host::{
    FrameworkShutdown, JobCall, JobExecution, JobProcess, WorkerError,
};
use lk_agents_abi::worker::job::{ShutdownCause, StartJob};
use lk_agents_abi::{ATTRIBUTE_SIMULATOR, HostRef};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::ipc_channel::Channel;

use super::wire::{ToChild, bounded};
use super::{ToParent, secs};
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
    cause: Option<ShutdownCause>,
    connected: bool,
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

enum ShutdownCallback {
    /// `ctx.add_shutdown_callback`'s; one livekit-agents registered (`framework`) runs in the
    /// shutdown's span, any other in a span of its own named `name`.
    Host { callback: HostRef, takes_reason: bool, framework: bool, name: String },
    /// A session's or an avatar's close, in the shutdown's span.
    Framework { name: String, shutdown: FrameworkShutdown },
}

impl ShutdownCallback {
    fn name(&self) -> &str {
        match self {
            ShutdownCallback::Host { name, .. } | ShutdownCallback::Framework { name, .. } => name,
        }
    }
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
    ended: CancellationToken,
    /// Held while the room connects, so a second `connect()` waits for the first.
    connecting: tokio::sync::Mutex<()>,
    /// Set once the job ends: the application calls still running are cancelled.
    tasks_closed: CancellationToken,
    tracer: telemetry::Tracer,
    /// Times the session report's outputs.
    clock: Arc<dyn lk_agents_abi::Clock>,
    /// Where the job's Cloud telemetry goes, signed with the server's credentials.
    cloud: Option<telemetry::CloudEndpoint>,
    /// The job's root span, from its start until its shutdown ends.
    root: Mutex<Option<telemetry::ScopedSpan<telemetry::JobEntrypoint>>>,
    /// The job the parent handed over; none for a job run outside one.
    job: OnceLock<telemetry::Job>,
    /// The primary session's `agent_session`, once it started.
    session: OnceLock<telemetry::Cx>,
    /// The job's `job_entrypoint`, once it runs.
    entry: OnceLock<telemetry::Cx>,
}

impl JobExecution for JobRun {
    fn session_closed(&self, reason: String) {
        self.inner.end(reason, ShutdownCause::SessionClosed);
    }

    fn add_framework_shutdown(&self, name: String, shutdown: FrameworkShutdown) {
        let callback = ShutdownCallback::Framework { name, shutdown };
        self.inner.state.lock().shutdown_callbacks.push(callback);
    }

    fn session_started(&self, trace: telemetry::Cx) {
        let _ = self.inner.session.set(trace);
    }

    fn rpc_trace(&self) -> Option<telemetry::Cx> {
        self.inner.session.get().or(self.inner.entry.get()).cloned()
    }
}

impl JobRun {
    /// The job ends with `reason`; the first reason given is the one kept.
    pub(crate) fn end(&self, reason: String, cause: ShutdownCause) {
        self.inner.end(reason, cause);
    }

    pub(crate) fn report_to(&self, parent: Arc<Channel<ToParent, ToChild>>) {
        let _ = self.inner.parent.set(parent);
    }
}

fn cancelled(error: &HostError) -> bool {
    error.kind == HostErrorKind::Cancelled
}

impl Inner {
    fn end(&self, reason: String, cause: ShutdownCause) {
        let mut st = self.state.lock();
        // The session's close can reach the job before the application's own `shutdown`.
        if st.cause.is_none() || cause == ShutdownCause::Application {
            st.cause = Some(cause);
        }
        if st.shutdown.is_none() {
            st.shutdown = Some(reason);
            self.ended.cancel();
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
                    log::error!(target: lk_agents_abi::LOG_TARGET, exc_text:% = error.exc_text(); "{error_msg}");
                }
            }
            Err(()) => {
                telemetry::end(span, lk_agents_abi::Elapsed);
                let after = timeout.unwrap_or_default() as i64;
                log::error!(target: lk_agents_abi::LOG_TARGET, "{} timed out after {after}s", D::NAME);
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
        let exporter = telemetry::exporter();
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
        let _ = self.entry.set(span.cx().clone());
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
        if let (Some(exporter), Some(job)) = (telemetry::exporter(), self.job.get()) {
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
        let wait_entry = entry.wait(job, cx);
        tokio::pin!(wait_entry);
        let warn_at = tokio::time::sleep(CONNECT_WARNING_DELAY);
        tokio::pin!(warn_at);
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
                () = self.ended.cancelled() => break,
            }
        }
        let (reason, cause) = {
            let st = self.state.lock();
            let cause = st.cause.unwrap_or(ShutdownCause::ProcessShutdown);
            (st.shutdown.clone().unwrap_or_default(), cause)
        };
        let shutdown =
            telemetry::start(job, telemetry::JobShutdown { reason: reason.clone(), cause });

        // The entrypoint returns, or is cancelled.
        if !entry_done
            && tokio::time::timeout(ENTRYPOINT_EXIT_TIMEOUT, &mut wait_entry).await.is_err()
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "entrypoint did not exit in time, cancelling");
            let _ = entry.cancel(shutdown.cx(), cx).await;
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
                log::error!(target: lk_agents_abi::LOG_TARGET, exc_text:% = e.exc_text(); "unhandled exception while running the job task");
                // A crash before ctx.connect() resolves nothing else: the shutdown
                // still runs, so the cleanup can flush the crash logs.
                self.end("job crashed".into(), ShutdownCause::EntrypointCrashed);
                Some(e)
            }
            Ok(()) => {
                telemetry::event(root, telemetry::EntrypointReturned);
                let unfinished = {
                    let st = self.state.lock();
                    !st.connected && st.cause != Some(ShutdownCause::Application) && !self.fake
                };
                if unfinished {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "The job task completed without establishing a connection or performing a proper shutdown. Ensure that job_ctx.connect()/job_ctx.shutdown() is called and the job is correctly finalized.");
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
            !st.connected && st.cause != Some(ShutdownCause::Application)
        };
        if unconnected {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "The room connection was not established within 10 seconds after calling job_entry. This might mean that job_ctx.connect() was never invoked, or that no VoiceSession with an active RoomIO has been started.");
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
                log::error!(target: lk_agents_abi::LOG_TARGET, exc_text:% = error.exc_text(); "error while closing the session")
            }
            Ok(Ok(())) => {}
            Err(_) => {
                log::error!(target: lk_agents_abi::LOG_TARGET, "VoiceSession.aclose() timed out after {aclose_timeout:.1}s; proceeding with shutdown so registered callbacks still run.")
            }
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
    async fn announce_exit(&self, reason: &str, cause: ShutdownCause) {
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::ShuttingDown).await;
        }
        let user_initiated = cause == ShutdownCause::Application;
        log::debug!(target: lk_agents_abi::LOG_TARGET, reason = reason, user_initiated = user_initiated; "shutting down job task");
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
                let name = callback.name().to_owned();
                let (host, cx, under, reason) =
                    (host.clone(), cx.clone(), under.clone(), reason.to_owned());
                (name, spawner.spawn(shutdown_callback(host, cx, under, callback, reason)))
            })
            .collect();
        for (name, run) in runs {
            if let Ok(Err(error)) = run.await {
                log::error!(target: lk_agents_abi::LOG_TARGET, callback = name.as_str(), exc_text:% = error.exc_text(); "error while shutting down the job");
            }
        }
    }
}

/// One shutdown callback, in a span of its own unless livekit-agents registered it.
async fn shutdown_callback(
    host: Arc<dyn JobProcess>,
    cx: CallContext,
    parent: telemetry::Cx,
    callback: ShutdownCallback,
    reason: String,
) -> Result<(), HostError> {
    let (callback, takes_reason, name) = match callback {
        ShutdownCallback::Framework { shutdown: FrameworkShutdown::Native(close), .. } => {
            close().await;
            return Ok(());
        }
        ShutdownCallback::Framework { shutdown: FrameworkShutdown::Host(close), .. } => {
            return host.shutdown_callback(&parent, &cx, close, None).await;
        }
        ShutdownCallback::Host { callback, takes_reason, framework: true, .. } => {
            let reason = takes_reason.then_some(reason);
            return host.shutdown_callback(&parent, &cx, callback, reason).await;
        }
        ShutdownCallback::Host { callback, takes_reason, framework: false, name } => {
            (callback, takes_reason, name)
        }
    };
    let reason = takes_reason.then_some(reason);
    let span = telemetry::start(&parent, telemetry::ShutdownCallback { callback: name });
    let ran = host.shutdown_callback(span.cx(), &cx, callback, reason).await;
    telemetry::end(span, ran.as_ref());
    ran
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
        api_key: String,
        api_secret: String,
    ) -> Result<Self, WorkerError> {
        let configured = std::env::var("LIVEKIT_OBSERVABILITY_URL").ok();
        let observability_url =
            lk_agents_net::strings::observability_url(&url, configured.as_deref());
        let cloud = observability_url
            .clone()
            .filter(|_| !fake_job && !api_key.is_empty())
            .map(|base_url| telemetry::CloudEndpoint { base_url, api_key, api_secret });
        Ok(Self {
            inner: Arc::new(Inner {
                host: OnceLock::new(),
                parent: OnceLock::new(),
                spawner: TokioSpawner::new(
                    (env.runtime)().map_err(|error| WorkerError::Failed(error.to_string()))?,
                ),
                fake: fake_job,
                session_end_timeout,
                session_end,
                session_aclose_timeout,
                observability_url,
                state: Mutex::new(State { redaction, ..State::default() }),
                ended: CancellationToken::new(),
                connecting: tokio::sync::Mutex::new(()),
                tasks_closed: CancellationToken::new(),
                tracer: env.tracer.clone(),
                clock: env.clock.clone(),
                cloud,
                root: Mutex::new(None),
                job: OnceLock::new(),
                session: OnceLock::new(),
                entry: OnceLock::new(),
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
        let callback = ShutdownCallback::Host { callback, takes_reason, framework, name };
        self.inner.state.lock().shutdown_callbacks.push(callback);
    }

    fn start_trace(&self, job: StartJob) {
        let inner = &self.inner;
        let telemetry::JobTrace { job, span } =
            telemetry::start_job(&inner.tracer, inner.cloud.as_ref(), &job);
        let _ = inner.job.set(job);
        *inner.root.lock() = Some(span);
    }

    /// The host's `observability_url`, else the server's when it is LiveKit Cloud's.
    fn shutdown(&self, reason: String) {
        self.inner.end(reason, ShutdownCause::Application);
    }

    fn room_disconnected(&self) {
        self.inner.end("room disconnected".into(), ShutdownCause::RoomDisconnected);
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

    /// A simulator leaving ends the job.
    fn participant_left(&self, attributes: BTreeMap<String, String>) {
        if attributes.contains_key(ATTRIBUTE_SIMULATOR) {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "simulator disconnected, shutting down the job");
            self.inner.end("simulation completed".into(), ShutdownCause::SimulationCompleted);
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
            log::warn!(target: lk_agents_abi::LOG_TARGET, "job_ctx.{name}() is not executed while in console mode");
            return Ok(console_result);
        }
        let (tx, rx) = oneshot::channel();
        let closed = self.inner.tasks_closed.clone();
        let task = TokioSpawner::new(self.inner.spawner.handle().clone()).spawn(async move {
            let called = tokio::select! {
                called = call.call(&cx, &context) => called.map_err(WorkerError::Host),
                () = closed.cancelled() => Err(WorkerError::Cancelled),
            };
            if let Err(WorkerError::Host(error)) = &called
                && !cancelled(error)
            {
                log::error!(target: lk_agents_abi::LOG_TARGET, exc_text:% = error.exc_text(); "error in {name}");
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
        self.inner.tasks_closed.cancel();
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
        if let (Some(exporter), Some(job)) = (telemetry::exporter(), self.inner.job.get()) {
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

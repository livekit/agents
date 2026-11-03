//! A job's run inside its process (`_core.JobRun`): what ends the job (the room, the user, the
//! parent, a crash, the simulator leaving; the first wins), the entrypoint's supervision, the
//! order and bounds of the shutdown (whose steps the host runs and opens the spans of), the
//! participant entrypoints, the room connected once, the application calls in flight and the
//! recording's setup.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::telemetry::decl::SpanDecl;
use lk_agents_abi::telemetry::policy::SDK_SHUTDOWN_TIMEOUT_SECONDS;
use lk_agents_abi::telemetry::spans::job::{
    EntrypointCancelled, EntrypointReturned, JobShutdown, OnSessionEnd, RoomDisconnect,
    SessionEndUpload, ShutdownCallback as ShutdownCallbackSpan,
};
use lk_agents_abi::worker::ipc_host::{
    JobCall, JobExecution, JobProcess, ParticipantEntrypoint, WorkerError,
};
use lk_agents_abi::worker::job::StartJobFields;
use lk_agents_abi::{ATTRIBUTE_SIMULATOR, HostRef};
use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::otlp::{CloudEndpoint, CloudRecording};
use lk_agents_net::task::{TaskSet, TokioSpawner};
use lk_agents_net::telemetry::{Cx, ScopedSpan, Tracer};
use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};

use super::ipc_channel::Channel;
use super::job_trace::{self, JobTrace};
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
    /// The first reason the job ends with, and whether the user asked for it.
    shutdown: Option<(String, bool)>,
    connected: bool,
    user_shutdown: bool,
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
    /// The server asked for the job's recording.
    recording: bool,
    state: Mutex<State>,
    ended: watch::Sender<bool>,
    /// Held while the room connects, so a second `connect()` waits for the first.
    connecting: tokio::sync::Mutex<()>,
    /// Set once the job ends: the application calls still running are cancelled.
    tasks_closed: watch::Sender<bool>,
    /// The participant entrypoints, which run to their own end.
    participants: Arc<TaskSet>,
    tracer: Tracer,
    /// Where the job's Cloud telemetry goes, signed with the server's credentials.
    cloud: Option<CloudEndpoint>,
    /// The job's root span, from its start until its shutdown ends.
    trace: Mutex<Option<JobTrace>>,
    /// The job the exporter's registry knows it by.
    job_key: OnceLock<lk_agents_abi::telemetry::record::JobKey>,
}

impl JobExecution for JobRun {}

impl JobRun {
    /// The job ends with `reason`; the first reason given is the one kept.
    pub(crate) fn end(&self, reason: String, user_initiated: bool) {
        self.inner.end(reason, user_initiated);
    }

    pub(crate) fn report_to(&self, parent: Arc<Channel<ToParent, ToChild>>) {
        let _ = self.inner.parent.set(parent);
    }
}

/// `cx` with `span` current.
fn under(cx: &CallContext, span: &Cx) -> CallContext {
    CallContext { span: span.span(), ..cx.clone() }
}

fn cancelled(error: &HostError) -> bool {
    error.kind == "CancelledError"
}

impl Inner {
    fn end(&self, reason: String, user_initiated: bool) {
        let mut st = self.state.lock();
        st.user_shutdown |= user_initiated;
        if st.shutdown.is_none() {
            st.shutdown = Some((reason, user_initiated));
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
    async fn step<D: SpanDecl<End = ()>>(
        &self,
        span: ScopedSpan<D>,
        run: impl Future<Output = Result<(), HostError>>,
        timeout: Option<f64>,
        error_msg: &str,
    ) {
        let outcome = match timeout {
            Some(t) => tokio::time::timeout(secs(t), run).await.map_err(|_| ()),
            None => Ok(run.await),
        };
        match outcome {
            Ok(Ok(())) => span.end(()),
            Ok(Err(error)) => {
                span.fail(&error);
                self.log(Level::Error, error_msg, Some(error), Vec::new());
            }
            Err(()) => {
                span.fail(&lk_agents_abi::Elapsed);
                let msg =
                    format!("{} timed out after {}s", D::NAME, timeout.unwrap_or_default() as i64);
                self.log(Level::Error, &msg, None, Vec::new());
            }
        }
    }

    /// The job's root span (one of its own when the parent never handed a job over).
    fn take_trace(&self) -> JobTrace {
        let trace = self.trace.lock().take();
        trace.unwrap_or_else(|| job_trace::start(&self.tracer, None, &StartJobFields::default()))
    }

    async fn run(self: Arc<Self>) -> Result<(), WorkerError> {
        let trace = self.take_trace();
        let result = self.clone().run_traced(&trace).await;
        if let Err(error) = &result {
            trace.span.fail(&HostError::new(error.to_string()));
        }
        let JobTrace { key, span } = trace;
        span.end(());
        if let Some(exporter) = self.tracer.exporter() {
            exporter.close_job(&key, secs(SDK_SHUTDOWN_TIMEOUT_SECONDS)).await;
        }
        result
    }

    async fn run_traced(self: Arc<Self>, trace: &JobTrace) -> Result<(), WorkerError> {
        let (host, cx) = self.host.get().expect("run() sets the host first");
        let job = trace.span.cx();
        let cx = &under(cx, job);
        let entry = host.start_entrypoint(cx).await?;
        let wait_entry = host.wait(cx, entry.clone());
        tokio::pin!(wait_entry);
        let warn_at = tokio::time::sleep(CONNECT_WARNING_DELAY);
        tokio::pin!(warn_at);
        let mut ended = self.ended.subscribe();
        let (mut entry_done, mut warned) = (false, self.fake);
        loop {
            tokio::select! {
                r = &mut wait_entry, if !entry_done => {
                    entry_done = true;
                    match r {
                        Err(e) if cancelled(&e) => trace.span.event(EntrypointCancelled),
                        Err(e) => {
                            trace.span.fail(&e);
                            self.log(Level::Error, "unhandled exception while running the job task", Some(e), Vec::new());
                            // A crash before ctx.connect() resolves nothing else: the shutdown
                            // still runs, so the cleanup can flush the crash logs.
                            self.end("job crashed".into(), false);
                        }
                        Ok(()) => {
                            trace.span.event(EntrypointReturned);
                            let unfinished = {
                                let st = self.state.lock();
                                !st.connected && !st.user_shutdown && !self.fake
                            };
                            if unfinished {
                                self.log(Level::Warning, "The job task completed without establishing a connection or performing a proper shutdown. Ensure that job_ctx.connect()/job_ctx.shutdown() is called and the job is correctly finalized.", None, Vec::new());
                            }
                        }
                    }
                }
                () = &mut warn_at, if !entry_done && !warned => {
                    warned = true;
                    let unconnected = {
                        let st = self.state.lock();
                        !st.connected && !st.user_shutdown
                    };
                    if unconnected {
                        self.log(Level::Warning, "The room connection was not established within 10 seconds after calling job_entry. This might mean that job_ctx.connect() was never invoked, or that no VoiceSession with an active RoomIO has been started.", None, Vec::new());
                    }
                }
                _ = ended.wait_for(|e| *e) => break,
            }
        }
        let (reason, user_initiated) = self.state.lock().shutdown.clone().unwrap_or_default();
        let shutdown = job.start_span(JobShutdown { reason: reason.clone(), user_initiated });
        let cx = &under(cx, shutdown.cx());

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
            let _ = host.cancel(cx, entry).await;
            trace.span.event(EntrypointCancelled);
        }

        let aclose_timeout = self.session_aclose_timeout;
        match tokio::time::timeout(secs(aclose_timeout), host.close_session(cx)).await {
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

        if let Some(session_end) = &self.session_end {
            let span = shutdown.cx().start_span(OnSessionEnd { callback: session_end.clone() });
            let call = under(cx, span.cx());
            self.step(
                span,
                host.session_end(&call),
                Some(self.session_end_timeout),
                "error while executing the on_session_end callback",
            )
            .await;
        }
        let span = shutdown.cx().start_span(SessionEndUpload);
        let call = under(cx, span.cx());
        self.step(
            span,
            host.upload(&call),
            Some(lk_agents_abi::telemetry::policy::UPLOAD_TIMEOUT_SECONDS as f64),
            "error in job_ctx._on_session_end",
        )
        .await;
        // The parent learns the job is ending, then why.
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::ShuttingDown).await;
        }
        let extra = vec![
            ("reason", ExtraValue::Str(reason.clone())),
            ("user_initiated", ExtraValue::Bool(user_initiated)),
        ];
        self.log(Level::Debug, "shutting down job task", None, extra);
        if let Some(parent) = self.parent.get() {
            let _ = parent.send(ToParent::Exiting { reason: bounded(reason.clone()) }).await;
        }
        let disconnect = shutdown.cx().start_span(RoomDisconnect);
        let disconnected = disconnect.check(host.disconnect(&under(cx, disconnect.cx())).await);
        disconnect.end(());
        disconnected?;

        // The shutdown callbacks run together.
        let callbacks = std::mem::take(&mut self.state.lock().shutdown_callbacks);
        let spawner = TokioSpawner::new(self.spawner.handle().clone());
        let runs: Vec<_> = callbacks
            .into_iter()
            .map(|ShutdownCallback { callback, takes_reason, framework, name }| {
                let (host, cx, parent) = (host.clone(), cx.clone(), shutdown.cx().clone());
                let reason = takes_reason.then(|| reason.clone());
                spawner.spawn(async move {
                    if framework {
                        return host.shutdown_callback(&cx, callback, reason).await;
                    }
                    let span = parent.start_span(ShutdownCallbackSpan { callback: name });
                    let call = under(&cx, span.cx());
                    let ran = span.check(host.shutdown_callback(&call, callback, reason).await);
                    span.end(());
                    ran
                })
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
        shutdown.end(());
        Ok(())
    }
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
        recording: bool,
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
            .map(|base_url| CloudEndpoint { base_url, api_key, api_secret });
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
                recording,
                state: Mutex::new(State { redaction, ..State::default() }),
                ended: watch::channel(false).0,
                connecting: tokio::sync::Mutex::new(()),
                tasks_closed: watch::channel(false).0,
                participants: Arc::default(),
                tracer: env.tracer.clone(),
                cloud,
                trace: Mutex::new(None),
                job_key: OnceLock::new(),
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
        let trace = job_trace::start(&inner.tracer, inner.cloud.as_ref(), &fields);
        let _ = inner.job_key.set(trace.key.clone());
        *inner.trace.lock() = Some(trace);
    }

    /// The host's `observability_url`, else the server's when it is LiveKit Cloud's.
    fn observability_url(&self) -> Option<String> {
        self.inner.observability_url.clone()
    }

    /// The server asked for the recording of a real job that has an observability endpoint.
    fn buffer_logs(&self) -> bool {
        !self.inner.fake && self.inner.recording && self.inner.observability_url.is_some()
    }

    fn shutdown(&self, reason: String, user_initiated: bool) {
        self.inner.end(reason, user_initiated);
    }

    fn connected(&self) {
        self.inner.state.lock().connected = true;
    }

    async fn connect(
        &self,
        connect: Arc<dyn JobCall>,
        context: CallContext,
    ) -> Result<(), WorkerError> {
        let inner = self.inner.clone();
        self.inner
            .spawner
            .run(async move {
                let _connecting = inner.connecting.lock().await;
                if inner.state.lock().connected {
                    return Ok(());
                }
                connect.call(&context).await?;
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
            let (job, participant, context) = (job.clone(), participant.clone(), context.clone());
            self.inner.participants.spawn(&spawner, "participant entrypoint", async move {
                let ran = entry.run(&context, job, participant).await;
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
            self.inner.end("simulation completed".into(), true);
        }
    }

    async fn api_call(
        &self,
        call: Arc<dyn JobCall>,
        name: String,
        console_result: HostRef,
        context: CallContext,
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
                called = call.call(&context) => called.map_err(WorkerError::Host),
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
    fn init_recording(&self, options: BTreeMap<String, bool>) -> Result<Option<bool>, WorkerError> {
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
            return Ok(None);
        }
        st.recording_initialized = true;
        st.redaction = redaction;
        let uploads = ["traces", "logs", "audio", "transcript"].iter().any(|key| get(key, true));
        drop(st);
        if let (Some(exporter), Some(key)) =
            (self.inner.tracer.exporter(), self.inner.job_key.get())
        {
            let cloud = observability && self.inner.cloud.is_some();
            let recording = CloudRecording {
                traces: cloud && get("traces", true),
                logs: cloud && get("logs", true),
                metrics: cloud && get("traces", true),
                redaction,
                simulation: None,
            };
            exporter.register_job(key, recording);
        }
        Ok((uploads && observability).then_some(redaction))
    }

    fn recording_initialized(&self) -> bool {
        self.inner.state.lock().recording_initialized
    }

    fn redaction(&self) -> bool {
        self.inner.state.lock().redaction
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

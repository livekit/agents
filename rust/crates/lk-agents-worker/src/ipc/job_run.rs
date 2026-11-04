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
    trace: Mutex<Option<telemetry::JobTrace>>,
    /// The job the exporter's registry knows it by.
    job_key: OnceLock<telemetry::JobKey>,
    /// The simulation run the job is part of.
    simulation: OnceLock<Option<telemetry::Simulation>>,
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
fn under(cx: &CallContext, span: &telemetry::Cx) -> CallContext {
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
    async fn step<D: telemetry::SpanDecl<End = ()>>(
        &self,
        span: telemetry::ScopedSpan<D>,
        run: impl Future<Output = Result<(), HostError>>,
        timeout: Option<f64>,
        error_msg: &str,
    ) {
        let outcome = match timeout {
            Some(t) => tokio::time::timeout(secs(t), run).await.map_err(|_| ()),
            None => Ok(run.await),
        };
        match outcome {
            Ok(Ok(())) => telemetry::end_with(span, ()),
            Ok(Err(error)) => {
                telemetry::fail(&span, &error);
                self.log(Level::Error, error_msg, Some(error), Vec::new());
            }
            Err(()) => {
                telemetry::fail(&span, &lk_agents_abi::Elapsed);
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
    ) -> Result<(), HostError> {
        let Some(report) = host.session_report(cx).await? else { return Ok(()) };
        let (Some(options), Some(key)) = (self.state.lock().recording, self.job_key.get()) else {
            return Ok(());
        };
        let exporter = self.tracer.exporter();
        let outputs = telemetry::ReportOutputs {
            exporter: exporter.as_deref(),
            cloud: self.cloud.as_ref().filter(|_| self.observability_url.is_some()),
            clock: &*self.clock,
        };
        let simulated = self.simulation.get().is_some_and(Option::is_some);
        let limit = Duration::from_secs(telemetry::UPLOAD_TIMEOUT_SECONDS);
        telemetry::session_report::export(outputs, key, &report, &options, simulated, limit)
            .await
            .map_err(|error| HostError::new(error.to_string()))
    }

    /// The job's root span (one of its own when the parent never handed a job over).
    fn take_trace(&self) -> telemetry::JobTrace {
        let trace = self.trace.lock().take();
        trace
            .unwrap_or_else(|| telemetry::start_job(&self.tracer, None, &StartJobFields::default()))
    }

    async fn run(self: Arc<Self>) -> Result<(), WorkerError> {
        let trace = self.take_trace();
        let result = self.clone().run_traced(&trace).await;
        if let Err(error) = &result {
            telemetry::fail(&trace.span, &HostError::new(error.to_string()));
        }
        let telemetry::JobTrace { key, span, .. } = trace;
        telemetry::end_with(span, ());
        if let Some(exporter) = self.tracer.exporter() {
            exporter.close_job(&key, secs(telemetry::SHUTDOWN_TIMEOUT_SECONDS)).await;
        }
        result
    }

    async fn run_traced(self: Arc<Self>, trace: &telemetry::JobTrace) -> Result<(), WorkerError> {
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
                        Err(e) if cancelled(&e) => telemetry::event(&trace.span, telemetry::EntrypointCancelled),
                        Err(e) => {
                            telemetry::fail(&trace.span, &e);
                            self.log(Level::Error, "unhandled exception while running the job task", Some(e), Vec::new());
                            // A crash before ctx.connect() resolves nothing else: the shutdown
                            // still runs, so the cleanup can flush the crash logs.
                            self.end("job crashed".into(), false);
                        }
                        Ok(()) => {
                            telemetry::event(&trace.span, telemetry::EntrypointReturned);
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
        let shutdown = telemetry::start(
            &job,
            telemetry::JobShutdown { reason: reason.clone(), user_initiated },
        );
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
            telemetry::event(&trace.span, telemetry::EntrypointCancelled);
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
            let span = telemetry::start(
                &shutdown.cx(),
                telemetry::OnSessionEnd { callback: session_end.clone() },
            );
            let call = under(cx, span.cx());
            self.step(
                span,
                host.session_end(&call),
                Some(self.session_end_timeout),
                "error while executing the on_session_end callback",
            )
            .await;
        }
        let span = telemetry::start(&shutdown.cx(), telemetry::SessionEndUpload);
        let call = under(cx, span.cx());
        self.step(
            span,
            self.session_end_upload(host, &call),
            Some(telemetry::UPLOAD_TIMEOUT_SECONDS as f64),
            "failed to upload the session report",
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
        let disconnect = telemetry::start(&shutdown.cx(), telemetry::RoomDisconnect);
        let disconnected =
            telemetry::check(&disconnect, host.disconnect(&under(cx, disconnect.cx())).await);
        telemetry::end_with(disconnect, ());
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
                    let span =
                        telemetry::start(&parent, telemetry::ShutdownCallback { callback: name });
                    let call = under(&cx, span.cx());
                    let ran = telemetry::check(
                        &span,
                        host.shutdown_callback(&call, callback, reason).await,
                    );
                    telemetry::end_with(span, ());
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
        telemetry::end_with(shutdown, ());
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
                trace: Mutex::new(None),
                job_key: OnceLock::new(),
                simulation: OnceLock::new(),
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
        let trace = telemetry::start_job(&inner.tracer, inner.cloud.as_ref(), &fields);
        let _ = inner.job_key.set(trace.key.clone());
        let _ = inner.simulation.set(trace.simulation.clone());
        *inner.trace.lock() = Some(trace);
    }

    /// The host's `observability_url`, else the server's when it is LiveKit Cloud's.
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
        if let (Some(exporter), Some(key)) =
            (self.inner.tracer.exporter(), self.inner.job_key.get())
        {
            let cloud = observability && self.inner.cloud.is_some();
            // Metrics follow the recording as a whole, as 1.8.10's process meter did. The only
            // logs are the session report's records, which the recording options select.
            let recording = telemetry::CloudRecording {
                traces: cloud && options.traces,
                logs: cloud,
                metrics: cloud && uploads,
                redaction,
                simulation: self.inner.simulation.get().cloned().flatten(),
            };
            exporter.register_job(key, recording);
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

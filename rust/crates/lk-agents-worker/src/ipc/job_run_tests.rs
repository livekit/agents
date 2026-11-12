#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use super::*;
use crate::ipc::ChildChannel;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::RefRegistry;
use lk_agents_abi::telemetry::policy::UPLOAD_TIMEOUT_SECONDS;
use lk_agents_abi::telemetry::record::{Record, RecordSink, SpanContext, SpanRecord, Status};
use lk_agents_abi::worker::ipc_host::JobRun as _;
use lk_agents_abi::worker::ipc_host::JobTask;
use lk_agents_abi::worker::job::{ShutdownCause, StartJob};
use lk_agents_telemetry::Tracer;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Default)]
struct Objects {
    next: AtomicU64,
    values: Mutex<BTreeMap<u64, String>>,
}
impl RefRegistry for Objects {
    fn release(&self, id: u64) {
        self.values.lock().remove(&id);
    }
}
impl Objects {
    fn insert(self: &Arc<Self>, value: String) -> HostRef {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.values.lock().insert(id, value);
        let registry: Arc<dyn RefRegistry> = self.clone();
        HostRef::new(id, &registry)
    }
}

struct Host {
    objects: Arc<Objects>,
    segment: Duration,
    reporting: tokio::sync::Notify,
    cancelled: Arc<AtomicBool>,
    events: Mutex<Vec<&'static str>>,
    /// Each step and the span current in it.
    calls: Mutex<Vec<(&'static str, Option<SpanContext>)>>,
}

#[derive(Default)]
struct Spans(Mutex<Vec<SpanRecord>>);
impl RecordSink for Spans {
    fn record(&self, record: Record) {
        if let Record::Span(span) = record {
            self.0.lock().push(span);
        }
    }
}
struct ReportCleanup(Arc<AtomicBool>, bool);
impl Drop for ReportCleanup {
    fn drop(&mut self) {
        if self.1 {
            self.0.store(true, Ordering::Release);
        }
    }
}
/// An entrypoint that returns at once.
struct Entrypoint;

impl JobTask for Entrypoint {
    fn wait<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn cancel<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
}

impl JobProcess for Host {
    fn ping(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn start_job(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: StartJob,
    ) -> BoxFuture<'_, Result<Arc<dyn JobExecution>, HostError>> {
        unreachable!("this test starts the admitted job directly")
    }
    fn start_entrypoint<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<Arc<dyn JobTask>, HostError>> {
        Box::pin(async { Ok(Arc::new(Entrypoint) as Arc<dyn JobTask>) })
    }
    fn close_session<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn session_end<'a>(
        &'a self,
        span: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.calls.lock().push(("session_end", span.span()));
        Box::pin(async { Ok(()) })
    }
    fn session_report<'a>(
        &'a self,
        span: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<Option<lk_agents_abi::telemetry::report::SessionReport>, HostError>>
    {
        self.calls.lock().push(("session_report", span.span()));
        Box::pin(async {
            let mut cleanup = ReportCleanup(self.cancelled.clone(), true);
            self.reporting.notify_one();
            tokio::time::sleep(self.segment).await;
            self.events.lock().push("prepared");
            tokio::time::sleep(self.segment).await;
            self.events.lock().push("uploaded");
            cleanup.1 = false;
            Ok(None)
        })
    }
    fn disconnect<'a>(
        &'a self,
        span: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.calls.lock().push(("disconnect", span.span()));
        Box::pin(async {
            self.events.lock().push("disconnected");
            Ok(())
        })
    }
    fn shutdown_callback<'a>(
        &'a self,
        span: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
        _: HostRef,
        _: Option<String>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.calls.lock().push(("shutdown_callback", span.span()));
        Box::pin(async {
            self.events.lock().push("callback");
            Ok(())
        })
    }
}

#[tokio::test(start_paused = true)]
async fn report_preparation_and_upload_share_one_budget_and_shutdown_always_continues() {
    crate::ipc::tests::logs::capture();
    for (segment, late) in
        [(UPLOAD_TIMEOUT_SECONDS * 2 / 3, true), (UPLOAD_TIMEOUT_SECONDS / 4, false)]
    {
        let host = Arc::new(Host {
            objects: Arc::default(),
            segment: Duration::from_secs(segment),
            reporting: tokio::sync::Notify::new(),
            cancelled: Arc::default(),
            events: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
        });
        let spans = Arc::new(Spans::default());
        let clock: Arc<dyn lk_agents_abi::Clock> = Arc::new(lk_agents_abi::VirtualClock::new());
        let env = ProcessEnv {
            tracer: Tracer::new(spans.clone(), clock.clone(), 1),
            ..ProcessEnv::current_runtime()
        };
        let run = JobRun::new(
            &env,
            true,
            5.0,
            false,
            "ws://localhost".into(),
            Some("on_end".into()),
            5.0,
            String::new(),
            String::new(),
        )
        .unwrap();
        run.add_shutdown_callback(
            host.objects.insert("callback".into()),
            false,
            false,
            "done".into(),
        );
        let closed = Arc::new(AtomicBool::new(false));
        let close = {
            let closed = closed.clone();
            move || -> BoxFuture<'static, ()> {
                Box::pin(async move { closed.store(true, Ordering::Release) })
            }
        };
        JobExecution::add_framework_shutdown(
            &run,
            "VoiceSession.aclose".into(),
            FrameworkShutdown::Native(Box::new(close)),
        );
        run.shutdown("test".into());
        let running = tokio::spawn({
            let host = host.clone();
            async move { run.run(host, CallContext::default()).await }
        });
        host.reporting.notified().await;
        tokio::time::timeout(Duration::from_secs(UPLOAD_TIMEOUT_SECONDS + 1), running)
            .await
            .expect("report callback restarted the job's budget")
            .unwrap()
            .unwrap();
        assert_eq!(host.cancelled.load(Ordering::Acquire), late);
        assert!(closed.load(Ordering::Acquire), "the shutdown runs the framework's native close");
        let events = host.events.lock().clone();
        assert_eq!(
            events,
            if late {
                vec!["prepared", "disconnected", "callback"]
            } else {
                vec!["prepared", "uploaded", "disconnected", "callback"]
            }
        );
        let spans = spans.0.lock().clone();
        let named = |name: &str| spans.iter().find(|s| s.name == name).unwrap().clone();
        let (job, shutdown, upload) =
            (named("job_entrypoint"), named("job_shutdown"), named("session_end_upload"));
        assert_eq!(shutdown.parent, Some(job.context.span_id));
        let calls = host.calls.lock().clone();
        // `on_session_end` runs first; the report it read is the one the outputs read.
        let order: Vec<&str> = calls.iter().map(|(c, _)| *c).collect();
        assert_eq!(order[..2], ["session_end", "session_report"]);
        for (step, call) in [
            ("on_session_end", "session_end"),
            ("session_end_upload", "session_report"),
            ("room_disconnect", "disconnect"),
            ("shutdown_callback", "shutdown_callback"),
        ] {
            let span = named(step);
            assert_eq!(span.parent, Some(shutdown.context.span_id), "{step}");
            let current = calls.iter().find(|(c, _)| *c == call).and_then(|(_, s)| *s);
            assert_eq!(current, Some(span.context), "{step} ran outside its span");
        }
        assert_eq!(upload.status != Status::Unset, late, "a late upload records the timeout");
        if late {
            assert!(
                crate::ipc::tests::logs::lines()
                    .iter()
                    .any(|line| line.message.contains("session_end_upload timed out"))
            );
        }
        assert!(host.objects.values.lock().is_empty(), "shutdown retained a host object");
    }
}

#[tokio::test]
async fn the_application_ends_the_job_even_when_its_session_closed_first() {
    use lk_agents_abi::worker::ipc_host::{JobExecution, JobRun as _};
    let run = JobRun::new(
        &ProcessEnv::current_runtime(),
        true,
        5.0,
        false,
        "ws://localhost".into(),
        None,
        5.0,
        String::new(),
        String::new(),
    )
    .unwrap();
    run.session_closed("user_initiated".into());
    run.end("shutdown".into(), ShutdownCause::Application);
    let st = run.inner.state.lock();
    assert_eq!(st.shutdown.as_deref(), Some("user_initiated"));
    assert_eq!(st.cause, Some(ShutdownCause::Application));
}

/// Only the simulator leaving ends the job, as a completed simulation.
/// https://github.com/livekit/agents/pull/6037
#[tokio::test(start_paused = true)]
async fn the_simulator_leaving_ends_the_job_as_a_completed_simulation() {
    use lk_agents_abi::ATTRIBUTE_SIMULATOR;
    use lk_agents_abi::telemetry::record::Value;
    let host = Arc::new(Host {
        objects: Arc::default(),
        segment: Duration::ZERO,
        reporting: tokio::sync::Notify::new(),
        cancelled: Arc::default(),
        events: Mutex::new(Vec::new()),
        calls: Mutex::new(Vec::new()),
    });
    let spans = Arc::new(Spans::default());
    let clock: Arc<dyn lk_agents_abi::Clock> = Arc::new(lk_agents_abi::VirtualClock::new());
    let env = ProcessEnv {
        tracer: Tracer::new(spans.clone(), clock, 1),
        ..ProcessEnv::current_runtime()
    };
    let run = Arc::new(
        JobRun::new(
            &env,
            true,
            5.0,
            false,
            "ws://localhost".into(),
            None,
            5.0,
            String::new(),
            String::new(),
        )
        .unwrap(),
    );
    run.add_shutdown_callback(host.objects.insert("callback".into()), false, false, "cb".into());
    let running = tokio::spawn({
        let (run, host) = (run.clone(), host.clone());
        async move { run.run(host, CallContext::default()).await }
    });

    run.participant_left(BTreeMap::from([("lk.agent.state".into(), "listening".into())]));
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(!running.is_finished(), "a participant who is not the simulator ended the job");

    run.participant_left(BTreeMap::from([(ATTRIBUTE_SIMULATOR.into(), "true".into())]));
    tokio::time::timeout(Duration::from_secs(60), running).await.unwrap().unwrap().unwrap();
    let shutdown = spans.0.lock().iter().find(|span| span.name == "job_shutdown").cloned();
    let cause =
        shutdown.unwrap().attributes.into_iter().find(|(key, _)| key == "lk.shutdown.cause");
    assert_eq!(cause.map(|(_, value)| value), Some(Value::String("simulation_completed".into())));
    assert_eq!(*host.events.lock(), ["prepared", "uploaded", "disconnected", "callback"]);
}

/// A job process run as a thread over a real channel: `ChildChannel` serves it, its job is a
/// real `JobRun`, and its entrypoint raises `crash` or returns. Shutdown callbacks record their
/// reason.
struct JobThread {
    me: std::sync::Weak<JobThread>,
    crash: Option<&'static str>,
    objects: Arc<Objects>,
    reasons: Arc<Mutex<Vec<Option<String>>>>,
    client: OnceLock<Arc<ChildChannel>>,
}

/// The entrypoint the job thread runs: it raises its error, or returns.
struct Settles(Option<&'static str>);

impl JobTask for Settles {
    fn wait<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        let settled = self.0.map_or(Ok(()), |crash| Err(HostError::new(crash)));
        Box::pin(async move { settled })
    }
    fn cancel<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
}

impl lk_agents_abi::worker::ipc_host::ChildInitialize for JobThread {
    fn initialize(&self, _: Option<String>) -> Result<(), HostError> {
        Ok(())
    }
}

impl JobProcess for JobThread {
    fn ping(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    /// As `_JobProc._core_start_job`: the job task runs the job, then stops the client.
    fn start_job(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: StartJob,
    ) -> BoxFuture<'_, Result<Arc<dyn JobExecution>, HostError>> {
        Box::pin(async move {
            let env = ProcessEnv::current_runtime();
            let run = JobRun::new(
                &env,
                true,
                5.0,
                false,
                "ws://localhost".into(),
                None,
                5.0,
                String::new(),
                String::new(),
            )
            .map_err(|error| HostError::new(error.to_string()))?;
            run.add_shutdown_callback(
                self.objects.insert("record".into()),
                true,
                false,
                "r".into(),
            );
            let run = Arc::new(run);
            let host: Arc<dyn JobProcess> = self.me.upgrade().expect("the thread holds its host");
            let client = self.client.get().expect("the client serves the job").clone();
            let running = run.clone();
            tokio::spawn(async move {
                let _ = running.run(host, CallContext::default()).await;
                running.close_tasks().await;
                lk_agents_abi::worker::ipc_host::IpcClient::stop(&*client);
            });
            Ok(run as Arc<dyn JobExecution>)
        })
    }
    fn start_entrypoint<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<Arc<dyn JobTask>, HostError>> {
        let entry: Arc<dyn JobTask> = Arc::new(Settles(self.crash));
        Box::pin(async move { Ok(entry) })
    }
    fn close_session<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn session_end<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn session_report<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<Option<lk_agents_abi::telemetry::report::SessionReport>, HostError>>
    {
        Box::pin(async { Ok(None) })
    }
    fn disconnect<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn shutdown_callback<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a CallContext,
        _: HostRef,
        reason: Option<String>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.reasons.lock().push(reason);
        Box::pin(async { Ok(()) })
    }
}

/// Launches [`JobThread`]s the way `JobLauncher` launches a thread runner.
struct ThreadLauncher {
    crash: Option<&'static str>,
    reasons: Arc<Mutex<Vec<Option<String>>>>,
}

impl lk_agents_abi::worker::ipc_host::JobLauncher for ThreadLauncher {
    fn launch<'a>(
        &'a self,
        _: &telemetry::Cx,
        _: &'a CallContext,
        endpoint: i64,
        launch: Arc<dyn lk_agents_abi::worker::ipc_host::LaunchReport>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        use lk_agents_abi::worker::ipc_host::{IpcClient, ThreadLaunch as _};
        let handle = tokio::runtime::Handle::current();
        let child = Arc::new_cyclic(|me| JobThread {
            me: me.clone(),
            crash: self.crash,
            objects: Arc::default(),
            reasons: self.reasons.clone(),
            client: OnceLock::new(),
        });
        std::thread::spawn(move || {
            let _entered = handle.enter();
            let env = ProcessEnv::current_runtime();
            let client = ChildChannel::new(&env, endpoint, child.clone(), false).unwrap();
            let client = Arc::new(client);
            let _ = child.client.set(client.clone());
            handle.block_on(async {
                let _ = client.serve(child.clone()).await;
                client.aclose().await;
            });
            let launch: Arc<dyn std::any::Any + Send + Sync> = launch;
            launch.downcast::<crate::ipc::ThreadLaunch>().unwrap().thread_exited();
        });
        Box::pin(async { Ok(()) })
    }

    fn completed<'a>(
        &'a self,
        _: &telemetry::Cx,
        _: &'a CallContext,
        _: lk_agents_abi::worker::ipc::JobCompletion,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
}

/// A job whose entrypoint raises ends as failed, not as a success.
/// https://github.com/livekit/agents/issues/8445
#[tokio::test(flavor = "multi_thread")]
#[ignore = "https://github.com/livekit/agents/issues/8445 reproduces"]
async fn a_raising_entrypoint_fails_the_job() {
    use lk_agents_abi::worker::ipc::{JobRunnerOptions, JobStatus};
    use lk_agents_abi::worker::ipc_host::JobSupervisor as _;
    let reasons = Arc::new(Mutex::new(Vec::new()));
    let launcher =
        Arc::new(ThreadLauncher { crash: Some("entrypoint raised"), reasons: reasons.clone() });
    let options = JobRunnerOptions {
        thread: true,
        initialize_timeout: 5.0,
        close_timeout: 5.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 2.5,
        ping_timeout: 60.0,
        high_ping_threshold: 0.5,
        http_proxy: None,
        asyncio_debug: false,
    };
    let env = ProcessEnv::current_runtime();
    let sup =
        crate::ipc::JobSupervisor::new(&env, launcher, options, CallContext::default()).unwrap();
    let limit = Duration::from_secs(10);
    tokio::time::timeout(limit, sup.start()).await.unwrap().unwrap();
    tokio::time::timeout(limit, sup.initialize()).await.unwrap().unwrap();

    let job = crate::ipc::pool::tests::job("j1");
    let launched = sup.launch_job(job, CallContext::default(), telemetry::Cx::NONE);
    tokio::time::timeout(limit, launched).await.unwrap().unwrap();
    assert!(tokio::time::timeout(limit, sup.join()).await.unwrap());

    assert_eq!(*reasons.lock(), [Some("job crashed".to_owned())], "the crash ended the job");
    assert_eq!(sup.status(), Some(JobStatus::Failed));
}

/// A job the worker shuts down without a reason ends with a stated one, and cleanly.
/// https://github.com/livekit/agents/pull/6284
#[tokio::test(flavor = "multi_thread")]
async fn a_job_shut_down_by_its_worker_states_why_and_succeeds() {
    use lk_agents_abi::worker::ipc::{JobRunnerOptions, JobStatus};
    use lk_agents_abi::worker::ipc_host::JobSupervisor as _;
    let reasons = Arc::new(Mutex::new(Vec::new()));
    let launcher = Arc::new(ThreadLauncher { crash: None, reasons: reasons.clone() });
    let options = JobRunnerOptions {
        thread: true,
        initialize_timeout: 5.0,
        close_timeout: 5.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 2.5,
        ping_timeout: 60.0,
        high_ping_threshold: 0.5,
        http_proxy: None,
        asyncio_debug: false,
    };
    let env = ProcessEnv::current_runtime();
    let sup =
        crate::ipc::JobSupervisor::new(&env, launcher, options, CallContext::default()).unwrap();
    let limit = Duration::from_secs(10);
    tokio::time::timeout(limit, sup.start()).await.unwrap().unwrap();
    tokio::time::timeout(limit, sup.initialize()).await.unwrap().unwrap();
    let job = crate::ipc::pool::tests::job("j1");
    let launched = sup.launch_job(job, CallContext::default(), telemetry::Cx::NONE);
    tokio::time::timeout(limit, launched).await.unwrap().unwrap();

    tokio::time::timeout(limit, sup.aclose()).await.unwrap();
    assert!(tokio::time::timeout(limit, sup.join()).await.unwrap());

    assert_eq!(*reasons.lock(), [Some("parent process shutdown".to_owned())]);
    assert_eq!(sup.status(), Some(JobStatus::Success));
}

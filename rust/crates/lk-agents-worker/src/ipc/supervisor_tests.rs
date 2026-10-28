use super::*;

fn options() -> SupervisorOptions {
    SupervisorOptions {
        pid: 42,
        initialize_timeout: 1.0,
        close_timeout: 2.0,
        memory_warn_mb: 0.0,
        memory_limit_mb: 0.0,
        ping_interval: 0.5,
        ping_timeout: 1.0,
        high_ping_threshold: 0.5,
        killable: true,
    }
}

#[test]
fn due_health_deadline_precedes_a_same_instant_pong() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 10_000, Input::Supervise);
    let effects = state.handle(
        Duration::from_secs(1),
        11_000,
        Input::Message(ToParent::Pong { sent_ms: 11_000 }),
    );
    assert!(effects.iter().any(|e| matches!(e, Effect::BeginKill)));
    assert!(state.pong_deadline.is_none(), "a same-instant pong cannot revive health");
}

#[test]
fn writes_are_correlated_and_a_stale_failure_does_not_kill_the_process() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.initialized = true;
    let effects = state.handle(
        Duration::ZERO,
        10_000,
        Input::StartJob { request_id: 1, job: Box::default() },
    );
    let id = effects
        .iter()
        .find_map(|e| match e {
            Effect::Send { id, .. } => Some(*id),
            _ => None,
        })
        .unwrap();
    state.handle(Duration::ZERO, 10_000, Input::Written { id, result: Ok(()) });
    let stale = state.handle(
        Duration::ZERO,
        10_000,
        Input::Written {
            id,
            result: Err(Arc::new(WireError::Unexpected("stale write failure".into()))),
        },
    );
    assert!(!stale.iter().any(|e| matches!(e, Effect::BeginKill)));
    let effects = state.handle(Duration::ZERO, 10_000, Input::Shutdown);
    let id = effects
        .iter()
        .find_map(|e| match e {
            Effect::Send { id, .. } => Some(*id),
            _ => None,
        })
        .unwrap();
    let failed = state.handle(
        Duration::ZERO,
        10_000,
        Input::Written {
            id,
            result: Err(Arc::new(WireError::Io(std::io::Error::from(
                std::io::ErrorKind::BrokenPipe,
            )))),
        },
    );
    assert!(
        failed.iter().any(|e| matches!(e, Effect::BeginKill)),
        "an accepted write failure starts native cleanup"
    );
}

#[test]
fn process_reap_deadline_retains_ownership_after_os_exit() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Exited(Some(0)));
    assert!(state.exited());
    assert!(!state.reaped(), "OS exit alone is not a reaping receipt");
    let overdue = state.tick(Duration::from_secs(2), 2_000);
    assert!(overdue.iter().any(|e| matches!(
        e,
        Effect::IncompleteCleanup(CleanupObservation { pid: 42, exited: true, reaped: false })
    )));
    assert!(state.tick(Duration::from_secs(3), 3_000).is_empty(), "deadline reports once");
    state.handle(Duration::from_secs(4), 4_000, Input::Reaped(0));
    assert!(state.reaped());
    assert!(state.wake_at().is_none());
}

#[test]
fn cancelled_launch_is_terminated_when_the_process_appears() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Kill);
    state.handle(Duration::ZERO, 0, Input::KillBegun(None));
    let launched = state.handle(Duration::from_millis(10), 10, Input::Launched(43));
    assert!(launched.iter().any(|e| matches!(e, Effect::BeginKill)));
    assert!(!state.kill_settled(), "a late launch still has a termination obligation");
    state.handle(Duration::from_millis(20), 20, Input::KillBegun(Some(false)));
    let failed = state.handle(
        Duration::from_millis(20),
        20,
        Input::KillFinished(Err("permission denied".into())),
    );
    assert!(failed.iter().any(|e| matches!(e,
        Effect::SignalFailed(SignalFailure { pid: 43, cause }) if cause == "permission denied")));
    assert!(!state.killed(), "failed signalling does not establish process death");
    assert!(!state.kill_settled());
}

#[test]
fn reaping_receipt_without_os_exit_does_not_release_ownership() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Reaped(0));
    assert!(!state.reaped());
    state.handle(Duration::from_secs(1), 1_000, Input::Exited(Some(0)));
    assert!(state.reaped());
}

#[test]
fn a_launched_process_cannot_wait_indefinitely_for_the_host_to_initialize_it() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Launched(43));
    let deadline = state.tick(Duration::from_secs(1), 1_000);
    assert!(deadline.iter().any(|effect| matches!(effect, Effect::BeginKill)));
}

#[test]
fn a_thread_transport_failure_or_startup_timeout_never_requests_an_os_kill() {
    let mut opts = options();
    opts.killable = false;
    opts.pid = 0;
    let mut state = Supervision::new(opts.clone(), Duration::ZERO);
    let failed = state.handle(
        Duration::ZERO,
        0,
        Input::ReaderClosed(Some(Arc::new(WireError::Io(std::io::Error::from(
            std::io::ErrorKind::BrokenPipe,
        ))))),
    );
    assert!(!failed.iter().any(|effect| matches!(effect, Effect::BeginKill | Effect::Kill)));
    let mut startup = Supervision::new(opts, Duration::ZERO);
    startup.handle(Duration::ZERO, 0, Input::Launched(0));
    let expired = startup.tick(Duration::from_secs(1), 1_000);
    assert!(!expired.iter().any(|effect| matches!(effect, Effect::BeginKill | Effect::Kill)));
}

#[test]
fn post_spawn_launch_failure_keeps_termination_and_reaping_obligations() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Launched(43));
    let failed =
        state.handle(Duration::ZERO, 0, Input::LaunchFailed("bootstrap pipe broke".into()));
    assert!(failed.iter().any(|effect| matches!(effect, Effect::BeginKill)));
    assert!(!state.reaped());
    let overdue = state.tick(Duration::from_secs(3), 3_000);
    assert!(overdue.iter().any(|effect| matches!(effect, Effect::IncompleteCleanup(_))));
}

#[test]
fn launch_rejection_without_a_process_is_not_reported_as_os_exit_or_reaping() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::LaunchRejected("spawn refused".into()));
    assert!(state.cleaned());
    assert!(!state.exited());
    assert!(!state.reaped());
    assert!(state.wake_at().is_none());
}

#[test]
fn kill_grace_is_included_before_the_reaping_deadline() {
    let mut opts = options();
    opts.close_timeout = 0.01;
    let mut state = Supervision::new(opts, Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Kill);
    state.handle(Duration::ZERO, 0, Input::KillBegun(Some(true)));
    let during_grace = state.tick(Duration::from_millis(20), 20);
    assert!(!during_grace.iter().any(|effect| matches!(effect, Effect::IncompleteCleanup(_))));
    let kill = state.tick(KILL_GRACE, 500);
    assert!(kill.iter().any(|effect| matches!(effect, Effect::Kill)));
    assert!(!kill.iter().any(|effect| matches!(effect, Effect::IncompleteCleanup(_))));
}

#[test]
fn unknown_os_exit_status_waits_for_the_start_methods_receipt_before_logging_failure() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    let exited = state.handle(Duration::ZERO, 0, Input::Exited(None));
    assert!(!exited.iter().any(|effect| matches!(effect, Effect::Log(Level::Error, _, _))));
    let clean = state.handle(Duration::ZERO, 0, Input::Reaped(0));
    assert!(!clean.iter().any(|effect| matches!(effect, Effect::Log(Level::Error, _, _))));
    let mut failed = Supervision::new(options(), Duration::ZERO);
    failed.handle(Duration::ZERO, 0, Input::Exited(None));
    let receipt = failed.handle(Duration::ZERO, 0, Input::Reaped(7));
    assert!(receipt.iter().any(|effect| matches!(effect,
        Effect::Log(Level::Error, message, _) if message.ends_with("exit code 7"))));
}

#[test]
fn ipc_version_mismatch_starts_cleanup_and_never_accepts_a_job() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(
        Duration::ZERO,
        0,
        Input::Initialize(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: None,
        }),
    );
    let refused = state.handle(
        Duration::ZERO,
        0,
        Input::Message(ToParent::Ready { version: PROTOCOL_VERSION + 1, error: None }),
    );
    assert!(
        refused.iter().any(|effect| matches!(effect,
        Effect::Initialized(Err(InitError::Version(version))) if *version == PROTOCOL_VERSION + 1))
    );
    assert!(refused.iter().any(|effect| matches!(effect, Effect::BeginKill)));
    let submitted =
        state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 1, job: Box::default() });
    assert!(
        !submitted
            .iter()
            .any(|effect| matches!(effect, Effect::Send { message: ToChild::StartJob { .. }, .. }))
    );
}

#[test]
fn work_is_refused_until_a_compatible_ready_reply() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    let submitted =
        state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 1, job: Box::default() });
    assert!(
        !submitted
            .iter()
            .any(|effect| matches!(effect, Effect::Send { message: ToChild::StartJob { .. }, .. }))
    );
}

#[test]
fn ipc_protocol_failure_after_native_exit_keeps_the_original_cause() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(
        Duration::ZERO,
        0,
        Input::Initialize(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: None,
        }),
    );
    state.handle(Duration::ZERO, 0, Input::Exited(Some(0)));
    state.handle(Duration::ZERO, 0, Input::Reaped(0));
    assert!(!state.cleaned(), "the socket can still contain the failure preceding exit");
    let effects = state.handle(
        Duration::ZERO,
        0,
        Input::ReaderClosed(Some(Arc::new(WireError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "partial Ready frame",
        ))))),
    );
    assert!(effects.iter().any(|effect| matches!(effect,
        Effect::Initialized(Err(InitError::Protocol(error))) if matches!(error.as_ref(),
            WireError::Io(cause) if cause.kind() == std::io::ErrorKind::UnexpectedEof && cause.to_string() == "partial Ready frame"))));
    assert!(matches!(state.protocol_failure(), Some(WireError::Io(_))));
    assert!(state.cleaned());
}

#[test]
fn start_job_admission_ack_correlates_acceptance_and_rejects_duplicate_work() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.initialized = true;
    let sent =
        state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 7, job: Box::default() });
    assert!(sent.iter().any(|effect| matches!(
        effect,
        Effect::Send { message: ToChild::StartJob { request_id: 7, .. }, .. }
    )));
    assert!(!state.running_job, "writing a request does not establish admission");
    let duplicate =
        state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 8, job: Box::default() });
    assert!(duplicate.iter().any(|effect| matches!(
        effect,
        Effect::JobStarted { request_id: 8, result: Err(StartJobError::AlreadyRunning) }
    )));
    let accepted = state.handle(
        Duration::ZERO,
        0,
        Input::Message(ToParent::JobAccepted { request_id: 7, error: None }),
    );
    assert!(state.running_job);
    assert!(
        accepted
            .iter()
            .any(|effect| matches!(effect, Effect::JobStarted { request_id: 7, result: Ok(()) }))
    );
}

#[test]
fn pending_job_failure_is_not_replaced_by_a_zero_exit() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.initialized = true;
    state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 7, job: Box::default() });
    let failure = Arc::new(WireError::Malformed(postcard::Error::DeserializeBadEnum));
    let failed = state.handle(Duration::ZERO, 0, Input::ReaderClosed(Some(failure.clone())));
    assert!(failed.iter().any(|effect| matches!(effect,
        Effect::JobStarted { request_id: 7, result: Err(StartJobError::Protocol(error)) } if Arc::ptr_eq(error, &failure))));
    assert!(failed.iter().any(|effect| matches!(effect, Effect::BeginKill)));
    state.handle(Duration::ZERO, 0, Input::Exited(Some(0)));
    assert!(matches!(state.protocol_failure(), Some(WireError::Malformed(_))));
    assert!(!state.running_job);
}

#[test]
fn invalid_job_protobuf_and_timestamps_are_refused_at_the_launch_boundary() {
    let mut job = StartJob { fake_job: true, job: vec![0xff], ..StartJob::default() };
    assert!(
        matches!(validate_start_job(&job), Err(StartJobError::InvalidFields(reason)) if reason.contains("protobuf"))
    );
    job.job.clear();
    job.launched_at = f64::NAN;
    assert!(
        matches!(validate_start_job(&job), Err(StartJobError::InvalidFields(reason)) if reason.contains("launched_at"))
    );
    job.launched_at = -1.0;
    assert!(validate_start_job(&job).is_err());
    job.launched_at = 0.0;
    assert!(validate_start_job(&job).is_ok());
}

#[test]
fn a_version_refusal_queued_before_exit_keeps_its_typed_cause() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(
        Duration::ZERO,
        0,
        Input::Initialize(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: None,
        }),
    );
    state.handle(Duration::ZERO, 0, Input::Exited(Some(0)));
    state.handle(Duration::ZERO, 0, Input::Reaped(0));
    let refused = state.handle(
        Duration::ZERO,
        0,
        Input::Message(ToParent::Ready { version: PROTOCOL_VERSION + 1, error: None }),
    );
    assert!(
        refused.iter().any(|effect| matches!(effect,
        Effect::Initialized(Err(InitError::Version(version))) if *version == PROTOCOL_VERSION + 1))
    );
    assert!(
        matches!(state.protocol_failure(), Some(WireError::Version { theirs }) if *theirs == PROTOCOL_VERSION + 1)
    );
    assert!(state.cleaned());
}

#[test]
fn child_initiated_shutdown_uses_the_exit_deadline_instead_of_health_pings() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Supervise);
    state.handle(Duration::ZERO, 0, Input::Message(ToParent::ShuttingDown));
    let finalizing = state.tick(Duration::from_secs(1), 1_000);
    assert!(
        !finalizing.iter().any(|effect| matches!(
            effect,
            Effect::BeginKill | Effect::Send { message: ToChild::Ping { .. }, .. }
        )),
        "finalization is governed by close_timeout, not the old health deadline"
    );
    let overdue = state.tick(Duration::from_secs(2), 2_000);
    assert!(
        overdue.iter().any(|effect| matches!(effect, Effect::BeginKill)),
        "child-initiated shutdown still has a bounded exit phase"
    );
}

#[test]
fn an_in_flight_ping_write_after_orderly_shutdown_does_not_fail_the_job() {
    let mut state = Supervision::new(options(), Duration::ZERO);
    state.handle(Duration::ZERO, 0, Input::Supervise);
    let ping = state
        .tick(Duration::ZERO, 0)
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Send { id, message: ToChild::Ping { .. } } => Some(id),
            _ => None,
        })
        .unwrap();
    state.handle(Duration::ZERO, 0, Input::Message(ToParent::ShuttingDown));
    state.handle(
        Duration::ZERO,
        0,
        Input::Written {
            id: ping,
            result: Err(Arc::new(WireError::Io(std::io::Error::from(
                std::io::ErrorKind::BrokenPipe,
            )))),
        },
    );
    assert!(state.protocol_failure().is_none(), "the child already declared its orderly exit");

    let mut required = Supervision::new(options(), Duration::ZERO);
    let shutdown = required
        .handle(Duration::ZERO, 0, Input::Shutdown)
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Send { id, message: ToChild::Shutdown { .. } } => Some(id),
            _ => None,
        })
        .unwrap();
    required.handle(Duration::ZERO, 0, Input::Message(ToParent::ShuttingDown));
    required.handle(
        Duration::ZERO,
        0,
        Input::Written {
            id: shutdown,
            result: Err(Arc::new(WireError::Io(std::io::Error::from(
                std::io::ErrorKind::BrokenPipe,
            )))),
        },
    );
    assert!(
        matches!(required.protocol_failure(), Some(WireError::Io(_))),
        "required protocol write failures keep their original cause"
    );
}

#[test]
fn memory_warnings_include_the_job_baseline_and_rate_limit_growth() {
    let mut state = Supervision::new(
        SupervisorOptions { memory_warn_mb: 500.0, ping_timeout: 120.0, ..options() },
        Duration::ZERO,
    );
    state.handle(
        Duration::ZERO,
        0,
        Input::Initialize(Hello {
            version: PROTOCOL_VERSION,
            asyncio_debug: false,
            http_proxy: None,
            health_timeout_ms: None,
        }),
    );
    state.handle(
        Duration::ZERO,
        0,
        Input::Message(ToParent::Ready { version: PROTOCOL_VERSION, error: None }),
    );
    state.handle(Duration::ZERO, 0, Input::StartJob { request_id: 1, job: Box::default() });
    state.handle(
        Duration::ZERO,
        0,
        Input::Message(ToParent::JobAccepted { request_id: 1, error: None }),
    );
    state.handle(Duration::ZERO, 0, Input::Supervise);
    let mut warnings = Vec::new();
    for (seconds, memory) in [(0, 100.0), (5, 600.04), (10, 610.0), (15, 700.0)] {
        let effects = state.handle(
            Duration::from_secs(seconds),
            seconds as i64 * 1000,
            Input::Sampled(Sample::Mb(memory, "uss".into())),
        );
        assert!(!effects.iter().any(|effect| matches!(effect, Effect::BeginKill)));
        warnings.extend(effects.into_iter().filter_map(|effect| match effect {
            Effect::Log(Level::Warning, message, extra) => Some((message, extra)),
            _ => None,
        }));
    }
    let warning = "job process memory usage is above the warning threshold (advisory only, the process will not be terminated)";
    assert_eq!(warnings.len(), 2);
    assert_eq!(
        warnings[0],
        (
            warning.into(),
            vec![
                ("memory_usage_mb", ExtraValue::Float(600.0)),
                ("memory_metric", ExtraValue::Str("uss".into())),
                ("memory_warn_mb", ExtraValue::Float(500.0)),
                ("memory_limit_mb", ExtraValue::Float(0.0)),
                ("uptime", ExtraValue::Float(5.0)),
                ("has_running_job", ExtraValue::Bool(true)),
                ("baseline_memory_mb", ExtraValue::Float(100.0)),
                ("growth_memory_mb", ExtraValue::Float(500.0)),
            ]
        )
    );
    assert_eq!(warnings[1].0, warning);
}

#[test]
fn unavailable_or_interrupted_memory_sampling_stops_without_retrying() {
    for sample in [Sample::Unavailable("process no longer exists".into()), Sample::Interrupted] {
        let unavailable = matches!(sample, Sample::Unavailable(_));
        let mut state = Supervision::new(
            SupervisorOptions { memory_warn_mb: 500.0, ping_timeout: 120.0, ..options() },
            Duration::ZERO,
        );
        state.handle(Duration::ZERO, 0, Input::Supervise);
        let effects = state.handle(Duration::ZERO, 0, Input::Sampled(sample));
        assert_eq!(
            effects.iter().filter(|effect| matches!(effect, Effect::Log(..))).count(),
            usize::from(unavailable)
        );
        if unavailable {
            assert!(effects.iter().any(|effect| matches!(effect, Effect::Log(Level::Warning, message, _) if message.contains("process no longer exists"))));
        }
        assert!(
            !state
                .tick(Duration::from_secs(30), 30_000)
                .iter()
                .any(|effect| matches!(effect, Effect::SampleMemory))
        );
    }
}

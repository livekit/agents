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
    let effects = state.handle(Duration::ZERO, 10_000, Input::StartJob(Box::default()));
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
        Input::Written { id, result: Err("stale write failure".into()) },
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
        Input::Written { id, result: Err("broken pipe".into()) },
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
    let failed = state.handle(Duration::ZERO, 0, Input::ReaderClosed(Some("broken pipe".into())));
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

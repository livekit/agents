//! Supervision contracts over the socket: each pins a past bug of the process lifecycle.

use super::*;

/// Lets the supervisor's actor work through what is queued, on paused time.
async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}

fn short(close_timeout: f64) -> SupervisorOptions {
    SupervisorOptions { pid: 42, initialize_timeout: 1.0, close_timeout, ..opts() }
}

/// Initialized and ready to take a job.
async fn ready(sup: &ProcSupervisor, child: &mut DuplexStream) {
    let init = waits(sup).await;
    assert_eq!(recv(child).await, ToChild::Hello(hello()));
    write(child, &ToParent::Ready { version: PROTOCOL_VERSION, error: None }).await.unwrap();
    init.await.unwrap().unwrap();
}

fn errors(host: &FakeHost) -> Vec<String> {
    let logs = host.logs.lock().unwrap();
    logs.iter().filter(|(level, ..)| *level == Level::Error).map(|(_, m, _)| m.clone()).collect()
}

// Catches: OS exit alone taken as the reaping receipt, or the cleanup deadline reported twice.
#[tokio::test(start_paused = true)]
async fn process_reap_deadline_retains_ownership_after_os_exit() {
    let (sup, host, _child) = setup(short(1.0));
    sup.exited(Some(0)).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    settle().await;
    assert_eq!(
        *host.incomplete.lock().unwrap(),
        [CleanupObservation { pid: 42, exited: true, reaped: false }]
    );
    tokio::time::sleep(Duration::from_secs(2)).await;
    settle().await;
    assert_eq!(host.incomplete.lock().unwrap().len(), 1, "the deadline reports once");
    sup.reaped(0).await;
    assert!(sup.wait_cleaned().await);
}

// Catches: a launch cancelled before its process existed leaving the late process running, or
// a failed kill taken as the process's death.
#[tokio::test(start_paused = true)]
async fn cancelled_launch_is_terminated_when_the_process_appears() {
    let (sup, host, _child) = setup(short(60.0));
    host.alive.store(false, Ordering::SeqCst);
    sup.kill().await;
    host.alive.store(true, Ordering::SeqCst);
    *host.kill_error.lock().unwrap() = Some("permission denied".into());
    sup.launched(43).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the late process is killed");
    assert_eq!(
        *host.signal_failures.lock().unwrap(),
        [SignalFailure { pid: 43, cause: "permission denied".into() }]
    );
    assert!(!sup.killed(), "a failed signal does not establish the process's death");
}

// Catches: a reaping receipt with no OS exit releasing the process.
#[tokio::test(start_paused = true)]
async fn reaping_receipt_without_os_exit_does_not_release_ownership() {
    let (sup, _host, _child) = setup(short(60.0));
    let reaping = sup.clone();
    tokio::spawn(async move { reaping.reaped(0).await });
    settle().await;
    let cleaned = sup.wait_cleaned();
    tokio::pin!(cleaned);
    assert!(futures_util::poll!(&mut cleaned).is_pending());
    sup.exited(Some(0)).await;
    assert!(cleaned.await);
}

// Catches: a launched process the host never initializes living forever.
#[tokio::test(start_paused = true)]
async fn a_launched_process_cannot_wait_indefinitely_for_the_host_to_initialize_it() {
    let (sup, host, _child) = setup(short(60.0));
    sup.launched(43).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert!(
        errors(&host).contains(
            &"process launch did not enter initialization before its deadline".to_owned()
        )
    );
}

// Catches: a job thread (no process of its own) being sent an OS kill on a transport failure or
// a startup timeout.
#[tokio::test(start_paused = true)]
async fn a_thread_transport_failure_or_startup_timeout_never_requests_an_os_kill() {
    let thread = SupervisorOptions { killable: false, pid: 0, ..short(60.0) };
    let (sup, host, child) = setup(thread.clone());
    drop(child);
    settle().await;
    assert!(sup.completion(None).failure.is_some(), "the transport failure is recorded");
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
    let (sup, host, _child) = setup(thread);
    sup.launched(0).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 0);
}

// Catches: a launch that failed after spawning dropping its kill and reaping obligations.
#[tokio::test(start_paused = true)]
async fn post_spawn_launch_failure_keeps_termination_and_reaping_obligations() {
    let (sup, host, _child) = setup(short(1.0));
    host.dumps.store(true, Ordering::SeqCst);
    sup.launched(43).await;
    sup.launch_failed("bootstrap pipe broke".into()).await;
    settle().await;
    tokio::time::sleep(KILL_GRACE).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    tokio::time::sleep(Duration::from_secs(2)).await;
    settle().await;
    assert_eq!(host.incomplete.lock().unwrap().len(), 1, "unreaped past its deadline");
    assert!(!host.released.load(Ordering::SeqCst));
}

// Catches: a launch refused before any process existed reported as an OS exit or a reaping.
#[tokio::test(start_paused = true)]
async fn launch_rejection_without_a_process_is_not_reported_as_os_exit_or_reaping() {
    let (sup, host, _child) = setup(short(60.0));
    sup.not_launched("spawn refused".into()).await;
    assert!(!sup.wait_exit().await, "no OS exit");
    assert!(!sup.wait_cleaned().await, "no reaping");
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: the reaping deadline expiring inside the kill's stack-dump grace.
#[tokio::test(start_paused = true)]
async fn kill_grace_is_included_before_the_reaping_deadline() {
    let (sup, host, _child) = setup(short(0.01));
    host.dumps.store(true, Ordering::SeqCst);
    let killing = sup.clone();
    tokio::spawn(async move { killing.kill().await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    settle().await;
    assert!(host.incomplete.lock().unwrap().is_empty(), "inside the grace");
    tokio::time::sleep(KILL_GRACE - Duration::from_millis(20)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert!(host.incomplete.lock().unwrap().is_empty());
}

// Catches: an unknown exit status logged as a failure before the start method's receipt told it.
#[tokio::test(start_paused = true)]
async fn unknown_os_exit_status_waits_for_the_start_methods_receipt_before_logging_failure() {
    let (sup, host, _child) = setup(short(60.0));
    sup.exited(None).await;
    sup.reaped(0).await;
    assert!(errors(&host).is_empty(), "{:?}", errors(&host));
    let (sup, host, _child) = setup(short(60.0));
    sup.exited(None).await;
    sup.reaped(7).await;
    assert_eq!(errors(&host), ["process exited with non-zero exit code 7"]);
}

// Catches: a protocol failure read after the native exit lost, or its cause replaced.
#[tokio::test(start_paused = true)]
async fn ipc_protocol_failure_after_native_exit_keeps_the_original_cause() {
    let (sup, _host, mut child) = setup(short(60.0));
    let init = waits(&sup).await;
    assert_eq!(recv(&mut child).await, ToChild::Hello(hello()));
    sup.exited(Some(0)).await;
    let reaping = sup.clone();
    tokio::spawn(async move { reaping.reaped(0).await });
    settle().await;
    // the socket can still hold the failure that preceded the exit: a truncated Ready frame
    child.write_all(&100u32.to_be_bytes()).await.unwrap();
    child.write_all(&[1, 2]).await.unwrap();
    drop(child);
    assert!(matches!(init.await.unwrap(), Err(InitError::Protocol(_))));
    assert!(sup.wait_cleaned().await);
    assert!(sup.completion(Some(0)).failure.is_some(), "a zero exit keeps the IPC failure");
}

// Catches: a written start-job request taken as admitted, or a second job taken while one is
// pending.
#[tokio::test(start_paused = true)]
async fn start_job_admission_ack_correlates_acceptance_and_rejects_duplicate_work() {
    let (sup, _host, mut child) = setup(short(60.0));
    ready(&sup, &mut child).await;
    let first = sup.clone();
    let started = tokio::spawn(async move { first.start_job(job()).await });
    let ToChild::StartJob { request_id, .. } = recv_skipping_pings(&mut child).await else {
        panic!("no start-job request")
    };
    assert!(!started.is_finished(), "writing a request does not establish admission");
    assert!(matches!(sup.start_job(job()).await, Err(StartJobError::AlreadyRunning)));
    write(&mut child, &ToParent::JobAccepted { request_id, error: None }).await.unwrap();
    assert!(started.await.unwrap().is_ok());
}

// Catches: a job that failed on a protocol error reported clean because the process exited 0.
#[tokio::test(start_paused = true)]
async fn pending_job_failure_is_not_replaced_by_a_zero_exit() {
    let (sup, host, mut child) = setup(short(60.0));
    ready(&sup, &mut child).await;
    let first = sup.clone();
    let started = tokio::spawn(async move { first.start_job(job()).await });
    recv_skipping_pings(&mut child).await;
    // a frame whose message tag no version knows
    child.write_all(&1u32.to_be_bytes()).await.unwrap();
    child.write_all(&[0xff]).await.unwrap();
    assert!(matches!(started.await.unwrap(), Err(StartJobError::Protocol(_))));
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    sup.exited(Some(0)).await;
    assert!(sup.completion(Some(0)).failure.is_some());
}

// Catches: a version refusal read after the exit losing its typed cause.
#[tokio::test(start_paused = true)]
async fn a_version_refusal_queued_before_exit_keeps_its_typed_cause() {
    let (sup, _host, mut child) = setup(short(60.0));
    let init = waits(&sup).await;
    assert_eq!(recv(&mut child).await, ToChild::Hello(hello()));
    sup.exited(Some(0)).await;
    let reaping = sup.clone();
    tokio::spawn(async move { reaping.reaped(0).await });
    settle().await;
    let theirs = PROTOCOL_VERSION + 1;
    write(&mut child, &ToParent::Ready { version: theirs, error: None }).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(v)) if v == theirs));
    assert!(sup.wait_cleaned().await);
    assert!(sup.completion(Some(0)).failure.is_some());
}

// Catches: a child's own shutdown still judged by the health pings, or left unbounded.
#[tokio::test(start_paused = true)]
async fn child_initiated_shutdown_uses_the_exit_deadline_instead_of_health_pings() {
    let opts = SupervisorOptions { ping_interval: 0.25, ping_timeout: 0.5, ..short(2.0) };
    let (sup, host, mut child) = setup(opts);
    ready(&sup, &mut child).await;
    sup.supervise().await;
    write(&mut child, &ToParent::ShuttingDown).await.unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 0, "governed by close_timeout, not the pings");
    tokio::time::sleep(Duration::from_millis(1100)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the exit phase is still bounded");
}

// Catches: the memory warning missing the job's baseline and growth.
#[tokio::test(start_paused = true)]
async fn memory_warnings_include_the_job_baseline() {
    let opts = SupervisorOptions { memory_warn_mb: 500.0, ..short(60.0) };
    let (sup, host, mut child) = setup(opts);
    host.samples.lock().unwrap().extend([Ok((100.0, "uss".into())), Ok((600.04, "uss".into()))]);
    ready(&sup, &mut child).await;
    let first = sup.clone();
    let started = tokio::spawn(async move { first.start_job(job()).await });
    let ToChild::StartJob { request_id, .. } = recv_skipping_pings(&mut child).await else {
        panic!("no start-job request")
    };
    write(&mut child, &ToParent::JobAccepted { request_id, error: None }).await.unwrap();
    started.await.unwrap().unwrap();
    sup.supervise().await;
    let warning = "job process memory usage is above the warning threshold (advisory only, the process will not be terminated)";
    tokio::time::timeout(Duration::from_secs(30), async {
        while host.find(warning).is_none() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .unwrap();
    let (level, extra) = host.find(warning).unwrap();
    assert_eq!(level, Level::Warning);
    let field = |name: &str| extra.iter().find(|(key, _)| *key == name).unwrap().1.clone();
    assert_eq!(field("memory_usage_mb"), ExtraValue::Float(600.0));
    assert_eq!(field("baseline_memory_mb"), ExtraValue::Float(100.0));
    assert_eq!(field("growth_memory_mb"), ExtraValue::Float(500.0));
    assert_eq!(field("has_running_job"), ExtraValue::Bool(true));
}

/// What the child reads within `within`, pings skipped; `None` when it reads nothing else.
async fn next_within(child: &mut DuplexStream, within: Duration) -> Option<ToChild> {
    tokio::time::timeout(within, recv_skipping_pings(child)).await.ok()
}

// Catches: a pong read at the instant the health deadline expires reviving the process: what is
// due runs before the input.
#[tokio::test(start_paused = true)]
async fn the_due_health_deadline_precedes_a_same_instant_pong() {
    let opts = SupervisorOptions { ping_interval: 0.5, ping_timeout: 1.0, ..short(60.0) };
    let (sup, host, mut child) = setup(opts);
    sup.supervise().await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    write(&mut child, &ToParent::Pong { sent_ms: unix_now_ms() as u64 }).await.unwrap();
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert!(host.find("process is unresponsive, killing process").is_some());
}

// Catches: a job whose protobuf or timestamps are invalid reaching the child.
#[tokio::test(start_paused = true)]
async fn invalid_job_protobuf_and_timestamps_are_refused_at_the_launch_boundary() {
    use super::supervisor::validate_start_job as validate;
    let refused = |job: &StartJob, field: &str| {
        matches!(validate(job), Err(StartJobError::InvalidFields(reason)) if reason.contains(field))
    };
    let mut invalid = StartJob { job: vec![0xff], ..job() };
    assert!(refused(&invalid, "protobuf"));
    invalid.job.clear();
    invalid.launched_at = f64::NAN;
    assert!(refused(&invalid, "launched_at"));
    invalid.launched_at = -1.0;
    assert!(validate(&invalid).is_err());
    invalid.launched_at = 0.0;
    assert!(validate(&invalid).is_ok());

    let (sup, _host, mut child) = setup(short(60.0));
    ready(&sup, &mut child).await;
    let refused = sup.start_job(StartJob { job: vec![0xff], ..job() }).await;
    assert!(matches!(refused, Err(StartJobError::InvalidFields(_))));
    assert_eq!(next_within(&mut child, Duration::from_secs(1)).await, None);
}

// Catches: a job sent to a child that has not answered `Ready`.
#[tokio::test(start_paused = true)]
async fn work_is_refused_until_a_compatible_ready_reply() {
    let (sup, _host, mut child) = setup(short(60.0));
    let _init = waits(&sup).await;
    assert_eq!(recv(&mut child).await, ToChild::Hello(hello()));
    assert!(matches!(sup.start_job(job()).await, Err(StartJobError::NotReady)));
    assert_eq!(next_within(&mut child, Duration::from_millis(500)).await, None);
}

// Catches: a child that refused the protocol version still given a job.
#[tokio::test(start_paused = true)]
async fn no_job_is_accepted_after_a_version_refusal() {
    let (sup, host, mut child) = setup(short(60.0));
    let init = waits(&sup).await;
    assert_eq!(recv(&mut child).await, ToChild::Hello(hello()));
    write(&mut child, &ToParent::Ready { version: PROTOCOL_VERSION + 1, error: None })
        .await
        .unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(_))));
    assert!(sup.start_job(job()).await.is_err());
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the refusal starts the cleanup");
    assert!(!matches!(
        next_within(&mut child, Duration::from_millis(500)).await,
        Some(ToChild::StartJob { .. })
    ));
}

// Catches: a memory sample that failed because the process is gone (or was interrupted) being
// retried every interval.
#[tokio::test(start_paused = true)]
async fn unavailable_or_interrupted_memory_sampling_stops_without_retrying() {
    let unavailable = || MemoryError::Unavailable("process no longer exists".into());
    for failed in [unavailable(), MemoryError::Interrupted] {
        let logged = matches!(failed, MemoryError::Unavailable(_));
        let opts = SupervisorOptions { memory_warn_mb: 500.0, ping_timeout: 120.0, ..short(60.0) };
        let (sup, host, mut child) = setup(opts);
        // A retry would read the second sample and warn.
        host.samples.lock().unwrap().extend([Err(failed), Ok((600.0, "uss".into()))]);
        ready(&sup, &mut child).await;
        sup.supervise().await;
        let drain =
            tokio::spawn(async move { while read::<ToChild, _>(&mut child).await.is_ok() {} });
        tokio::time::sleep(Duration::from_secs(30)).await;
        settle().await;
        let messages = host.messages();
        let failures = messages.iter().filter(|m| m.contains("failed to get memory info")).count();
        assert_eq!(failures, usize::from(logged));
        assert!(!messages.iter().any(|m| m.contains("above the warning threshold")), "retried");
        drop(drain);
    }
}

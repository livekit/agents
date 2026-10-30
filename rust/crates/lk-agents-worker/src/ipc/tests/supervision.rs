//! Supervision contracts over the socket: each pins a past bug of the process lifecycle.

use super::*;

/// Lets the supervisor's actor work through what is queued, on paused time.
async fn settle() {
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
}

fn short(close_timeout: f64) -> SupervisorOptions {
    SupervisorOptions { initialize_timeout: 1.0, close_timeout, ..opts() }
}

/// Initialized and ready to take a job.
async fn ready(sup: &RunnerSupervisor, child: &Child) {
    let init = waits(sup).await;
    assert_eq!(recv(child).await, ToChild::Hello(hello()));
    write(child, &ToParent::Ready { error: None }).await.unwrap();
    init.await.unwrap().unwrap();
}

fn errors(host: &FakeHost) -> Vec<String> {
    let logs = host.logs.lock();
    logs.iter().filter(|(level, ..)| *level == Level::Error).map(|(_, m, _)| m.clone()).collect()
}

// Catches: a stopped runner that never ends released, or its missed cleanup reported twice.
#[tokio::test(start_paused = true)]
async fn a_runner_that_does_not_end_after_its_stop_is_reported_once_and_kept() {
    let (sup, host, _child) = setup(short(1.0));
    let killing = sup.clone();
    tokio::spawn(async move { killing.kill().await });
    tokio::time::sleep(Duration::from_secs(4)).await;
    settle().await;
    let incomplete = |host: &FakeHost| {
        host.messages()
            .iter()
            .filter(|m| *m == "incomplete process cleanup; retaining ownership")
            .count()
    };
    assert_eq!(incomplete(&host), 1, "the deadline reports once");
    assert!(!host.released.load(Ordering::SeqCst));
    finish(&sup, &host, -9).await;
    assert!(sup.wait_cleaned().await);
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: a kill requested while the launch was still running leaving the late runner running.
#[tokio::test(start_paused = true)]
async fn a_kill_during_the_launch_stops_the_runner_once_it_exists() {
    let host = Arc::new(FakeHost::default());
    let (parent, _child) = memory();
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let executor = FakeExecutor { gate: Some(gate.clone()), ..executor(parent, &host) };
    let sup = start(executor, &host, short(60.0));
    let killing = sup.clone();
    tokio::spawn(async move { killing.kill().await });
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 0, "nothing to stop yet");
    gate.add_permits(1);
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the late runner is stopped");
}

// Catches: a launched process the host never initializes living forever.
#[tokio::test(start_paused = true)]
async fn a_launched_process_cannot_wait_indefinitely_for_the_host_to_initialize_it() {
    let (_sup, host, _child) = setup(short(60.0));
    tokio::time::sleep(Duration::from_millis(1100)).await;
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    assert!(
        errors(&host).contains(
            &"process launch did not enter initialization before its deadline".to_owned()
        )
    );
}

// Catches: a launch refused before any runner existed reported as a runner that ended.
#[tokio::test(start_paused = true)]
async fn launch_rejection_without_a_process_is_not_reported_as_os_exit_or_reaping() {
    let host = Arc::new(FakeHost::default());
    let (parent, _child) = memory();
    let refusal = Some("spawn refused".into());
    let sup = start(FakeExecutor { refusal, ..executor(parent, &host) }, &host, short(60.0));
    let initialized = sup.initialize(hello()).await;
    assert!(matches!(initialized, Err(InitError::Failed(cause)) if cause == "spawn refused"));
    assert!(!sup.wait_exit().await, "no exit");
    assert!(!sup.wait_cleaned().await, "nothing ran");
    assert!(host.released.load(Ordering::SeqCst));
}

// Catches: a protocol failure read after the native exit lost, or its cause replaced.
#[tokio::test(start_paused = true)]
async fn ipc_protocol_failure_after_native_exit_keeps_the_original_cause() {
    let (sup, host, mut child) = setup_raw(short(60.0)).await;
    let init = waits(&sup).await;
    raw_hello(&mut child).await;
    finish(&sup, &host, 0).await;
    // the socket can still hold the failure that preceded the exit: a truncated Ready frame
    child.write_all(&100u32.to_be_bytes()).await.unwrap();
    child.write_all(&[1, 2]).await.unwrap();
    drop(child);
    assert!(matches!(init.await.unwrap(), Err(InitError::Protocol(_))));
    assert!(sup.wait_cleaned().await);
    assert!(sup.completion().failure.is_some(), "a zero exit keeps the IPC failure");
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
    let (sup, host, mut child) = setup_raw(short(60.0)).await;
    let init = waits(&sup).await;
    raw_hello(&mut child).await;
    let ready = postcard::to_allocvec(&ToParent::Ready { error: None }).unwrap();
    child.write_u32(ready.len() as u32).await.unwrap();
    child.write_all(&ready).await.unwrap();
    init.await.unwrap().unwrap();
    let first = sup.clone();
    let started = tokio::spawn(async move { first.start_job(job()).await });
    // a frame whose message tag no version knows
    child.write_all(&1u32.to_be_bytes()).await.unwrap();
    child.write_all(&[0xff]).await.unwrap();
    assert!(matches!(started.await.unwrap(), Err(StartJobError::Protocol(_))));
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1);
    finish(&sup, &host, 0).await;
    assert!(sup.completion().failure.is_some());
}

// Catches: a version refusal read after the exit losing its typed cause.
#[tokio::test(start_paused = true)]
async fn a_version_refusal_queued_before_exit_keeps_its_typed_cause() {
    let (parent, mut child) = tokio::io::duplex(1 << 16);
    let (sup, host) = supervise(Channel::new(parent), short(60.0));
    let init = waits(&sup).await;
    finish(&sup, &host, 0).await;
    let theirs = VERSION + 1;
    child.write_all(&MAGIC).await.unwrap();
    child.write_all(&theirs.to_be_bytes()).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(v)) if v == theirs));
    assert!(sup.wait_cleaned().await);
    assert!(sup.completion().failure.is_some());
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
    host.samples.lock().extend([Some(100.0), Some(600.04)]);
    ready(&sup, &mut child).await;
    let first = sup.clone();
    let started = tokio::spawn(async move { first.start_job(job()).await });
    let ToChild::StartJob { request_id, .. } = recv_skipping_pings(&mut child).await else {
        panic!("no start-job request")
    };
    write(&mut child, &ToParent::JobAccepted { request_id, error: None }).await.unwrap();
    started.await.unwrap().unwrap();
    sup.supervise().await;
    let warning = "process memory usage is above the warning threshold (advisory only, the process will not be stopped)";
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
async fn next_within(child: &Child, within: Duration) -> Option<ToChild> {
    tokio::time::timeout(within, recv_skipping_pings(child)).await.ok()
}

// Catches: a job whose protobuf or timestamps are invalid reaching the child.
#[tokio::test(start_paused = true)]
async fn invalid_job_protobuf_and_timestamps_are_refused_at_the_launch_boundary() {
    use super::supervisor::validate_start_job as validate;
    let refused = |job: &StartJob, field: &str| matches!(validate(job), Err(StartJobError::InvalidFields(reason)) if reason.contains(field));
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
    let (parent, mut child) = tokio::io::duplex(1 << 16);
    let (sup, host) = supervise(Channel::new(parent), short(60.0));
    let init = waits(&sup).await;
    child.write_all(&MAGIC).await.unwrap();
    child.write_all(&(VERSION + 1).to_be_bytes()).await.unwrap();
    assert!(matches!(init.await.unwrap(), Err(InitError::Version(_))));
    assert!(sup.start_job(job()).await.is_err());
    settle().await;
    assert_eq!(host.kills.load(Ordering::SeqCst), 1, "the refusal starts the cleanup");
}

// Catches: a runner with no readable footprint (a thread, a process gone) sampled every interval.
#[tokio::test(start_paused = true)]
async fn memory_sampling_stops_when_the_runner_has_no_footprint() {
    let opts = SupervisorOptions { memory_warn_mb: 500.0, ping_timeout: 120.0, ..short(60.0) };
    let (sup, host, mut child) = setup(opts);
    // A retry would read the second sample and warn.
    host.samples.lock().extend([None, Some(600.0)]);
    ready(&sup, &mut child).await;
    sup.supervise().await;
    let drain = tokio::spawn(async move { while child.recv().await.is_ok() {} });
    tokio::time::sleep(Duration::from_secs(30)).await;
    settle().await;
    assert!(!host.messages().iter().any(|m| m.contains("above the warning threshold")), "retried");
    drop(drain);
}

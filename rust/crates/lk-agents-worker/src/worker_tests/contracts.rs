//! The job dispatch contracts, driven through the worker's transport: each pins a past bug.

use super::*;
use livekit_protocol::worker_message::Message as Sent;

fn job_status(message: &protocol::WorkerMessage, id: &str) -> Option<protocol::JobStatus> {
    match &message.message {
        Some(Sent::UpdateJob(update)) if update.job_id == id => {
            protocol::JobStatus::try_from(update.status).ok()
        }
        _ => None,
    }
}

fn accepted(message: &protocol::WorkerMessage, id: &str) -> bool {
    matches!(&message.message, Some(Sent::Availability(reply)) if reply.job_id == id && reply.available)
}

type Run = tokio::task::JoinHandle<Result<(), RuntimeError>>;

/// Runs the rig's worker after it registered, with `decisions` decisions let through.
fn running(rig: RuntimeRig, decisions: usize) -> (Rig, Run) {
    let RuntimeRig {
        mut runtime,
        handle,
        peer,
        sent,
        started,
        decisions: gate,
        launches,
        connections,
        load,
        launched,
        ..
    } = rig;
    peer.send(register_message("worker")).unwrap();
    gate.add_permits(decisions);
    let task = tokio::spawn(async move { runtime.run().await });
    let rig =
        Rig { handle, peer, sent, started, decisions: gate, launches, connections, load, launched };
    (rig, task)
}

/// The rig's parts a running worker leaves to the test.
struct Rig {
    handle: WorkerClient,
    peer: Peer,
    sent: tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    started: tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    decisions: Arc<tokio::sync::Semaphore>,
    launches: Arc<tokio::sync::Semaphore>,
    connections: Connections,
    load: Arc<HostLoad>,
    launched: Arc<Mutex<Vec<crate::RunningJobInfo>>>,
}

impl Rig {
    /// The peer of the worker's next connection, which registers it.
    fn next_peer(&self) -> Peer {
        let (peer, incoming) = tokio::sync::mpsc::unbounded_channel();
        peer.send(register_message("worker")).unwrap();
        self.connections.lock().push_back(incoming);
        peer
    }

    fn set_load(&self, load: f32) {
        *self.load.value.lock() = load;
    }

    /// Offers `id`, lets its decision accept and waits for the acceptance on the wire.
    async fn accept(&mut self, id: &str) {
        self.peer.send(Ok(offer(id))).unwrap();
        self.decisions.add_permits(1);
        until_sent(&mut self.sent, |m| accepted(m, id)).await;
    }

    /// Accepts, assigns and launches `id`.
    async fn launch(&mut self, id: &str) {
        self.accept(id).await;
        self.launches.add_permits(1);
        self.peer.send(Ok(assignment_message(id))).unwrap();
        while self.launched.lock().iter().all(|info| info.job.id != id) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        while self.handle.snapshot().active_jobs == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

fn refused(message: &protocol::WorkerMessage, id: &str) -> bool {
    matches!(&message.message, Some(Sent::Availability(reply)) if reply.job_id == id && !reply.available)
}

fn status(message: &protocol::WorkerMessage) -> Option<&protocol::UpdateWorkerStatus> {
    match &message.message {
        Some(Sent::UpdateWorker(update)) => Some(update),
        _ => None,
    }
}

fn decisions_started(started: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>) -> usize {
    std::iter::from_fn(|| started.try_recv().ok()).filter(|step| *step == "decision").count()
}

async fn close(rig: Rig, task: Run) {
    rig.handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
}

// Catches: a repeated offer or assignment of one job launching it twice.
#[tokio::test(start_paused = true)]
async fn worker_registration_vectors_duplicate_availability_launches_once() {
    let fresh = runtime_rig();
    fresh.launches.add_permits(2);
    let (mut rig, task) = running(fresh, 2);
    rig.peer.send(Ok(offer("a"))).unwrap();
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| accepted(m, "a")).await;
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    assert_eq!(rig.started.recv().await, Some("decision"));
    assert_eq!(rig.started.recv().await, Some("launch"));
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(rig.started.try_recv().is_err(), "one decision and one launch");
    assert_eq!(rig.handle.snapshot().active_jobs, 1);
    close(rig, task).await;
}

// Catches: an assignment for a job this worker never accepted launched, or left unanswered.
#[tokio::test(start_paused = true)]
async fn worker_registration_vectors_failed_and_unknown_assignments_report_failure() {
    let fresh = runtime_rig();
    let (mut rig, task) = running(fresh, 0);
    rig.peer.send(Ok(assignment_message("ghost"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "ghost") == Some(protocol::JobStatus::JsFailed))
        .await;
    assert!(rig.started.try_recv().is_err(), "nothing launched");
    close(rig, task).await;
}

// Catches: an accepted job waiting forever for its assignment; it is dropped at 7.5 s and a late
// assignment is answered failed.
#[tokio::test(start_paused = true)]
async fn assignment_times_out_at_exactly_7_5s() {
    let fresh = runtime_rig();
    let (mut rig, task) = running(fresh, 1);
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| accepted(m, "a")).await;
    tokio::time::sleep(ASSIGNMENT_TIMEOUT + Duration::from_millis(100)).await;
    assert_eq!(rig.handle.snapshot().reserved_slots, 0, "the slot is released");
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "a") == Some(protocol::JobStatus::JsFailed)).await;
    assert_eq!(rig.started.recv().await, Some("decision"));
    assert!(rig.started.try_recv().is_err(), "the late assignment launches nothing");
    close(rig, task).await;
}

// Catches: a failed launch keeping its slot or leaving the server without the job's failure.
#[tokio::test(start_paused = true)]
async fn a_failed_launch_releases_the_slot() {
    let fresh = runtime_rig();
    fresh.failing_launches.store(true, std::sync::atomic::Ordering::SeqCst);
    fresh.launches.add_permits(1);
    let (mut rig, task) = running(fresh, 1);
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| accepted(m, "a")).await;
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "a") == Some(protocol::JobStatus::JsFailed)).await;
    let snapshot = rig.handle.snapshot();
    assert_eq!((snapshot.reserved_slots, snapshot.active_jobs), (0, 0));
    close(rig, task).await;
}

// Catches: an assignment that arrived before its decision being accepted by the late decision.
#[tokio::test(start_paused = true)]
async fn an_early_assignment_fails_once_and_cannot_be_accepted_by_a_late_decision() {
    let fresh = runtime_rig();
    let (mut rig, task) = running(fresh, 0);
    rig.peer.send(Ok(offer("a"))).unwrap();
    assert_eq!(rig.started.recv().await, Some("decision"));
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "a") == Some(protocol::JobStatus::JsFailed)).await;
    rig.decisions.add_permits(1);
    tokio::time::sleep(Duration::from_secs(1)).await;
    while let Ok(message) = rig.sent.try_recv() {
        assert!(!accepted(&message, "a"), "a late decision accepted a failed job");
    }
    close(rig, task).await;
}

// Catches: a job that already ended being launched again by a repeated assignment.
#[tokio::test(start_paused = true)]
async fn worker_registration_vectors_terminal_assignment_cannot_launch_again() {
    let fresh = runtime_rig();
    fresh.launches.add_permits(2);
    let (mut rig, task) = running(fresh, 1);
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| accepted(m, "a")).await;
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    assert_eq!(rig.started.recv().await, Some("decision"));
    assert_eq!(rig.started.recv().await, Some("launch"));
    tokio::time::sleep(Duration::from_millis(100)).await;
    rig.handle
        .send(WorkerCommand::JobStatus {
            job_id: "a".into(),
            status: protocol::JobStatus::JsSuccess,
        })
        .unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "a") == Some(protocol::JobStatus::JsSuccess)).await;
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "a") == Some(protocol::JobStatus::JsSuccess)).await;
    assert!(rig.started.try_recv().is_err(), "launched once");
    close(rig, task).await;
}

fn offer_with_metadata(id: &str, metadata: &str) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Availability(protocol::AvailabilityRequest {
            job: Some(protocol::Job {
                id: id.into(),
                metadata: metadata.into(),
                ..Default::default()
            }),
            ..Default::default()
        })),
    }
}

fn assignment_with(id: &str, metadata: &str, url: Option<&str>) -> ServerMessage {
    ServerMessage {
        message: Some(server_message::Message::Assignment(protocol::JobAssignment {
            job: Some(protocol::Job {
                id: id.into(),
                metadata: metadata.into(),
                ..Default::default()
            }),
            token: "token".into(),
            url: url.map(Into::into),
        })),
    }
}

fn is_status(want: livekit_protocol::WorkerStatus) -> impl Fn(&protocol::WorkerMessage) -> bool {
    move |m| status(m).is_some_and(|s| s.status == Some(want as i32))
}

// Catches: a full assignment history never rolling the connection over, or rolling it over before
// its pending assignment settled. Over the new connection a running job stays held and a settled
// one may be offered again.
#[tokio::test(start_paused = true)]
async fn a_full_assignment_history_rolls_over_after_pending_assignments_settle() {
    let mut fresh = runtime_rig();
    fresh.runtime.set_assignment_capacity(2);
    let (mut rig, task) = running(fresh, 0);
    rig.launch("done").await;
    rig.handle
        .send(WorkerCommand::JobStatus {
            job_id: "done".into(),
            status: protocol::JobStatus::JsSuccess,
        })
        .unwrap();
    rig.accept("pending").await;
    rig.peer.send(Ok(offer("overload"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "overload")).await;
    let next = rig.next_peer();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(
        rig.connections.lock().len(),
        1,
        "no rollover while an assignment is pending"
    );
    rig.launches.add_permits(1);
    rig.peer.send(Ok(assignment_message("pending"))).unwrap();
    let migrated =
        until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    let Some(Sent::MigrateJob(migrated)) = migrated.message else { unreachable!() };
    assert_eq!(migrated.job_ids, ["pending"], "the running job is held over the new connection");
    decisions_started(&mut rig.started);
    next.send(Ok(offer("pending"))).unwrap();
    next.send(Ok(offer("done"))).unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(decisions_started(&mut rig.started), 1, "only the settled job is asked about again");
    close(rig, task).await;
}

// Catches: not reporting held jobs after a reconnect, so the server re-dispatches them.
#[tokio::test(start_paused = true)]
async fn active_jobs_are_migrated_after_re_registration() {
    let (mut rig, task) = running(runtime_rig_with_retry(1), 0);
    rig.launch("a").await;
    let _next = rig.next_peer();
    rig.peer.send(Err(TransportError::Closed("lost".into()))).unwrap();
    until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::Register(_)))).await;
    let migrated =
        until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    let Some(Sent::MigrateJob(migrated)) = migrated.message else { unreachable!() };
    assert_eq!(migrated.job_ids, ["a"]);
    close(rig, task).await;
}

// Catches: asking the host about a job while the worker is already over its threshold.
#[tokio::test(start_paused = true)]
async fn an_overloaded_worker_rejects_before_asking_the_user() {
    let (mut rig, task) = running(runtime_rig(), 1);
    rig.set_load(0.95);
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "a")).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    close(rig, task).await;
}

// Catches: a draining worker still taking jobs, as simulation (always available) would.
#[tokio::test(start_paused = true)]
async fn a_draining_worker_refuses_new_jobs_even_in_simulation() {
    let cfg = WorkerConfig { simulation: true, ..Default::default() };
    let (mut rig, task) = running(runtime_rig_with(0, cfg), 1);
    until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::UpdateWorker(_)))).await;
    rig.handle.send(WorkerCommand::Drain { timeout: None }).unwrap();
    until_sent(&mut rig.sent, |m| status(m).is_some_and(|s| s.draining)).await;
    rig.peer.send(Ok(offer("a"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "a")).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    close(rig, task).await;
}

// Catches: an assignment arriving at the instant its deadline expires launching the job; what is
// due runs first, so the job is gone and the assignment fails.
#[tokio::test(start_paused = true)]
async fn the_assignment_deadline_precedes_a_simultaneous_assignment() {
    let (mut rig, task) = running(runtime_rig(), 0);
    rig.accept("late").await;
    rig.launches.add_permits(1);
    tokio::time::sleep(ASSIGNMENT_TIMEOUT).await;
    rig.peer.send(Ok(assignment_message("late"))).unwrap();
    until_sent(&mut rig.sent, |m| job_status(m, "late") == Some(protocol::JobStatus::JsFailed))
        .await;
    assert!(rig.launched.lock().is_empty());
    assert_eq!(rig.handle.snapshot().reserved_slots, 0);
    close(rig, task).await;
}

// Catches: judging an offer on the last periodic load instead of a sample taken for it.
#[tokio::test(start_paused = true)]
async fn availability_waits_for_a_fresh_load_sample() {
    let fresh = runtime_rig();
    fresh.load.gated.store(true, std::sync::atomic::Ordering::SeqCst);
    let (mut rig, task) = running(fresh, 1);
    until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::UpdateWorker(_)))).await;
    rig.peer.send(Ok(offer("a"))).unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(decisions_started(&mut rig.started), 0, "decided before its load sample");
    rig.set_load(0.99);
    rig.load.gate.add_permits(1);
    until_sent(&mut rig.sent, |m| refused(m, "a")).await;
    close(rig, task).await;
}

// Catches: the 0.5 s load sampling and 2.5 s status intervals swapped or collapsed.
#[tokio::test(start_paused = true)]
async fn load_samples_every_half_second_and_status_reports_every_two_and_a_half() {
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, |m| status(m).is_some()).await;
    tokio::time::sleep(Duration::from_millis(250)).await;
    let samples = rig.load.samples.load(std::sync::atomic::Ordering::SeqCst);
    tokio::time::sleep(UPDATE_STATUS_INTERVAL).await;
    let sampled = rig.load.samples.load(std::sync::atomic::Ordering::SeqCst) - samples;
    let reported = std::iter::from_fn(|| rig.sent.try_recv().ok()).filter(|m| status(m).is_some());
    assert_eq!((sampled, reported.count()), (5, 1));
    assert_eq!(UPDATE_LOAD_INTERVAL, Duration::from_millis(500));
    close(rig, task).await;
}

// Catches: a capacity change not reaching the server's status, in either direction.
#[tokio::test(start_paused = true)]
async fn the_status_reports_each_capacity_transition() {
    use livekit_protocol::WorkerStatus::{WsAvailable, WsFull};
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.set_load(0.99);
    until_sent(&mut rig.sent, is_status(WsFull)).await;
    rig.set_load(0.1);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    close(rig, task).await;
}

// Catches: the effective load (with reserved slots charged) on the wire, which the server would
// count the reservations in twice.
#[tokio::test(start_paused = true)]
async fn the_wire_carries_raw_worker_load_not_effective_load() {
    let (mut rig, task) = running(runtime_rig(), 0);
    rig.set_load(0.3);
    rig.accept("a").await;
    assert_eq!(rig.handle.snapshot().reserved_slots, 1);
    let reported = until_sent(&mut rig.sent, |m| status(m).is_some()).await;
    assert_eq!(status(&reported).unwrap().load, 0.3);
    close(rig, task).await;
}

// Catches: the launched job built from the assignment's job (or its url left empty) instead of the
// offer's job, with the worker url when the assignment names none.
#[tokio::test(start_paused = true)]
async fn the_launch_runs_the_offered_job_at_the_assigned_or_worker_url() {
    let cfg = WorkerConfig {
        ws_url: "wss://worker".into(),
        load_threshold: EnvOption::Fixed(0.7),
        ..Default::default()
    };
    let (mut rig, task) = running(runtime_rig_with(0, cfg), 3);
    rig.launches.add_permits(3);
    for (id, url) in [("unset", None), ("empty", Some("")), ("named", Some("wss://sfu"))] {
        rig.peer.send(Ok(offer_with_metadata(id, "from-availability"))).unwrap();
        until_sent(&mut rig.sent, |m| accepted(m, id)).await;
        rig.peer.send(Ok(assignment_with(id, "from-assignment", url))).unwrap();
    }
    while rig.launched.lock().len() < 3 {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let launched = rig.launched.lock().clone();
    let got: Vec<_> = launched
        .iter()
        .map(|i| (i.job.id.as_str(), i.url.as_str(), i.job.metadata.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            ("unset", "wss://worker", "from-availability"),
            ("empty", "wss://worker", "from-availability"),
            ("named", "wss://sfu", "from-availability"),
        ]
    );
    close(rig, task).await;
}

// Catches: the dispatch timeline stamped at the wrong step: when the job was offered, accepted
// and assigned.
#[tokio::test(start_paused = true)]
async fn a_launched_job_carries_when_it_was_offered_accepted_and_assigned() {
    let (mut rig, task) = running(runtime_rig(), 0);
    rig.peer.send(Ok(offer("a"))).unwrap();
    assert_eq!(rig.started.recv().await, Some("decision"));
    tokio::time::sleep(Duration::from_secs(2)).await;
    rig.decisions.add_permits(1);
    until_sent(&mut rig.sent, |m| accepted(m, "a")).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    rig.launches.add_permits(1);
    rig.peer.send(Ok(assignment_message("a"))).unwrap();
    while rig.launched.lock().is_empty() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let info = rig.launched.lock()[0].clone();
    let after = |at: f64| ((at - info.received_at) * 1000.0).round() / 1000.0;
    assert_eq!((after(info.accepted_at), after(info.assigned_at)), (2.0, 5.0));
    close(rig, task).await;
}

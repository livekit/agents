use super::*;
use crate::transport::TransportError;
use crate::worker::tests::*;
use crate::*;
use livekit_protocol::worker_message::Message as Sent;
use livekit_protocol::{self as protocol, server_message};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

// Catches: writing lk.agent.name after the user's attributes, overriding a per-job rename.
#[test]
fn user_attributes_win_over_the_agent_name_default() {
    let args = AcceptArguments {
        attributes: BTreeMap::from([(ATTRIBUTE_AGENT_NAME.into(), "override".into())]),
        ..Default::default()
    };
    let args = accept_arguments("configured", args);
    let want = BTreeMap::from([(ATTRIBUTE_AGENT_NAME.to_string(), "override".to_string())]);
    assert_eq!(args.attributes, want);
}

// Catches: forgetting `agent: true` on the register permissions.
#[test]
fn register_marks_the_permission_as_an_agent() {
    let msg = register("a", "d", "1.2.3", &[], "i1");
    let Some(pb::worker_message::Message::Register(r)) = msg.message else {
        panic!("wrong variant")
    };
    #[allow(deprecated, reason = "the server still reads the deprecated `agent` permission")]
    let is_agent = r.allowed_permissions.unwrap().agent;
    assert!(is_agent);
}

// Catches: the draining status setting `load`, or a status other than WS_FULL.
#[test]
fn draining_status_carries_no_load() {
    let msg = update_worker_draining(2);
    let Some(pb::worker_message::Message::UpdateWorker(u)) = msg.message else {
        panic!("wrong variant")
    };
    assert_eq!(u.status, Some(pb::WorkerStatus::WsFull as i32));
    assert_eq!(u.load, 0.0);
    assert_eq!(u.job_count, 2);
}

// Catches: an exponential backoff replacing `min(n * 2, 10)`, losing the immediate first retry.
#[test]
fn the_first_reconnect_is_immediate_and_the_ladder_caps_at_ten() {
    let ladder: Vec<u64> = (0..8).map(|n| retry_delay(n).as_secs()).collect();
    assert_eq!(ladder, vec![0, 2, 4, 6, 8, 10, 10, 10]);
}

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

type Run = lk_agents_abi::OwnedTask<Result<(), RuntimeError>>;

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
    let spawner = lk_agents_net::task::TokioSpawner::current();
    let task = lk_agents_abi::SpawnerExt::spawn(&spawner, async move { runtime.run().await });
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
        self.connections.lock().push_back((peer.downgrade(), incoming));
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
    assert_eq!(rig.connections.lock().len(), 1, "no rollover while an assignment is pending");
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

// Catches: not reporting held jobs after a reconnect, so the server re-dispatches them, or
// reporting bare ids where the server needs each job as assigned, under its accepted identity.
#[tokio::test(start_paused = true)]
async fn active_jobs_are_migrated_after_re_registration() {
    let (mut rig, task) = running(runtime_rig_with_retry(1), 0);
    rig.launch("a").await;
    let _next = rig.next_peer();
    rig.peer.send(Err(TransportError::Failed("lost".into()))).unwrap();
    until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::Register(_)))).await;
    let migrated =
        until_sent(&mut rig.sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    let Some(Sent::MigrateJob(migrated)) = migrated.message else { unreachable!() };
    assert_eq!(migrated.job_ids, ["a"]);
    let [job] = migrated.jobs.as_slice() else { panic!("{:?}", migrated.jobs) };
    assert_eq!(job.id, "a");
    let identity = job.state.as_ref().map(|state| state.participant_identity.as_str());
    assert_eq!(identity, Some("agent-identity"), "the identity the rig accepted it under");
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
async fn a_draining_worker_refuses_new_jobs_even_with_capacity_limits_disabled() {
    let cfg =
        WorkerConfig { load_threshold: EnvOption::Fixed(f32::INFINITY), ..Default::default() };
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

// Catches: the 0.5 s load sampling and 1.5 s status intervals (the 1.x cadence) swapped,
// collapsed or slowed: the server then learns of capacity changes late.
#[tokio::test(start_paused = true)]
async fn load_samples_every_half_second_and_status_reports_every_one_and_a_half() {
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, |m| status(m).is_some()).await;
    tokio::time::sleep(Duration::from_millis(250)).await;
    let samples = rig.load.samples.load(std::sync::atomic::Ordering::SeqCst);
    tokio::time::sleep(UPDATE_STATUS_INTERVAL).await;
    let sampled = rig.load.samples.load(std::sync::atomic::Ordering::SeqCst) - samples;
    let reported = std::iter::from_fn(|| rig.sent.try_recv().ok()).filter(|m| status(m).is_some());
    assert_eq!((sampled, reported.count()), (3, 1));
    assert_eq!(UPDATE_LOAD_INTERVAL, Duration::from_millis(500));
    assert_eq!(UPDATE_STATUS_INTERVAL, Duration::from_millis(1_500));
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
    let after =
        |at: Option<f64>| ((at.unwrap() - info.received_at.unwrap()) * 1000.0).round() / 1000.0;
    assert_eq!((after(info.accepted_at), after(info.assigned_at)), (2.0, 5.0));
    close(rig, task).await;
}
#[tokio::test(start_paused = true)]
async fn hysteresis_applies_to_both_offers_and_published_status() {
    use livekit_protocol::WorkerStatus::{WsAvailable, WsFull};
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.set_load(0.71);
    until_sent(&mut rig.sent, is_status(WsFull)).await;
    rig.set_load(0.68);
    rig.peer.send(Ok(offer("oscillating"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "oscillating")).await;
    until_sent(&mut rig.sent, is_status(WsFull)).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    rig.set_load(0.62);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.accept("recovered").await;
    close(rig, task).await;
}

#[tokio::test(start_paused = true)]
async fn a_stalled_sampler_expires_offers_and_recovers_after_a_valid_sample() {
    use livekit_protocol::WorkerStatus::{WsAvailable, WsFull};
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.load.gated.store(true, std::sync::atomic::Ordering::SeqCst);
    rig.peer.send(Ok(offer("stalled"))).unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;
    until_sent(&mut rig.sent, |m| refused(m, "stalled")).await;
    until_sent(&mut rig.sent, is_status(WsFull)).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    assert_eq!(rig.handle.snapshot().reserved_slots, 0);
    rig.load.gated.store(false, std::sync::atomic::Ordering::SeqCst);
    rig.load.gate.add_permits(1);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.accept("recovered").await;
    close(rig, task).await;
}

#[tokio::test(start_paused = true)]
async fn launched_jobs_keep_startup_reservations_independent_of_prewarming() {
    let cfg = WorkerConfig {
        load_threshold: EnvOption::Fixed(0.35),
        num_idle_processes: EnvOption::Fixed(0),
        ..Default::default()
    };
    let (mut rig, task) = running(runtime_rig_with(0, cfg), 0);
    rig.launch("a").await;
    rig.launch("b").await;
    decisions_started(&mut rig.started);
    rig.peer.send(Ok(offer("burst"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "burst")).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    tokio::time::sleep(Duration::from_secs(5)).await;
    rig.accept("after-startup").await;
    close(rig, task).await;
}
#[tokio::test(start_paused = true)]
async fn invalid_samples_cannot_keep_an_idle_measurement_fresh() {
    use livekit_protocol::WorkerStatus::{WsAvailable, WsFull};
    let (mut rig, task) = running(runtime_rig(), 0);
    until_sent(&mut rig.sent, is_status(WsAvailable)).await;
    rig.set_load(f32::NAN);
    tokio::time::sleep(Duration::from_secs(5)).await;
    until_sent(&mut rig.sent, is_status(WsFull)).await;
    rig.peer.send(Ok(offer("invalid"))).unwrap();
    until_sent(&mut rig.sent, |m| refused(m, "invalid")).await;
    assert_eq!(decisions_started(&mut rig.started), 0);
    rig.set_load(0.1);
    rig.accept("valid").await;
    close(rig, task).await;
}

mod generated_sequences {
    use super::*;

    async fn wait_for_snapshot(handle: &WorkerClient, predicate: impl Fn(&Snapshot) -> bool) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !predicate(&handle.snapshot()) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("worker did not settle the generated operation");
    }

    async fn next_host_operation(
        started: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    ) -> &'static str {
        tokio::time::timeout(Duration::from_secs(1), started.recv())
            .await
            .expect("host operation was not started")
            .expect("worker dropped the host")
    }

    fn record_reply(
        message: protocol::WorkerMessage,
        replies: &mut std::collections::BTreeMap<String, Vec<bool>>,
    ) {
        if let Some(livekit_protocol::worker_message::Message::Availability(reply)) =
            message.message
        {
            replies.entry(reply.job_id).or_default().push(reply.available);
        }
    }

    async fn receive_reply(
        sent: &mut tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
        replies: &mut std::collections::BTreeMap<String, Vec<bool>>,
        job_id: &str,
    ) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !replies.contains_key(job_id) {
                record_reply(sent.recv().await.expect("worker stopped"), replies);
            }
        })
        .await
        .expect("offered job was not answered");
    }

    #[tokio::test(start_paused = true)]
    async fn generated_offer_assignment_and_drain_sequences_settle_once() {
        // Base-three digits choose immediate completion, completion before launch returns, or an
        // expired assignment. Each sequence also varies duplicate offers and assignment messages.
        for seed in 0..81_u64 {
            let RuntimeRig {
                mut runtime,
                handle,
                peer,
                mut sent,
                mut started,
                decisions,
                launches,
                ..
            } = runtime_rig();
            let mut notifications = handle.subscribe();
            let spawner = lk_agents_net::task::TokioSpawner::current();
            let task =
                lk_agents_abi::SpawnerExt::spawn(&spawner, async move { runtime.run().await });
            let first = sent.recv().await.expect("register was not sent");
            assert!(matches!(
                first.message,
                Some(livekit_protocol::worker_message::Message::Register(_))
            ));
            peer.send(register_message("sequence-worker")).unwrap();
            assert!(matches!(
                notifications.recv().await.unwrap(),
                WorkerNotification::Registered(_)
            ));
            let mut replies = std::collections::BTreeMap::new();
            let mut choices = seed;
            for index in 0..4 {
                let mode = choices % 3;
                choices /= 3;
                let job_id = format!("seed-{seed}-job-{index}");
                decisions.add_permits(1);
                for _ in 0..=seed % 3 {
                    peer.send(Ok(offer(&job_id))).unwrap();
                }
                assert_eq!(next_host_operation(&mut started).await, "decision", "seed {seed}");
                receive_reply(&mut sent, &mut replies, &job_id).await;
                assert_eq!(replies[&job_id], [true], "seed {seed}");
                // Let the writer's completion reach the worker before advancing its assignment clock.
                tokio::time::sleep(Duration::from_millis(1)).await;
                if mode == 2 {
                    tokio::time::advance(crate::ASSIGNMENT_TIMEOUT).await;
                    wait_for_snapshot(&handle, |snapshot| snapshot.reserved_slots == 0).await;
                    for _ in 0..=seed % 2 {
                        peer.send(Ok(assignment_message(&job_id))).unwrap();
                    }
                    // A failed disposition establishes that the late assignment was consumed.
                    tokio::time::timeout(Duration::from_secs(1), async {
                        loop {
                            let message = sent.recv().await.expect("worker stopped");
                            let failed = matches!(&message.message,
                                Some(livekit_protocol::worker_message::Message::UpdateJob(update))
                                    if update.job_id == job_id
                                        && update.status == protocol::JobStatus::JsFailed as i32);
                            record_reply(message, &mut replies);
                            if failed {
                                break;
                            }
                        }
                    })
                    .await
                    .expect("late assignment was not refused");
                    assert!(
                        started.try_recv().is_err(),
                        "expired assignment launched: seed {seed}"
                    );
                    continue;
                }
                for _ in 0..=seed % 3 {
                    peer.send(Ok(assignment_message(&job_id))).unwrap();
                }
                assert_eq!(next_host_operation(&mut started).await, "launch", "seed {seed}");
                if mode == 1 {
                    handle
                        .send(WorkerCommand::JobStatus {
                            job_id: job_id.clone(),
                            status: protocol::JobStatus::JsSuccess,
                        })
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                launches.add_permits(1);
                wait_for_snapshot(&handle, |snapshot| snapshot.reserved_slots == 0).await;
                if mode == 0 {
                    assert_eq!(handle.snapshot().active_jobs, 1, "seed {seed}");
                    handle
                        .send(WorkerCommand::JobStatus {
                            job_id,
                            status: protocol::JobStatus::JsSuccess,
                        })
                        .unwrap();
                }
                wait_for_snapshot(&handle, |snapshot| snapshot.active_jobs == 0).await;
                assert!(started.try_recv().is_err(), "duplicate assignment launched: seed {seed}");
            }

            // Keep one accepted request pending when draining; its assignment timeout must release
            // the drain, and offers received during drain must never invoke the host or be accepted.
            let pending_id = format!("seed-{seed}-pending");
            decisions.add_permits(1);
            peer.send(Ok(offer(&pending_id))).unwrap();
            assert_eq!(next_host_operation(&mut started).await, "decision");
            receive_reply(&mut sent, &mut replies, &pending_id).await;
            tokio::time::sleep(Duration::from_millis(1)).await;
            handle.send(WorkerCommand::Drain { timeout: Some(Duration::from_secs(20)) }).unwrap();
            wait_for_snapshot(&handle, |snapshot| snapshot.draining).await;
            assert!(
                handle.snapshot().drain_outcome.is_none(),
                "drained with a pending job: seed {seed}"
            );
            let refused_id = format!("seed-{seed}-during-drain");
            peer.send(Ok(offer(&refused_id))).unwrap();
            receive_reply(&mut sent, &mut replies, &refused_id).await;
            assert_eq!(replies[&refused_id], [false], "seed {seed}");
            tokio::time::advance(crate::ASSIGNMENT_TIMEOUT).await;
            handle.drain(Some(Duration::from_secs(20))).await.unwrap();
            assert_eq!(handle.snapshot().reserved_slots, 0, "seed {seed}");
            assert_eq!(handle.snapshot().active_jobs, 0, "seed {seed}");
            handle.send(WorkerCommand::Close).unwrap();
            task.await.unwrap().unwrap();
            while let Ok(message) = sent.try_recv() {
                record_reply(message, &mut replies);
            }
            assert!(
                replies.values().all(|answers| answers.len() == 1),
                "duplicate reply: seed {seed}"
            );
            assert!(started.try_recv().is_err(), "draining worker invoked host: seed {seed}");
            let mut drain_finished = 0;
            while let Ok(event) = notifications.try_recv() {
                match event {
                    WorkerNotification::Drained(DrainOutcome::Finished) => drain_finished += 1,
                    WorkerNotification::Drained(DrainOutcome::TimedOut) => {
                        panic!("drain timed out: seed {seed}")
                    }
                    _ => {}
                }
            }
            assert_eq!(drain_finished, 1, "seed {seed}");
        }
    }
}

#[tokio::test(start_paused = true)]
async fn dropping_a_pending_decision_cancels_the_callback_it_owns() {
    use lk_agents_net::signal;
    struct HoldingHost {
        started: Mutex<Option<signal::Producer<Option<()>>>>,
        ended: Mutex<Option<signal::Producer<Option<()>>>>,
    }
    #[async_trait::async_trait]
    impl WorkerHost for HoldingHost {
        async fn on_job_request(&self, _: JobRequest) -> JobDecision {
            let _ended = self.ended.lock().take().unwrap();
            self.started.lock().take().unwrap().resolve(());
            std::future::pending().await
        }
        async fn launch_job(&self, _: RunningJobInfo) -> HostResult<JobId> {
            Err(HostError::new("no launch"))
        }
        async fn terminate_job(&self, _: JobId) -> HostResult {
            Ok(())
        }
        async fn user_load(&self) -> Option<HostResult<f32>> {
            Some(Ok(0.0))
        }
    }
    let mut rig = runtime_rig_with(
        0,
        WorkerConfig {
            devmode: true,
            load_threshold: EnvOption::Fixed(f32::INFINITY),
            ..Default::default()
        },
    );
    let (started, running) = signal::once();
    let (ended, stopped) = signal::once();
    rig.runtime.host = Arc::new(HoldingHost {
        started: Mutex::new(Some(started)),
        ended: Mutex::new(Some(ended)),
    });
    let mut outbox = Outbox::new(Duration::from_secs(10), rig.runtime.clock.clone());
    rig.runtime.on_connecting(&mut outbox);
    rig.runtime.on_registered(
        protocol::RegisterWorkerResponse { worker_id: "worker".into(), ..Default::default() },
        &mut outbox,
    );
    rig.runtime.on_server(offer("held"), &mut outbox);
    running.wait().await.unwrap();
    let slot = rig.runtime.pending[0].slot;
    rig.runtime.drop_pending(slot);
    let ended = tokio::time::timeout(Duration::from_secs(1), stopped.wait())
        .await
        .expect("removing the pending decision must end its callback");
    assert!(ended.is_err(), "cancellation ends the producer without fabricating a decision");
    assert_eq!(rig.runtime.reserved_slots(), 0);
}

// Registers the worker and reads the snapshot right after the `Registered` notification. Catches a
// notification that is sent before the snapshot shows the registration. For example, a waiter on
// another thread wakes up, reads `registered == false`, and waits again. The later snapshot update
// does not wake it, so it waits forever.
#[tokio::test(start_paused = true)]
async fn the_registration_is_published_before_its_notification() {
    let mut rig = runtime_rig();
    let mut notifications = rig.handle.subscribe();
    let mut outbox = Outbox::new(Duration::from_secs(10), rig.runtime.clock.clone());
    rig.runtime.on_connecting(&mut outbox);
    rig.runtime.on_registered(
        protocol::RegisterWorkerResponse { worker_id: "worker".into(), ..Default::default() },
        &mut outbox,
    );
    assert!(matches!(notifications.try_recv(), Ok(WorkerNotification::Registered(_))));
    // Read the published snapshot, as the woken waiter does. Do not publish a new one.
    let state = rig.handle.snapshot();
    assert!(state.registered, "notified of a registration the snapshot does not show");
    assert_eq!(state.worker_id, "worker");
}

// Token mode: job token requests on the worker connection.

const CLAIMS: &str = r#"{"identity":"avatar","video":{"roomJoin":true,"room":"r"}}"#;

type Outgoing = UnboundedReceiver<protocol::WorkerMessage>;

/// Starts a worker loop and waits until it is registered.
async fn registered() -> (WorkerClient, Peer, Outgoing, Run) {
    use lk_agents_abi::SpawnerExt as _;
    let RuntimeRig { mut runtime, handle, peer, sent, .. } = runtime_rig();
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let task =
        lk_agents_net::task::TokioSpawner::current().spawn(async move { runtime.run().await });
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}
    (handle, peer, sent, task)
}

async fn stop(handle: WorkerClient, task: Run) {
    handle.send(WorkerCommand::Close).unwrap();
    tokio::time::timeout(Duration::from_secs(1), task).await.unwrap().unwrap().unwrap();
}

async fn next_request(sent: &mut Outgoing) -> protocol::CreateJobTokenRequest {
    let message = until_sent(sent, |m| matches!(m.message, Some(Sent::CreateJobToken(_)))).await;
    let Some(Sent::CreateJobToken(request)) = message.message else { unreachable!() };
    request
}

fn answer(request_id: &str, token: &str, error: &str) -> Result<ServerMessage, TransportError> {
    let response = protocol::CreateJobTokenResponse {
        request_id: request_id.into(),
        token: token.into(),
        expires_at: 600,
        error: error.into(),
    };
    Ok(ServerMessage { message: Some(server_message::Message::CreateJobToken(response)) })
}

fn ask(
    handle: &WorkerClient,
    claims: &'static str,
) -> lk_agents_abi::OwnedTask<Result<JobToken, ApiError>> {
    use lk_agents_abi::SpawnerExt as _;
    let handle = handle.clone();
    let spawner = lk_agents_net::task::TokioSpawner::current();
    spawner.spawn(async move { handle.create_job_token("AJ_1", claims).await })
}

// Two requests answered in reverse order, then a refusal. Catches an answer that goes to the
// wrong request, a request without its job id or claims, and a refusal that is read as a token
// or loses the server's reason.
#[tokio::test(start_paused = true)]
async fn a_job_token_request_is_answered_by_its_request_id() {
    let (handle, peer, mut sent, task) = registered().await;
    let first = ask(&handle, CLAIMS);
    let second = ask(&handle, r#"{"video":{"roomAdmin":true,"room":"r"}}"#);
    let (a, b) = (next_request(&mut sent).await, next_request(&mut sent).await);
    assert_eq!((a.job_id.as_str(), b.job_id.as_str()), ("AJ_1", "AJ_1"));
    let (a, b) = if a.claims == CLAIMS { (a, b) } else { (b, a) };
    assert_ne!(a.request_id, b.request_id);
    peer.send(answer(&b.request_id, "signed-b", "")).unwrap();
    peer.send(answer(&a.request_id, "signed-a", "")).unwrap();
    let issued = first.await.unwrap().unwrap();
    assert_eq!((issued.token.as_str(), issued.expires_at), ("signed-a", 600));
    assert_eq!(second.await.unwrap().unwrap().token, "signed-b");

    let refused = ask(&handle, CLAIMS);
    let request = next_request(&mut sent).await;
    peer.send(answer(&request.request_id, "", "job not on this worker")).unwrap();
    let error = refused.await.unwrap().unwrap_err();
    assert!(error.message.contains("job not on this worker"), "{error}");
    stop(handle, task).await;
}

// Catches a job that waits forever on a server that never answers.
#[tokio::test(start_paused = true)]
async fn an_unanswered_job_token_request_times_out() {
    let (handle, _peer, mut sent, task) = registered().await;
    let started = tokio::time::Instant::now();
    let asking = ask(&handle, CLAIMS);
    next_request(&mut sent).await;
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.message.contains("did not answer"), "{error}");
    assert_eq!(started.elapsed(), JOB_TOKEN_TIMEOUT);
    stop(handle, task).await;
}

// A request before registration, and one just before the connection drops. Catches a request
// sent unregistered, which could only time out, and one that waits through a lost connection,
// which its answer can no longer come on.
#[tokio::test(start_paused = true)]
async fn job_tokens_need_a_registered_connection_and_fail_when_it_drops() {
    use lk_agents_abi::SpawnerExt as _;
    let RuntimeRig { mut runtime, handle, peer, mut sent, .. } = runtime_rig_with_retry(5);
    let task =
        lk_agents_net::task::TokioSpawner::current().spawn(async move { runtime.run().await });
    until_sent(&mut sent, |m| matches!(m.message, Some(Sent::Register(_)))).await;
    let started = tokio::time::Instant::now();
    let error = handle.create_job_token("AJ_1", CLAIMS).await.unwrap_err();
    assert!(error.message.contains("not connected"), "{error}");
    assert!(started.elapsed() < JOB_TOKEN_TIMEOUT);

    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}
    let asking = ask(&handle, CLAIMS);
    next_request(&mut sent).await;
    peer.send(Err(TransportError::Failed("peer disconnected".into()))).unwrap();
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.message.contains("closed before it answered"), "{error}");
    stop(handle, task).await;
}

/// A job kept across a reload, as the CLI gives it back.
fn kept_job() -> RunningJobInfo {
    RunningJobInfo {
        accept_arguments: AcceptArguments { identity: "agent-7".into(), ..Default::default() },
        job: protocol::Job { id: "AJ_1".into(), ..Default::default() },
        url: "wss://sfu".into(),
        token: "kept".into(),
        worker_id: "old-worker".into(),
        fake_job: false,
        received_at: None,
        accepted_at: None,
        assigned_at: None,
        launched_at: 0.0,
    }
}

// A kept job is adopted after its launch, then the connection drops and the worker registers
// again. Catches a kept job whose status the worker refuses, and one missing from the migration
// re-sent at the next registration, so that the server refuses its job tokens.
#[tokio::test(start_paused = true)]
async fn an_adopted_kept_job_reports_its_status_and_is_re_sent_after_a_reconnect() {
    use lk_agents_abi::SpawnerExt as _;
    let RuntimeRig { mut runtime, handle, peer, mut sent, connections, .. } =
        runtime_rig_with_retry(1);
    let mut notifications = handle.subscribe();
    peer.send(register_message("worker")).unwrap();
    let task =
        lk_agents_net::task::TokioSpawner::current().spawn(async move { runtime.run().await });
    while !matches!(notifications.recv().await.unwrap(), WorkerNotification::Registered(_)) {}

    handle.send(WorkerCommand::AdoptJob(Box::new(kept_job()))).unwrap();
    until_sent(&mut sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    assert_eq!(handle.snapshot().active_jobs, 1);
    let status = protocol::JobStatus::JsRunning;
    handle.send(WorkerCommand::JobStatus { job_id: "AJ_1".into(), status }).unwrap();
    until_sent(&mut sent, |m| matches!(&m.message, Some(Sent::UpdateJob(u)) if u.job_id == "AJ_1"))
        .await;

    let (next, incoming) = tokio::sync::mpsc::unbounded_channel();
    next.send(register_message("worker")).unwrap();
    connections.lock().push_back((next.downgrade(), incoming));
    peer.send(Err(TransportError::Failed("lost".into()))).unwrap();
    let migrated = until_sent(&mut sent, |m| matches!(m.message, Some(Sent::MigrateJob(_)))).await;
    let Some(Sent::MigrateJob(migrated)) = migrated.message else { unreachable!() };
    assert_eq!(migrated.job_ids, ["AJ_1"]);
    stop(handle, task).await;
}

// Catches a reload whose token request goes out before the server learns the kept job runs here
// (the server then refuses it), or a migration without the job's accepted identity.
#[tokio::test(start_paused = true)]
async fn kept_jobs_are_migrated_before_their_token_requests() {
    let (handle, peer, mut sent, task) = registered().await;
    handle.send(WorkerCommand::MigrateJobs(vec![kept_job()])).unwrap();
    let asking = ask(&handle, CLAIMS);
    let mut order = Vec::new();
    while order.len() < 2 {
        match sent.recv().await.unwrap().message {
            Some(Sent::MigrateJob(migrated)) => {
                let [job] = migrated.jobs.as_slice() else { panic!("{migrated:?}") };
                let identity = job.state.as_ref().map(|s| s.participant_identity.as_str());
                assert_eq!((job.id.as_str(), identity), ("AJ_1", Some("agent-7")));
                order.push("migrate");
            }
            Some(Sent::CreateJobToken(request)) => {
                peer.send(answer(&request.request_id, "signed", "")).unwrap();
                order.push("token");
            }
            _ => {}
        }
    }
    assert_eq!(order, ["migrate", "token"]);
    asking.await.unwrap().unwrap();
    stop(handle, task).await;
}

/// A register response from a server that reports `agent_protocol`.
fn register_on(worker_id: &str, agent_protocol: i32) -> Result<ServerMessage, TransportError> {
    let server_info = protocol::ServerInfo { agent_protocol, ..Default::default() };
    let response = protocol::RegisterWorkerResponse {
        worker_id: worker_id.to_owned(),
        server_info: Some(server_info),
        ..Default::default()
    };
    Ok(ServerMessage { message: Some(server_message::Message::Register(response)) })
}

/// A transport that records each connect, then replaces the worker token as the CLI's
/// `Credentials` message does.
struct Refreshing {
    inner: ScriptedTransport,
    token: Option<Arc<Mutex<String>>>,
}

#[async_trait::async_trait]
impl crate::transport::WorkerTransport for Refreshing {
    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<crate::transport::Connected, TransportError> {
        let connected = self.inner.connect(url, worker_token, headers).await;
        if let Some(token) = &self.token {
            *token.lock() = "refreshed-worker-token".into();
        }
        connected
    }
}

/// Runs a worker over two scripted connections to servers that report `agent_protocol`.
/// Returns the run's outcome (`None` while it still runs) and each connect's `Authorization`.
fn run_against(
    cfg: RuntimeConfig,
    agent_protocol: i32,
    refresh: bool,
) -> (Option<Result<(), RuntimeError>>, Vec<Option<String>>) {
    let wire = Arc::new(Mutex::new(Wire::default()));
    let steps = vec![
        Step::Serves(vec![
            register_on("W-1", agent_protocol),
            Err(TransportError::Failed("peer disconnected".into())),
        ]),
        Step::Serves(vec![register_on("W-2", agent_protocol)]),
    ];
    let inner = ScriptedTransport { script: Mutex::new(steps.into()), wire: wire.clone() };
    let transport = Refreshing { inner, token: refresh.then(|| cfg.token.clone()) };
    let runtime =
        tokio::runtime::Builder::new_current_thread().enable_all().start_paused(true).build();
    let outcome = runtime.unwrap().block_on(async {
        let (mut worker, _handle) = Worker::with_components(
            Arc::new(transport),
            cfg,
            WorkerConfig { ws_url: "wss://gw".into(), ..Default::default() },
            Arc::new(LoadHost::default()),
            rt(Arc::new(TestClock(tokio::time::Instant::now()))),
            blocking_io(),
        );
        tokio::time::timeout(Duration::from_millis(50), worker.run()).await.ok()
    });
    let headers = wire.lock().headers.clone();
    let bearers = headers.iter().map(|headers| {
        headers.iter().find(|(name, _)| name == "Authorization").map(|(_, v)| v.clone())
    });
    (outcome, bearers.collect())
}

fn token_cfg(api_key: &str, api_secret: &str) -> RuntimeConfig {
    RuntimeConfig {
        api_key: api_key.into(),
        api_secret: api_secret.into(),
        token: Arc::new(Mutex::new("lk-worker-token".into())),
        ..runtime_cfg(5)
    }
}

// Catches a token mode worker that signs, drops or reuses an old worker token instead of the
// current one at each registration, a worker token that wins over a key pair, and a bearer sent
// with no credentials.
#[test]
fn token_mode_registers_with_the_current_worker_token_and_a_key_pair_wins() {
    let (_, bearers) = run_against(token_cfg("", ""), JOB_TOKEN_AGENT_PROTOCOL, true);
    let want = ["Bearer lk-worker-token", "Bearer refreshed-worker-token"].map(|b| Some(b.into()));
    assert_eq!(bearers, want);

    let (key, secret) = ("APIkey", "thiskeyistotallyunsafe");
    for bearer in run_against(token_cfg(key, secret), JOB_TOKEN_AGENT_PROTOCOL, true).1 {
        let jwt = bearer.as_deref().and_then(|b| b.strip_prefix("Bearer ")).expect("a bearer");
        lk_agents_net::token::extend_job_token(jwt, key, secret).expect("signed with the key pair");
    }

    assert_eq!(run_against(runtime_cfg(5), JOB_TOKEN_AGENT_PROTOCOL, false).1, [None, None]);
}

// Catches a token mode worker that stays registered on, or retries, a server that does not issue
// job tokens (its first token request would close the connection), and a key mode worker that
// the same check stops.
#[test]
fn token_mode_stops_on_a_server_without_job_tokens_and_a_key_pair_does_not() {
    let (outcome, bearers) = run_against(token_cfg("", ""), 1, false);
    let error = outcome.expect("the worker stopped").unwrap_err();
    assert!(matches!(error, RuntimeError::Configuration(_)), "{error}");
    assert!(error.to_string().contains("does not support token mode"), "{error}");
    assert_eq!(bearers.len(), 1, "not retried");

    let (outcome, _) = run_against(token_cfg("APIkey", "thiskeyistotallyunsafe"), 0, false);
    assert!(outcome.is_none(), "a key pair stays registered: {outcome:?}");
}

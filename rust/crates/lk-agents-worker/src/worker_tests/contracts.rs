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
    let RuntimeRig { mut runtime, handle, peer, sent, started, decisions: gate, .. } = rig;
    peer.send(register_message("worker")).unwrap();
    gate.add_permits(decisions);
    let task = tokio::spawn(async move { runtime.run().await });
    (Rig { handle, peer, sent, started, decisions: gate }, task)
}

/// The rig's parts a running worker leaves to the test.
struct Rig {
    handle: WorkerClient,
    peer: tokio::sync::mpsc::UnboundedSender<Result<ServerMessage, TransportError>>,
    sent: tokio::sync::mpsc::UnboundedReceiver<protocol::WorkerMessage>,
    started: tokio::sync::mpsc::UnboundedReceiver<&'static str>,
    decisions: Arc<tokio::sync::Semaphore>,
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

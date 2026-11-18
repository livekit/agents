use super::tests::{assignment_message, offer, register_message, runtime_rig, until_sent};
use super::*;
use livekit_protocol::{self as pb, server_message, worker_message};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure {
    Decision,
    Launch,
    Termination,
    Load,
    EndpointShutdown,
}

struct FailingHost {
    failure: Failure,
    failed: AtomicBool,
    launched: AtomicBool,
    terminated: AtomicBool,
}
impl FailingHost {
    fn fail(&self, at: Failure) {
        assert!(
            self.failure != at || self.failed.swap(true, Ordering::SeqCst),
            "host failure at {at:?}"
        );
    }
}
#[async_trait]
impl WorkerHost for FailingHost {
    async fn on_job_request(&self, request: JobRequest) -> JobDecision {
        self.fail(Failure::Decision);
        if request.job.id == "broken" && self.failure != Failure::Decision {
            JobDecision::Accept(crate::AcceptArguments::default())
        } else {
            JobDecision::Reject
        }
    }
    async fn launch_job(&self, info: RunningJobInfo) -> HostResult<JobId> {
        // The child exists, but its launch receipt is lost when the callback unwinds.
        self.launched.store(true, Ordering::SeqCst);
        self.fail(Failure::Launch);
        Ok(info.job.id)
    }
    async fn terminate_job(&self, _: JobId) -> HostResult {
        self.terminated.store(true, Ordering::SeqCst);
        self.fail(Failure::Termination);
        self.launched.store(false, Ordering::SeqCst);
        Ok(())
    }
    async fn user_load(&self) -> Option<HostResult<f32>> {
        self.fail(Failure::Load);
        Some(Ok(0.0))
    }
    async fn stop_endpoints(&self) {
        self.fail(Failure::EndpointShutdown);
    }
}

async fn survives(failure: Failure) {
    let mut rig = runtime_rig();
    let host = Arc::new(FailingHost {
        failure,
        failed: AtomicBool::new(false),
        launched: AtomicBool::new(false),
        terminated: AtomicBool::new(false),
    });
    rig.runtime.host = host.clone();
    let handle = rig.handle.clone();
    let task = tokio::spawn(async move { rig.runtime.run().await });
    rig.peer.send(register_message("worker")).unwrap();
    if failure != Failure::Load {
        rig.peer.send(Ok(offer("broken"))).unwrap();
        until_sent(&mut rig.sent, |message| {
            matches!(
                &message.message, Some(worker_message::Message::Availability(reply))
                    if reply.job_id == "broken" && reply.available == (failure != Failure::Decision)
            )
        })
        .await;
        if failure == Failure::Launch || failure == Failure::Termination {
            rig.peer.send(Ok(assignment_message("broken"))).unwrap();
            if failure == Failure::Launch {
                until_sent(&mut rig.sent, |message| matches!(
                    &message.message, Some(worker_message::Message::UpdateJob(status))
                        if status.job_id == "broken" && status.status == pb::JobStatus::JsFailed as i32
                )).await;
            } else {
                until_sent(&mut rig.sent, |message| matches!(
                    &message.message, Some(worker_message::Message::UpdateWorker(status)) if status.job_count == 1
                )).await;
                rig.peer
                    .send(Ok(pb::ServerMessage {
                        message: Some(server_message::Message::Termination(pb::JobTermination {
                            job_id: "broken".into(),
                        })),
                    }))
                    .unwrap();
            }
        }
    }
    rig.peer.send(Ok(offer("healthy"))).unwrap();
    until_sent(&mut rig.sent, |message| matches!(
        &message.message, Some(worker_message::Message::Availability(reply)) if reply.job_id == "healthy" && !reply.available
    )).await;
    assert!(host.failed.load(Ordering::SeqCst), "the fault was never injected");
    if failure == Failure::Launch {
        assert!(host.terminated.load(Ordering::SeqCst), "a committed launch was not cleaned up");
        assert!(!host.launched.load(Ordering::SeqCst), "the failed launch left a child running");
    }
    assert_eq!(handle.snapshot().reserved_slots, 0, "a dead callback retained capacity");
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_lost_decision_task_rejects_its_job_and_releases_capacity() {
    survives(Failure::Decision).await;
}
#[tokio::test(start_paused = true)]
async fn a_panicking_job_launcher_fails_only_its_assignment() {
    survives(Failure::Launch).await;
}
#[tokio::test(start_paused = true)]
async fn a_panicking_termination_callback_does_not_stop_the_worker() {
    survives(Failure::Termination).await;
}
#[tokio::test(start_paused = true)]
async fn a_panicking_load_callback_does_not_stop_the_worker() {
    survives(Failure::Load).await;
}

#[tokio::test(start_paused = true)]
async fn a_panicking_endpoint_shutdown_still_closes_the_transport_and_settles_the_worker() {
    let mut rig = runtime_rig();
    let host = Arc::new(FailingHost {
        failure: Failure::EndpointShutdown,
        failed: AtomicBool::new(false),
        launched: AtomicBool::new(false),
        terminated: AtomicBool::new(false),
    });
    rig.runtime.host = host.clone();
    let handle = rig.handle.clone();
    let closed = rig.closed.clone();
    let task = tokio::spawn(async move { rig.runtime.run().await });
    rig.peer.send(register_message("worker")).unwrap();
    rig.peer.send(Ok(offer("healthy"))).unwrap();
    until_sent(&mut rig.sent, |message| matches!(
        &message.message, Some(worker_message::Message::Availability(reply)) if reply.job_id == "healthy"
    )).await;
    handle.send(WorkerCommand::Close).unwrap();
    task.await.unwrap().unwrap();
    assert!(host.failed.load(Ordering::SeqCst));
    assert!(closed.load(Ordering::SeqCst) > 0, "the control transport was not closed");
    let snapshot = handle.snapshot();
    assert!(snapshot.finished.is_some());
    assert!(!snapshot.registered);
    assert_eq!(snapshot.reserved_slots, 0);
}

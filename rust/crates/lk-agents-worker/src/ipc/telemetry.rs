#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use lk_agents_abi::ATTRIBUTE_SIMULATOR_DISPATCH;
use lk_agents_abi::telemetry::decl::Seconds;
pub(crate) use lk_agents_abi::telemetry::instruments::{
    WorkerIdleProcesses, WorkerIdleProcessesTarget, WorkerJobs, WorkerLoad,
    WorkerProcessReadyDuration,
};
pub(crate) use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::telemetry::spans::job::{
    EntrypointStarted, JobAccepted, JobAssigned, JobReceived, JobStartedOnServer, ProcessAssigned,
};
use lk_agents_abi::worker::job::{Simulation, StartJob};
pub(crate) use lk_agents_net::export::exporter;
pub(crate) use lk_agents_net::export::otlp::CloudEndpoint;
use lk_agents_net::export::otlp::SessionIdentity;
pub(crate) use lk_agents_telemetry::Tracer;
pub(crate) use lk_agents_telemetry::process;

pub(crate) use lk_agents_abi::telemetry::decl::SpanDecl;
pub(crate) use lk_agents_abi::telemetry::policy::{
    SHUTDOWN_TIMEOUT_SECONDS, UPLOAD_TIMEOUT_SECONDS,
};
pub(crate) use lk_agents_abi::telemetry::report::RecordingOptions;
pub(crate) use lk_agents_abi::telemetry::spans::endpoint::{HttpRequest, HttpRequestEnd};
pub(crate) use lk_agents_abi::telemetry::spans::job::{
    EntrypointCancelled, EntrypointReturned, JobEntrypoint, JobShutdown, OnSessionEnd,
    RoomDisconnect, SessionEndUpload, ShutdownCallback,
};
pub(crate) use lk_agents_net::export::otlp::CloudRecording;
pub(crate) use lk_agents_net::export::session_report::{self, ReportOutputs};
pub(crate) use lk_agents_telemetry::{Cx, ScopedSpan, SpanContext, SpanEnd, end, event, start};

/// The job the exporter's registry knows, from its dispatch.
pub(crate) struct Job {
    pub key: JobKey,
    /// The simulation run the job is part of.
    pub simulation: Option<Simulation>,
}

/// A started job: who it is, and its root span, open until its shutdown ends.
pub(crate) struct JobTrace {
    pub job: Job,
    pub span: ScopedSpan<JobEntrypoint>,
}

/// A simulated job's run identity: the `lk.simulator.dispatch` job attribute, a protojson
/// `SimulationDispatch`. `None` for an ordinary job.
fn simulation(job: &livekit_protocol::Job) -> Option<Simulation> {
    let dispatch = job.attributes.get(ATTRIBUTE_SIMULATOR_DISPATCH)?;
    let dispatch: serde_json::Value = serde_json::from_str(dispatch).ok()?;
    let field = |camel: &str, snake: &str| {
        dispatch
            .get(camel)
            .or_else(|| dispatch.get(snake))
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    };
    let run_id = field("simulationRunId", "simulation_run_id")?;
    Some(Simulation { run_id: Some(run_id), job_id: field("jobId", "job_id") })
}

/// A protocol string field: empty is unset.
fn known(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn nanos(at: f64) -> u64 {
    (at * 1e9) as u64
}

/// Seconds between two known instants.
fn gap(start: Option<f64>, end: Option<f64>) -> Option<Seconds> {
    Some(Seconds((end? - start?).max(0.0)))
}

/// Opens the job's root span and registers the job with the process exporter.
pub(crate) fn start_job(
    tracer: &Tracer,
    cloud: Option<&CloudEndpoint>,
    start: &StartJob,
) -> JobTrace {
    let job = &start.info.job;
    let room = job.room.clone().unwrap_or_default();
    let key = JobKey(job.id.clone());
    if let Some(exporter) = exporter() {
        if let Some(cloud) = cloud {
            exporter.configure_cloud(cloud.clone());
        }
        exporter.open_job(
            key.clone(),
            SessionIdentity {
                room_id: known(&room.sid),
                job_id: known(&job.id),
                room_name: known(&room.name),
                agent_name: known(&job.agent_name),
            },
        );
    }
    let parent = start.traceparent.as_deref().and_then(SpanContext::parse_traceparent);
    let cx = tracer
        .root()
        .continuing(parent)
        .with_job(Some(key.clone()))
        // `gen_ai.conversation.id`: the room session; a session's spans carry its resolved thread.
        .with_conversation(Some(room.sid.as_str()).filter(|sid| !sid.is_empty()));
    let started = tracer.now_nanos() as f64 / 1e9;
    let f = &start.info;
    let decl = JobEntrypoint {
        dispatch_id: known(&job.dispatch_id),
        worker_id: known(&f.worker_id),
        agent_id: job.state.as_ref().and_then(|s| known(&s.agent_id)),
        accept_latency: gap(f.received_at, f.accepted_at),
        assignment_latency: gap(f.accepted_at, f.assigned_at),
        launch_latency: gap(f.assigned_at, Some(f.launched_at)),
        entrypoint_latency: gap(Some(f.launched_at), Some(started)),
        dispatch_latency: gap(f.received_at, Some(started)),
    };
    let start = nanos(f.received_at.unwrap_or(started));
    let span = lk_agents_telemetry::start_at(&cx, decl, start);
    if let Some(at) = f.received_at.map(nanos) {
        lk_agents_telemetry::event_at(&span, JobReceived, at);
    }
    if let Some(at) = f.accepted_at.map(nanos) {
        lk_agents_telemetry::event_at(&span, JobAccepted, at);
    }
    if let Some(at) = f.assigned_at.map(nanos) {
        lk_agents_telemetry::event_at(&span, JobAssigned, at);
    }
    lk_agents_telemetry::event_at(&span, ProcessAssigned, nanos(f.launched_at));
    lk_agents_telemetry::event_at(&span, EntrypointStarted, nanos(started));
    // `JobState.started_at`: unix nanoseconds.
    let started_on_server = job.state.as_ref().and_then(|s| u64::try_from(s.started_at).ok());
    if let Some(at) = started_on_server.filter(|at| *at > 0) {
        lk_agents_telemetry::event_at(&span, JobStartedOnServer, at);
    }
    JobTrace { job: Job { key, simulation: simulation(job) }, span }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: an unknown stage counted as an instant.
    #[test]
    fn unknown_stages_have_no_latency() {
        assert_eq!(gap(None, Some(2.0)), None);
        assert_eq!(gap(Some(0.0), Some(2.0)), Some(Seconds(2.0)));
        assert_eq!(gap(Some(3.0), Some(2.0)), Some(Seconds(0.0)));
    }

    // Catches: a simulated job's run identity lost (`lk.simulation.job_id` read from a field the
    // `SimulationDispatch` proto does not have), or an ordinary job taken for a simulation.
    #[test]
    fn the_simulation_identity_is_read_from_the_dispatch() {
        let mut job = livekit_protocol::Job::default();
        assert_eq!(simulation(&job), None);
        job.attributes.insert(
            ATTRIBUTE_SIMULATOR_DISPATCH.to_owned(),
            r#"{"simulationRunId":"SR_1","jobId":"SJ_2","mode":"SIMULATION_MODE_AUDIO"}"#
                .to_owned(),
        );
        assert_eq!(
            simulation(&job),
            Some(Simulation { run_id: Some("SR_1".into()), job_id: Some("SJ_2".into()) })
        );
        job.attributes.insert(
            ATTRIBUTE_SIMULATOR_DISPATCH.to_owned(),
            r#"{"simulation_run_id":"SR_1","job_id":"SJ_2"}"#.to_owned(),
        );
        assert_eq!(simulation(&job).and_then(|s| s.job_id), Some("SJ_2".into()));
        job.attributes.insert(ATTRIBUTE_SIMULATOR_DISPATCH.to_owned(), "{}".to_owned());
        assert_eq!(simulation(&job), None);
    }
}

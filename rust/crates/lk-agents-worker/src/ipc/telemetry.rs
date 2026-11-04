//! The job's telemetry: its root span, `job_entrypoint`, backdated to the job's receipt,
//! continuing the dispatch's `traceparent`, with one event per known dispatch stage and the
//! latencies between them; the job's registration with the exporter's Cloud destination; and the
//! names the job's logic spells `telemetry::` (`telemetry.md`, rule 4).

use lk_agents_abi::ATTRIBUTE_SIMULATOR_DISPATCH;
use lk_agents_abi::telemetry::decl::Seconds;
pub(crate) use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::telemetry::spans::job::{
    EntrypointStarted, JobAccepted, JobAssigned, JobEntrypoint, JobReceived, JobStartedOnServer,
    ProcessAssigned,
};
use lk_agents_abi::worker::job::StartJobFields;
pub(crate) use lk_agents_net::telemetry::Tracer;
pub(crate) use lk_agents_net::telemetry::otlp::CloudEndpoint;
use lk_agents_net::telemetry::otlp::JobIdentity;
pub(crate) use lk_agents_net::telemetry::otlp::Simulation;
use lk_agents_net::telemetry::parse_traceparent;
pub(crate) use lk_agents_net::telemetry::traceparent;
use prost::Message as _;

pub(crate) use lk_agents_abi::telemetry::decl::SpanDecl;
pub(crate) use lk_agents_abi::telemetry::policy::{
    SHUTDOWN_TIMEOUT_SECONDS, UPLOAD_TIMEOUT_SECONDS,
};
pub(crate) use lk_agents_abi::telemetry::report::RecordingOptions;
pub(crate) use lk_agents_abi::telemetry::spans::job::{
    EntrypointCancelled, EntrypointReturned, JobShutdown, OnSessionEnd, RoomDisconnect,
    SessionEndUpload, ShutdownCallback,
};
pub(crate) use lk_agents_net::telemetry::otlp::CloudRecording;
pub(crate) use lk_agents_net::telemetry::session_report::{self, ReportOutputs};
pub(crate) use lk_agents_net::telemetry::{Cx, ScopedSpan, check, end_with, event, fail, start};

/// The job's root span, open until its shutdown ends.
pub(crate) struct JobTrace {
    pub key: JobKey,
    pub span: ScopedSpan<JobEntrypoint>,
    /// The simulation run the job is part of, from its dispatch.
    pub simulation: Option<Simulation>,
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

fn nanos(at: f64) -> u64 {
    (at * 1e9) as u64
}

/// Seconds between two known instants (zero is unknown).
fn gap(start: f64, end: f64) -> Option<Seconds> {
    (start != 0.0 && end != 0.0).then(|| Seconds((end - start).max(0.0)))
}

/// The server's start instant in seconds: `JobState.started_at` is nanoseconds, or milliseconds
/// or seconds should that change.
fn server_started(started_at: i64) -> Option<f64> {
    let v = started_at as f64;
    (started_at > 0).then(|| {
        if v > 1e17 {
            v / 1e9
        } else if v > 1e11 {
            v / 1e3
        } else {
            v
        }
    })
}

/// The call's thread as its dispatch names it: the `thread_id` of the job's JSON metadata.
fn dispatched_thread(job: &livekit_protocol::Job) -> Option<String> {
    let metadata: serde_json::Value = serde_json::from_str(&job.metadata).ok()?;
    let thread = metadata.get("thread_id")?.as_str()?;
    (!thread.is_empty()).then(|| thread.to_owned())
}

/// Opens the job's root span and registers the job with the process exporter.
pub(crate) fn start_job(
    tracer: &Tracer,
    cloud: Option<&CloudEndpoint>,
    fields: &StartJobFields,
) -> JobTrace {
    let job = livekit_protocol::Job::decode(fields.job.as_slice()).unwrap_or_default();
    let room = job.room.clone().unwrap_or_default();
    let key = JobKey(job.id.clone());
    if let Some(exporter) = tracer.exporter() {
        if let Some(cloud) = cloud {
            exporter.configure_cloud(cloud.clone());
        }
        exporter.open_job(
            key.clone(),
            JobIdentity {
                room_id: room.sid.clone(),
                job_id: job.id.clone(),
                room_name: room.name.clone(),
                agent_name: Some(job.agent_name.clone()).filter(|name| !name.is_empty()),
            },
        );
    }
    let parent = fields.traceparent.as_deref().and_then(parse_traceparent);
    let cx = tracer
        .root()
        .continuing(parent)
        .with_job(Some(key.clone()))
        // `gen_ai.conversation.id`: the dispatched thread, else the room session.
        .with_conversation(
            dispatched_thread(&job)
                .as_deref()
                .or(Some(room.sid.as_str()).filter(|sid| !sid.is_empty())),
        );
    let started = tracer.now_nanos() as f64 / 1e9;
    let f = fields;
    let decl = JobEntrypoint {
        dispatch_id: Some(job.dispatch_id.clone()).filter(|id| !id.is_empty()),
        worker_id: Some(f.worker_id.clone()).filter(|id| !id.is_empty()),
        agent_id: job.state.as_ref().map(|s| s.agent_id.clone()).filter(|id| !id.is_empty()),
        accept_latency: gap(f.received_at, f.accepted_at),
        assignment_latency: gap(f.accepted_at, f.assigned_at),
        launch_latency: gap(f.assigned_at, f.launched_at),
        entrypoint_latency: gap(f.launched_at, started),
        dispatch_latency: gap(f.received_at, started),
    };
    let start = if f.received_at != 0.0 { nanos(f.received_at) } else { nanos(started) };
    let span = lk_agents_net::telemetry::start_at(&cx, decl, start);
    let stage = |at: f64| (at != 0.0).then(|| nanos(at));
    if let Some(at) = stage(f.received_at) {
        lk_agents_net::telemetry::event_at(&span, JobReceived, at);
    }
    if let Some(at) = stage(f.accepted_at) {
        lk_agents_net::telemetry::event_at(&span, JobAccepted, at);
    }
    if let Some(at) = stage(f.assigned_at) {
        lk_agents_net::telemetry::event_at(&span, JobAssigned, at);
    }
    if let Some(at) = stage(f.launched_at) {
        lk_agents_net::telemetry::event_at(&span, ProcessAssigned, at);
    }
    lk_agents_net::telemetry::event_at(&span, EntrypointStarted, nanos(started));
    if let Some(at) = job.state.as_ref().and_then(|s| server_started(s.started_at)) {
        lk_agents_net::telemetry::event_at(&span, JobStartedOnServer, nanos(at));
    }
    JobTrace { key, span, simulation: simulation(&job) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a unit of `JobState.started_at` misread, or an unknown stage counted as an instant.
    #[test]
    fn stages_and_server_start_are_read_in_their_units() {
        assert_eq!(server_started(1_700_000_000_000_000_000), Some(1_700_000_000.0));
        assert_eq!(server_started(1_700_000_000_000), Some(1_700_000_000.0));
        assert_eq!(server_started(1_700_000_000), Some(1_700_000_000.0));
        assert_eq!(server_started(0), None);
        assert_eq!(gap(0.0, 2.0), None);
        assert_eq!(gap(3.0, 2.0), Some(Seconds(0.0)));
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

    // Catches: the job's spans recording the room session as their conversation when the dispatch
    // names the call's thread.
    #[test]
    fn the_conversation_is_the_dispatched_thread() {
        let mut job = livekit_protocol::Job::default();
        assert_eq!(dispatched_thread(&job), None);
        job.metadata = r#"{"thread_id":"AT_1","other":2}"#.to_owned();
        assert_eq!(dispatched_thread(&job).as_deref(), Some("AT_1"));
        for metadata in ["not json", r#"{"thread_id":""}"#, r#"{"thread_id":3}"#] {
            job.metadata = metadata.to_owned();
            assert_eq!(dispatched_thread(&job), None, "{metadata}");
        }
    }
}

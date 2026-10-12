//! Worker control messages and conversions to the ABI job contracts.

use crate::{AcceptArguments, JobInfo, ServerType, WorkerPermissions};
use livekit_protocol as pb;
pub use livekit_protocol::agent_http::AgentEndpoint;
pub use livekit_protocol::server_message;
pub use livekit_protocol::{
    AvailabilityRequest, AvailabilityResponse, Job, JobAssignment, JobStatus, JobTermination,
    JobType, MigrateJobRequest, RegisterWorkerRequest, RegisterWorkerResponse, ServerInfo,
    ServerMessage, UpdateJobStatus, UpdateWorkerStatus,
    WorkerMessage, WorkerPing, WorkerPong,
};
use lk_agents_abi::ATTRIBUTE_AGENT_NAME;
use prost::Message as _;
use std::collections::HashMap;

fn server_type_proto(server_type: ServerType) -> JobType {
    match server_type {
        ServerType::Room => JobType::JtRoom,
        ServerType::Publisher => JobType::JtPublisher,
    }
}
/// `agent: true` is not a `WorkerPermissions` field but always set: it marks the participant as an
/// agent (deprecated in the proto, still read by the server).
fn permissions_proto(permissions: &WorkerPermissions) -> pb::ParticipantPermission {
    #[allow(deprecated)] // `agent`, see above
    pb::ParticipantPermission {
        can_subscribe: permissions.can_subscribe,
        can_publish: permissions.can_publish,
        can_publish_data: permissions.can_publish_data,
        can_publish_sources: permissions.can_publish_sources.clone(),
        hidden: permissions.hidden,
        can_update_metadata: permissions.can_update_metadata,
        agent: true,
        ..Default::default()
    }
}
pub fn job_info(job: &Job) -> JobInfo {
    JobInfo {
        id: job.id.clone(),
        dispatch_id: job.dispatch_id.clone(),
        ty: job.r#type,
        room: job.room.as_ref().map(|r| r.name.clone()),
        room_sid: job.room.as_ref().map(|r| r.sid.clone()),
        agent_name: job.agent_name.clone(),
        metadata: job.metadata.clone(),
        deployment: job.deployment.clone(),
        enable_recording: job.enable_recording,
        enable_redaction: job.enable_redaction,
        proto: encode_job(job),
    }
}
pub fn encode(msg: &WorkerMessage) -> Vec<u8> {
    msg.encode_to_vec()
}

/// The exact protobuf, so Python gets it rather than a lossy projection.
pub fn encode_job(job: &Job) -> Vec<u8> {
    job.encode_to_vec()
}

pub fn decode(bytes: &[u8]) -> Result<ServerMessage, prost::DecodeError> {
    ServerMessage::decode(bytes)
}

fn wrap(m: pb::worker_message::Message) -> WorkerMessage {
    WorkerMessage { message: Some(m) }
}

/// One WebTransport session, one stream per exchange. Declared only with routes, since it is what
/// requests the data plane.
pub const ENDPOINT_PROTOCOL: u32 = 1;

pub fn register(
    server_type: ServerType,
    agent_name: &str,
    deployment: &str,
    version: &str,
    permissions: &WorkerPermissions,
    endpoints: &[pb::agent_http::AgentEndpoint],
    instance_id: &str,
) -> WorkerMessage {
    wrap(pb::worker_message::Message::Register(RegisterWorkerRequest {
        r#type: server_type_proto(server_type) as i32,
        agent_name: agent_name.to_string(),
        version: version.to_string(),
        deployment: deployment.to_string(),
        allowed_permissions: Some(permissions_proto(permissions)),
        endpoints: endpoints.to_vec(),
        endpoint_protocol: if endpoints.is_empty() { 0 } else { ENDPOINT_PROTOCOL },
        instance_id: instance_id.to_string(),
        ..Default::default()
    }))
}

/// The capacity early-return leaves `terminate` false; an explicit reject sets it.
pub fn availability_reject(job_id: &str, terminate: bool) -> WorkerMessage {
    wrap(pb::worker_message::Message::Availability(AvailabilityResponse {
        job_id: job_id.to_string(),
        available: false,
        terminate,
        ..Default::default()
    }))
}

/// `lk.agent.name` is written first and user attributes merged over it, so the user's wins.
pub fn availability_accept(
    job_id: &str,
    agent_name: &str,
    args: &AcceptArguments,
) -> WorkerMessage {
    let mut attributes: HashMap<String, String> = HashMap::new();
    attributes.insert(ATTRIBUTE_AGENT_NAME.to_string(), agent_name.to_string());
    for (k, v) in &args.attributes {
        attributes.insert(k.clone(), v.clone());
    }
    wrap(pb::worker_message::Message::Availability(AvailabilityResponse {
        job_id: job_id.to_string(),
        available: true,
        participant_identity: args.identity.clone(),
        participant_name: args.name.clone(),
        participant_metadata: args.metadata.clone(),
        participant_attributes: attributes,
        ..Default::default()
    }))
}

/// Full, with zero load and the active job count.
pub fn update_worker_draining(job_count: u32) -> WorkerMessage {
    wrap(pb::worker_message::Message::UpdateWorker(UpdateWorkerStatus {
        status: Some(pb::WorkerStatus::WsFull as i32),
        load: 0.0,
        job_count,
        draining: true,
        ..Default::default()
    }))
}

/// `load` is the raw load, not the effective one, which already counts reserved slots.
pub fn update_worker(status: crate::WorkerStatus, load: f32, job_count: u32) -> WorkerMessage {
    let s = match status {
        crate::WorkerStatus::Available => pb::WorkerStatus::WsAvailable,
        crate::WorkerStatus::Full => pb::WorkerStatus::WsFull,
    };
    wrap(pb::worker_message::Message::UpdateWorker(UpdateWorkerStatus {
        status: Some(s as i32),
        load,
        job_count,
        ..Default::default()
    }))
}

pub fn update_job(job_id: &str, status: JobStatus) -> WorkerMessage {
    wrap(pb::worker_message::Message::UpdateJob(UpdateJobStatus {
        job_id: job_id.to_string(),
        status: status as i32,
        error: String::new(),
    }))
}

/// Sent once after registration: the jobs this worker already holds.
pub fn migrate_jobs(job_ids: Vec<String>) -> WorkerMessage {
    wrap(pb::worker_message::Message::MigrateJob(MigrateJobRequest { job_ids }))
}

/// The heartbeat is `WorkerPing`/`WorkerPong`, not the signal-protocol `Ping`/`Pong` beside them.
pub fn ping(timestamp: i64) -> WorkerMessage {
    wrap(pb::worker_message::Message::Ping(WorkerPing { timestamp }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AcceptArguments;

    // Catches: writing lk.agent.name after the user's attributes, overriding a per-job rename.
    #[test]
    fn user_attributes_win_over_the_agent_name_default() {
        let args = AcceptArguments {
            attributes: vec![(ATTRIBUTE_AGENT_NAME.into(), "override".into())],
            ..Default::default()
        };
        let msg = availability_accept("j1", "configured", &args);
        let Some(pb::worker_message::Message::Availability(a)) = msg.message else {
            panic!("wrong variant")
        };
        assert_eq!(a.participant_attributes[ATTRIBUTE_AGENT_NAME], "override");
    }

    // Catches: forgetting `agent: true` on the register permissions.
    #[test]
    fn register_marks_the_permission_as_an_agent() {
        let msg =
            register(ServerType::Room, "a", "d", "1.2.3", &WorkerPermissions::default(), &[], "i1");
        let Some(pb::worker_message::Message::Register(r)) = msg.message else {
            panic!("wrong variant")
        };
        #[allow(deprecated)]
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
}

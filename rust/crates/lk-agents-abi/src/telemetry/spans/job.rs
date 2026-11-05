//! The job: its entrypoint, shutdown and room connection.

use super::common::ParticipantAttrs;
use super::session::{AgentSession, SessionStart};
use crate::telemetry::decl::{Attrs, Content, External, Seconds, SpanValue, event, span};
use crate::telemetry::semconv::protocol;

/// The job, from its receipt (backdated) to the end of shutdown. Root of a job's trace, or a
/// continuation of the dispatch's `traceparent`.
#[span(
    name = "job_entrypoint",
    parent = External,
    events = [
        JobReceived, JobAccepted, JobAssigned, ProcessAssigned, EntrypointStarted,
        JobStartedOnServer, EntrypointReturned, EntrypointCancelled
    ]
)]
#[derive(Attrs, Default)]
pub struct JobEntrypoint {
    /// Agent dispatch id.
    #[attr("lk.dispatch_id")]
    pub dispatch_id: Option<String>,
    /// Worker id the job runs on.
    #[attr("lk.job.worker_id")]
    pub worker_id: Option<String>,
    /// Agent id of the job.
    #[attr("lk.job.agent_id")]
    pub agent_id: Option<String>,
    /// Seconds from the job request to its acceptance.
    #[attr("lk.job.accept_latency")]
    pub accept_latency: Option<Seconds>,
    /// Seconds from acceptance to the job assignment.
    #[attr("lk.job.assignment_latency")]
    pub assignment_latency: Option<Seconds>,
    /// Seconds from assignment to the job process launch.
    #[attr("lk.job.launch_latency")]
    pub launch_latency: Option<Seconds>,
    /// Seconds from launch to the entrypoint running.
    #[attr("lk.job.entrypoint_latency")]
    pub entrypoint_latency: Option<Seconds>,
    /// Seconds from the dispatch request to the entrypoint running.
    #[attr("lk.job.dispatch_latency")]
    pub dispatch_latency: Option<Seconds>,
}

/// The worker received the job request.
#[event(name = "job_received")]
#[derive(Attrs)]
pub struct JobReceived;

/// The worker accepted the job.
#[event(name = "job_accepted")]
#[derive(Attrs)]
pub struct JobAccepted;

/// The server assigned the job.
#[event(name = "job_assigned")]
#[derive(Attrs)]
pub struct JobAssigned;

/// A job process was assigned to it.
#[event(name = "process_assigned")]
#[derive(Attrs)]
pub struct ProcessAssigned;

/// The entrypoint started running.
#[event(name = "entrypoint_started")]
#[derive(Attrs)]
pub struct EntrypointStarted;

/// The server-side job start time, as the worker learned it.
#[event(name = "job_started_on_server")]
#[derive(Attrs)]
pub struct JobStartedOnServer;

/// The entrypoint returned.
#[event(name = "entrypoint_returned")]
#[derive(Attrs)]
pub struct EntrypointReturned;

/// The entrypoint was cancelled.
#[event(name = "entrypoint_cancelled")]
#[derive(Attrs)]
pub struct EntrypointCancelled;

/// The job shutting down.
#[span(name = "job_shutdown", parent = JobEntrypoint)]
#[derive(Attrs)]
pub struct JobShutdown {
    /// Why the job shut down.
    #[attr("lk.shutdown.reason")]
    pub reason: String,
    /// The application asked for the shutdown.
    #[attr("lk.shutdown.user_initiated")]
    pub user_initiated: bool,
}

/// The application's `on_session_end` callback, when one is registered.
#[span(name = "on_session_end", parent = JobShutdown)]
#[derive(Attrs)]
pub struct OnSessionEnd {
    /// Name of the application callback.
    #[attr("lk.callback.name")]
    pub callback: String,
}

/// Uploading the session report and recording.
#[span(name = "session_end_upload", parent = JobShutdown)]
#[derive(Attrs)]
pub struct SessionEndUpload;

/// Disconnecting from the room.
#[span(name = "room_disconnect", parent = JobShutdown)]
#[derive(Attrs)]
pub struct RoomDisconnect;

/// One application shutdown callback.
#[span(name = "shutdown_callback", parent = JobShutdown)]
#[derive(Attrs)]
pub struct ShutdownCallback {
    #[attr("lk.callback.name")]
    pub callback: String,
}

/// Connecting the job to its room.
#[span(name = "room_connect", parent = SessionStart | JobEntrypoint, end = RoomConnectEnd)]
#[derive(Attrs)]
pub struct RoomConnect {
    /// The room's subscription mode.
    #[attr("lk.room.auto_subscribe")]
    pub auto_subscribe: AutoSubscribe,
    /// End-to-end encryption is on.
    #[attr("lk.room.e2ee")]
    pub e2ee: bool,
}

#[derive(Attrs)]
pub struct RoomConnectEnd {
    #[attr(protocol::LK_PARTICIPANT_ID)]
    pub participant_id: String,
    #[attr(protocol::LK_PII_PARTICIPANT_IDENTITY)]
    pub participant_identity: Content<String>,
    /// Remote participants in the room once connected.
    #[attr("lk.room.remote_participant_count")]
    pub remote_participants: u32,
}

/// What the job subscribes to.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoSubscribe {
    SubscribeAll,
    SubscribeNone,
    AudioOnly,
    VideoOnly,
}

/// Waiting for the participant the session links to.
#[span(
    name = "wait_for_participant",
    parent = SessionStart | JobEntrypoint,
    end = Option<ParticipantAttrs>,
    outlives_parent = "session.start() returns before a participant is linked"
)]
#[derive(Attrs)]
pub struct WaitForParticipant {
    /// The room input filters participants by identity or kind.
    #[attr("lk.room_io.participant_filter")]
    pub participant_filter: bool,
}

/// Waiting for the linked participant's audio track and its first frame.
#[span(
    name = "wait_for_audio_track",
    parent = SessionStart | AgentSession,
    end = WaitForAudioTrackEnd,
    events = [TrackSubscribed, PreConnectAudio, FirstFrame],
    outlives_parent = "session.start() returns before the first frame"
)]
#[derive(Attrs)]
pub struct WaitForAudioTrack {
    #[attr(protocol::LK_PII_PARTICIPANT_IDENTITY)]
    pub participant_identity: Content<String>,
}

#[derive(Attrs)]
pub struct WaitForAudioTrackEnd {
    /// Seconds from the subscription to the first audio frame.
    #[attr("lk.first_frame_delay")]
    pub first_frame_delay: Option<Seconds>,
}

/// The participant's audio track was subscribed.
#[event(name = "track_subscribed")]
#[derive(Attrs)]
pub struct TrackSubscribed {
    #[attr(protocol::LK_TRACK_ID)]
    pub track_id: String,
    #[attr(protocol::LK_TRACK_SOURCE)]
    pub source: String,
}

/// Pre-connect audio was replayed into the session.
#[event(name = "pre_connect_audio")]
#[derive(Attrs)]
pub struct PreConnectAudio {
    /// Seconds of pre-connect audio replayed.
    #[attr("lk.pre_connect_audio.duration")]
    pub duration: Seconds,
}

/// The first audio frame arrived.
#[event(name = "first_frame")]
#[derive(Attrs)]
pub struct FirstFrame;

/// Publishing the agent's audio track.
#[span(
    name = "publish_audio_output",
    parent = SessionStart,
    end = PublishAudioOutputEnd,
    outlives_parent = "session.start() returns before the track is published"
)]
#[derive(Attrs)]
pub struct PublishAudioOutput;

#[derive(Attrs)]
pub struct PublishAudioOutputEnd {
    #[attr(protocol::LK_TRACK_ID)]
    pub track_id: String,
}

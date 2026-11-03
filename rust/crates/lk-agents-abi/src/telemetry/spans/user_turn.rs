//! The user's turn, from its first evidence to its outcome (`telemetry.md`, "The user turn").

use super::common::ParticipantAttrs;
use super::session::AgentSession;
use crate::telemetry::decl::{
    Attrs, BoolString, Content, Json, Measured, Seconds, SpanValue, event, span,
};
use crate::telemetry::semconv::otel;

/// One user turn, opened by its first evidence and ended with its outcome.
#[span(name = "user_turn", parent = AgentSession, end = UserTurnEnd)]
#[derive(Attrs)]
pub struct UserTurn {
    /// What opened the turn.
    #[attr("lk.user_turn.origin")]
    pub origin: UserTurnOrigin,
    #[attrs]
    pub participant: Option<ParticipantAttrs>,
    /// The STT.
    #[attr(otel::GEN_AI_REQUEST_MODEL, label = "model_name")]
    pub stt_model: Option<String>,
    /// The STT's provider.
    #[attr(otel::GEN_AI_PROVIDER_NAME, label = "model_provider")]
    pub stt_provider: Option<String>,
}

#[derive(Attrs)]
pub struct UserTurnEnd {
    /// How the turn ended.
    #[attr("lk.user_turn.outcome")]
    pub outcome: UserTurnOutcome,
    /// The user's transcript.
    #[attr("lk.pii.user_transcript")]
    pub transcript: Content<String>,
    /// The STT's confidence in the transcript.
    #[attr("lk.transcript_confidence")]
    pub confidence: Option<f64>,
    /// Seconds from the end of speech to the final transcript.
    #[attr(
        "lk.transcription_delay",
        unmeasured = 0,
        metric = "lk.agents.turn.transcription_delay"
    )]
    pub transcription_delay: Measured<Seconds>,
    /// Seconds from the end of speech to the turn decision.
    #[attr("lk.end_of_turn_delay", unmeasured = 0, metric = "lk.agents.turn.end_of_turn_delay")]
    pub end_of_turn_delay: Measured<Seconds>,
    /// Provider request ids the span covers, for joining against provider logs.
    #[attr("lk.provider_request_ids")]
    pub provider_request_ids: Vec<String>,
    /// The transcript events in arrival order: `received_at`, `type`, `transcript_length`.
    #[attr("lk.stt.events")]
    pub stt_events: Json<serde_json::Value>,
    /// Times the user resumed speaking before the decision, when any.
    #[attr("lk.eou.resume_count")]
    pub resumes: Option<u32>,
}

/// The first evidence of a turn.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserTurnOrigin {
    Speech,
    Transcript,
    ServerSpeech,
    Text,
}

/// How a user turn ended.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserTurnOutcome {
    Committed,
    TooFewWords,
    Empty,
    Backchannel,
    Dropped,
    Cleared,
    TranscriptionTimeout,
    Closed,
}

/// One speech segment of the user, with every overlap verdict on it.
#[span(name = "user_speaking", parent = UserTurn, end = UserSpeakingEnd)]
#[derive(Attrs)]
pub struct UserSpeaking {
    #[attrs]
    pub participant: Option<ParticipantAttrs>,
}

#[derive(Attrs)]
pub struct UserSpeakingEnd {
    /// The last overlap verdict: the speech interrupted the agent.
    #[attr("lk.is_interruption")]
    pub is_interruption: Option<BoolString>,
    /// The interruption detector's probability.
    #[attr("lk.interruption.probability")]
    pub probability: Option<f64>,
    /// Seconds of overlapping speech.
    #[attr("lk.interruption.total_duration")]
    pub total_duration: Option<Seconds>,
    /// Seconds the interruption prediction took.
    #[attr("lk.interruption.prediction_duration")]
    pub prediction_duration: Option<Seconds>,
    /// Seconds from the overlap's start to the verdict.
    #[attr("lk.interruption.detection_delay")]
    pub detection_delay: Option<Seconds>,
}

/// The end of speech to the end-of-turn decision.
#[span(
    name = "eou_wait",
    parent = UserTurn,
    end = EouWaitEnd,
    events = [Rearmed, UserResumed]
)]
#[derive(Attrs)]
pub struct EouWait {
    /// What decides the end of the turn.
    #[attr("lk.eou.source")]
    pub source: String,
    /// Seconds the endpointing waits.
    #[attr("lk.eou.endpointing_delay")]
    pub endpointing_delay: Seconds,
}

#[derive(Attrs)]
pub struct EouWaitEnd {
    /// How the wait ended.
    #[attr("lk.eou.outcome")]
    pub outcome: EouOutcome,
    /// Seconds the wait lasted.
    #[attr("lk.eou.wait_duration")]
    pub wait_duration: Seconds,
    /// Times the wait was re-armed.
    #[attr("lk.eou.rearm_count")]
    pub rearm_count: u32,
    /// Decisions that did not commit the turn.
    #[attr("lk.eou.not_committed_count")]
    pub not_committed_count: u32,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum EouOutcome {
    Committed,
    UserResumed,
    Dropped,
}

/// The wait was re-armed by a newer decision source.
#[event(name = "rearmed")]
#[derive(Attrs)]
pub struct Rearmed {
    #[attr("lk.eou.source")]
    pub source: String,
}

/// The user resumed speaking before the decision.
#[event(name = "user_resumed")]
#[derive(Attrs)]
pub struct UserResumed;

/// One end-of-turn prediction.
#[span(
    name = "eou_detection",
    parent = EouWait,
    end = Option<EouPrediction>,
    events = [Superseded]
)]
#[derive(Attrs)]
pub struct EouDetection;

#[derive(Attrs)]
pub struct EouPrediction {
    /// The conversation the detector read.
    #[attr("lk.pii.chat_ctx")]
    pub chat_ctx: Content<Json<serde_json::Value>>,
    #[attr("lk.eou.endpointing_delay")]
    pub endpointing_delay: Seconds,
    /// The language the detector predicted for.
    #[attr("lk.eou.language")]
    pub language: Option<String>,
    #[attr("lk.eou.source")]
    pub source: String,
    /// The prediction came from the detector's cache.
    #[attr("lk.eou.from_cache")]
    pub from_cache: bool,
    /// Probability the user finished their turn.
    #[attr("lk.eou.probability")]
    pub probability: f64,
    /// Below this probability the turn is unlikely to be over.
    #[attr("lk.eou.unlikely_threshold")]
    pub unlikely_threshold: Option<f64>,
    /// Seconds a streaming detector took to decide.
    #[attr("lk.eou.detection_delay")]
    pub detection_delay: Option<Seconds>,
}

/// A newer prediction replaced this one.
#[event(name = "superseded")]
#[derive(Attrs)]
pub struct Superseded {
    #[attr("lk.eou.outcome")]
    pub outcome: EouOutcome,
}

/// The agent's `on_user_turn_completed` hook, the committed turn's last child.
#[span(
    name = "on_user_turn_completed",
    parent = UserTurn,
    events = [StopResponse]
)]
#[derive(Attrs)]
pub struct OnUserTurnCompleted {
    #[attr("lk.agent_label")]
    pub agent_label: String,
}

/// The hook asked for no reply.
#[event(name = "stop_response")]
#[derive(Attrs)]
pub struct StopResponse;

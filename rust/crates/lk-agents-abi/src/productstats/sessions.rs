use crate::telemetry::decl::{Attrs, SpanValue, metric};

// A session is one call (`rtc_session`) or one text request (`a2a_session`).

/// The door a session came through.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entrypoint {
    RtcSession,
    A2aSession,
}

/// Sessions, counted once each when the session ends.
#[metric(
    name = "lk.product.agents.sessions",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct Sessions {
    /// A call or a text request.
    #[attr("lk.product.agents.session.entrypoint")]
    pub entrypoint: Entrypoint,
}

/// The models a voice session runs on.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipeline {
    SttLlmTts,
    Realtime,
    /// A realtime model with a separate TTS.
    RealtimeTts,
    Duplex,
}

/// What ends the person's turn.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDetection {
    Vad,
    Stt,
    RealtimeLlm,
    Manual,
    /// A turn detection model.
    Model,
}

/// How a session decides that the person is interrupting: the mode it runs, not the one asked.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interruption {
    /// The interruption model classifies overlapping speech.
    Adaptive,
    /// Any detected speech interrupts.
    Vad,
    /// Interruptions are disabled.
    Off,
}

/// Whether a false interruption pauses the agent's speech and resumes it.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseInterruption {
    On,
    Off,
}

/// Voice sessions, counted once each when the session starts, with what it starts with.
#[metric(
    name = "lk.product.agents.voice.sessions",
    kind = Counter,
    unit = "{session}",
    audience = ProductStats
)]
#[derive(Attrs)]
pub struct VoiceSessions {
    /// The models the session runs on.
    #[attr("lk.product.agents.voice.pipeline")]
    pub pipeline: Pipeline,
    /// What ends the person's turn.
    #[attr("lk.product.agents.voice.turn_detection")]
    pub turn_detection: TurnDetection,
    /// How the session decides that the person is interrupting.
    #[attr("lk.product.agents.voice.interruption")]
    pub interruption: Interruption,
    /// Whether a false interruption pauses the agent's speech and resumes it.
    #[attr("lk.product.agents.voice.false_interruption")]
    pub false_interruption: FalseInterruption,
}

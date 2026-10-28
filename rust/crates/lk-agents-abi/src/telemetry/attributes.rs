//! The `lk.*` wire vocabulary: protocol keys a frontend, a client SDK or LiveKit Cloud reads. Each
//! is declared once here; the core reads these names and the Python `livekit.agents.types`
//! constants are generated from them.

#[crate::host(file(module = "types", path = "livekit.agents.types"))]
const _: () = ();

#[crate::host(constant)]
pub const ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID: &str = "lk.segment_id";

#[crate::host(constant)]
pub const ATTRIBUTE_TRANSCRIPTION_TRACK_ID: &str = "lk.transcribed_track_id";

#[crate::host(constant)]
pub const ATTRIBUTE_TRANSCRIPTION_FINAL: &str = "lk.transcription_final";

/// The expression (delivery/emotion) the agent used for a transcription segment, surfaced so
/// the frontend can react to it, when expressive markup is stripped from the transcript. The
/// value is a JSON object ``{"value": ...}`` carrying the segment's leading expression -- the
/// ``<expression>`` tag for Inworld/ElevenLabs v3 or the ``<emotion>`` tag for Cartesia, e.g.
/// ``{"value": "speak happy"}``. A JSON object (rather than a bare string) so the shape can
/// gain fields later without breaking parsers.
#[crate::host(constant)]
pub const ATTRIBUTE_TRANSCRIPTION_EXPRESSION: &str = "lk.expression";

/// The identity of the agent participant that an avatar worker is publishing on behalf of.
#[crate::host(constant)]
pub const ATTRIBUTE_PUBLISH_ON_BEHALF: &str = "lk.publish_on_behalf";

/// The state of the agent, stored in the agent's attributes.
/// This can be retrieved on the client side by using `RemoteParticipant.attributes`.
///
/// With components-js, this can be easily retrieved using:
///
/// ```js
/// const { state, ... } = useVoiceAssistant();
/// ```
#[crate::host(constant)]
pub const ATTRIBUTE_AGENT_STATE: &str = "lk.agent.state";

/// The name of the agent, stored in the agent's attributes.
/// This is set when the agent joins a room and can be used to identify the agent type.
#[crate::host(constant)]
pub const ATTRIBUTE_AGENT_NAME: &str = "lk.agent.name";

/// Indicates that the participant is a simulator for testing purposes.
/// When set to "true", the agent will skip audio input/output processing.
#[crate::host(constant)]
pub const ATTRIBUTE_SIMULATOR: &str = "lk.simulator";

/// The job attribute carrying the run's protojson ``SimulationDispatch``,
/// delivered with the agent dispatch and read by
/// ``JobContext.simulation_context``.
#[crate::host(constant)]
pub const ATTRIBUTE_SIMULATOR_DISPATCH: &str = "lk.simulator.dispatch";

/// Telemetry metadata key marking the session as a simulation.
#[crate::host(constant)]
pub const ATTRIBUTE_SIMULATION_ENABLED: &str = "lk.simulation.enabled";

/// Telemetry metadata key naming the simulation run this session belongs to.
///
/// Sits alongside :data:`ATTRIBUTE_SIMULATION_ENABLED`: the boolean says a session is
/// synthetic, this says which run it came from, so a run's sessions can be aggregated
/// without resolving each one's room first.
#[crate::host(constant)]
pub const ATTRIBUTE_SIMULATION_RUN_ID: &str = "lk.simulation.run_id";

/// Telemetry metadata key naming the simulation run job (one scenario) behind the
/// session.
#[crate::host(constant)]
pub const ATTRIBUTE_SIMULATION_JOB_ID: &str = "lk.simulation.job_id";

/// Telemetry metadata key requesting PII redaction for the session.
#[crate::host(constant)]
pub const ATTRIBUTE_REDACTION_ENABLED: &str = "lk.redaction.enabled";

/// Text-stream topic for chat messages.
#[crate::host(constant)]
pub const TOPIC_CHAT: &str = "lk.chat";

/// Text-stream topic for transcription segments.
#[crate::host(constant)]
pub const TOPIC_TRANSCRIPTION: &str = "lk.transcription";

/// Minimum ``ParticipantInfo.client_protocol`` of a client that rebuilds transcription
/// events from ``lk.transcription`` text streams and ignores the deprecated ``rtc.Transcription``
/// data packet. While any considered participant is below this, the legacy packet is still
/// published. The value is defined by the client SDKs (``client-sdk-js`` ``src/version.ts``).
#[crate::host(constant)]
pub const CLIENT_PROTOCOL_TRANSCRIPTION_STREAMS: i32 = 3;

/// The key for the timed transcripts in the audio frame userdata.
#[crate::host(constant)]
pub const USERDATA_TIMED_TRANSCRIPT: &str = "lk.timed_transcripts";

/// The key for the time (``time.perf_counter()``) at which the synthesized text was first
/// sent to the TTS provider, attached to the audio frame userdata. Used to compute TTFB
/// without attributing upstream (e.g. LLM streaming) latency to the TTS.
#[crate::host(constant)]
pub const USERDATA_TTS_STARTED_TIME: &str = "lk.tts_started_time";

//! The `lk.*` wire vocabulary from `types.py`: protocol keys a frontend reads, declared again as
//! Python literals.

/// The `lk.segment_id` transcription attribute.
pub const ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID: &str = "lk.segment_id";
/// The `lk.transcribed_track_id` transcription attribute.
pub const ATTRIBUTE_TRANSCRIPTION_TRACK_ID: &str = "lk.transcribed_track_id";
/// The `lk.transcription_final` transcription attribute.
pub const ATTRIBUTE_TRANSCRIPTION_FINAL: &str = "lk.transcription_final";
/// The segment's leading expression (`<expression>`/`<emotion>` tag) when expressive markup is
/// stripped from the transcript, as a JSON object `{"value": ...}` so it can gain fields.
pub const ATTRIBUTE_TRANSCRIPTION_EXPRESSION: &str = "lk.expression";
/// The identity of the agent participant that an avatar worker is publishing on behalf of.
pub const ATTRIBUTE_PUBLISH_ON_BEHALF: &str = "lk.publish_on_behalf";
/// The state of the agent, stored in the agent's attributes. This can be retrieved on the client
/// side by using `RemoteParticipant.attributes`.
pub const ATTRIBUTE_AGENT_STATE: &str = "lk.agent.state";
/// The name of the agent, stored in the agent's attributes. This is set when the agent joins a
/// room and can be used to identify the agent type.
pub const ATTRIBUTE_AGENT_NAME: &str = "lk.agent.name";
/// Indicates that the participant is a simulator for testing purposes. When set to `"true"`, the
/// agent will skip audio input/output processing.
pub const ATTRIBUTE_SIMULATOR: &str = "lk.simulator";
/// The job attribute carrying the run's protojson `SimulationDispatch`, delivered with the agent
/// dispatch and read by `JobContext.simulation_context()`.
pub const ATTRIBUTE_SIMULATOR_DISPATCH: &str = "lk.simulator.dispatch";
/// Telemetry metadata key marking the session as a simulation.
pub const ATTRIBUTE_SIMULATION_ENABLED: &str = "lk.simulation.enabled";
/// Telemetry metadata key requesting PII redaction for the session.
pub const ATTRIBUTE_REDACTION_ENABLED: &str = "lk.redaction.enabled";

/// Data-channel topic for chat messages.
pub const TOPIC_CHAT: &str = "lk.chat";
/// Data-channel topic for transcription segments.
pub const TOPIC_TRANSCRIPTION: &str = "lk.transcription";

/// The key for the timed transcripts in the audio frame userdata.
pub const USERDATA_TIMED_TRANSCRIPT: &str = "lk.timed_transcripts";
/// The key for the time (`time.perf_counter()`) at which the synthesized text was first sent to
/// the TTS provider, attached to the audio frame userdata. Used to compute TTFB without
/// attributing upstream (e.g. LLM streaming) latency to the TTS.
pub const USERDATA_TTS_STARTED_TIME: &str = "lk.tts_started_time";

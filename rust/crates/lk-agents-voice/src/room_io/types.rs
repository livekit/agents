pub use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, RoomIoConfig, TextOutputConfig,
};
pub use lk_agents_rtc::{ParticipantIdentity, ParticipantKind, TrackPublishOptions, TrackSource};
use std::time::Duration;

/// `job.DEFAULT_PARTICIPANT_KINDS`: no `Agent` or `Egress`, so agents never auto-link to each other.
pub const DEFAULT_PARTICIPANT_KINDS: &[ParticipantKind] =
    &[ParticipantKind::Connector, ParticipantKind::Sip, ParticipantKind::Standard];

/// The track sources the audio input leg will read. Microphone only, as in the reference.
pub const AUDIO_INPUT_SOURCES: &[TrackSource] = &[TrackSource::Microphone];

/// `_pre_connect_audio.PRE_CONNECT_AUDIO_BUFFER_STREAM`.
pub const PRE_CONNECT_AUDIO_BUFFER_STREAM: &str = "lk.agent.pre-connect-audio-buffer";

/// `PreConnectAudioHandler._max_delta_s`: a buffer older than this is stale and dropped.
pub const PRE_CONNECT_MAX_DELTA: Duration = Duration::from_secs(1);

/// The kinds a [`RoomIoConfig`] auto-links: `participant_kinds`, or the defaults when empty.
pub fn accepted_kinds(config: &RoomIoConfig) -> &[ParticipantKind] {
    if config.participant_kinds.is_empty() {
        DEFAULT_PARTICIPANT_KINDS
    } else {
        &config.participant_kinds
    }
}

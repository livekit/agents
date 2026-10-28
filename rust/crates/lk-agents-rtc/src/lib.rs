#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy RTC sites are ratcheted."
)]

//! RTC capabilities supplied by the host SDK. Shared records live in `lk_agents_abi::voice::rtc`.

mod events;
mod fake;
mod playout;
mod traits;
mod wait;

pub use events::{
    AudioFrameSender, ByteChunks, ByteStream, ByteStreamSender, Receiver, RoomEvent, RoomEventBus, RpcHandler,
    TextChunks, TextStream, TextStreamSender, VideoFrameSender, assert_room_event_is_covered,
    room_event_coverage, room_event_vocabulary,
};
pub use fake::{
    ConnectCall, FakeAudioSource, FakeLocalParticipant, FakeRemoteParticipant, FakeRoom, FakeRtc,
    FakeTrackPublication, FrameSynth, RecordedByteStream, RecordedTextStream, Sent, StreamRequests, subscribe_local,
    tokio_test_block,
};
/// Re-exported so the 50 fps audio legs need no copy at this boundary.
pub use lk_agents_abi::AudioFrame;
pub use lk_agents_abi::voice::rtc::{
    AudioStreamOptions, AudioTrackFeature, ConnectionState, DEFAULT_QUEUE_SIZE_MS,
    DisconnectReason, ParticipantIdentity, ParticipantKind, PlaybackInfo, StreamInfo,
    StreamBytesOptions, StreamTextOptions, TRACK_KIND_AUDIO, TRACK_KIND_VIDEO, TrackPublishOptions, TrackSid, TrackSource, Transcription,
    TranscriptionSegment, UnknownDisconnectReason, UnknownWireValue, VIDEO_QUEUE_FRAMES,
    VideoFrameRef, frame_duration, silence_frame,
};
pub use playout::PlayoutLedger;
pub use lk_agents_abi::voice::rtc::RtcError;
pub use traits::{
    AudioSource, ByteStreamWriter, LocalParticipant, RemoteParticipant, RtcHost, RtcRoom, TextStreamWriter,
    TrackPublication,
};
pub use wait::{wait_for_agent, wait_for_participant, wait_for_participant_attribute, wait_for_track_publication};

pub type Result<T> = std::result::Result<T, RtcError>;

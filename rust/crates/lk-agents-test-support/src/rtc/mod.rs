//! Supplied-clock room peers for native unit tests and deterministic simulation.

pub mod events;
mod fake;
mod traits;
pub mod wait;

pub use fake::{
    ConnectCall, FakeAudioSource, FakeLocalParticipant, FakeRemoteParticipant, FakeRoom, FakeRtc,
    FakeTrackPublication, FrameSynth, RecordedByteStream, RecordedTextStream, Sent, StreamRequests,
    subscribe_local, tokio_test_block,
};
pub use lk_agents_abi::AudioFrame;
pub use lk_agents_abi::voice::rtc::events::{
    assert_room_event_is_covered, room_event_coverage, room_event_vocabulary,
};
pub use lk_agents_abi::voice::rtc::*;
pub use traits::RtcHost;

pub type Result<T> = std::result::Result<T, RtcError>;

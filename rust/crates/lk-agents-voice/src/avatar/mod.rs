use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use std::time::Duration;

pub mod effects;
pub mod machine;

pub const RPC_CLEAR_BUFFER: &str = "lk.clear_buffer";
pub const RPC_PLAYBACK_FINISHED: &str = "lk.playback_finished";
pub const RPC_PLAYBACK_STARTED: &str = "lk.playback_started";
pub const AUDIO_STREAM_TOPIC: &str = "lk.audio_stream";

pub fn stream_name(segment: PlaybackSegment) -> String {
    format!("AUDIO_{}_{}", segment.attachment, segment.segment)
}

/// The agent's playback identity in a 2.0 stream name; `None` for a name a v1 agent chose
/// (`AUDIO_<shortuuid>`), whose reports go to the plain methods.
pub fn stream_segment(name: &str) -> Option<PlaybackSegment> {
    let (attachment, segment) = name.strip_prefix("AUDIO_")?.split_once('_')?;
    Some(PlaybackSegment { attachment: attachment.parse().ok()?, segment: segment.parse().ok()? })
}

/// Where a report about `segment` goes: the 2.0 agent's route for that stream
/// (`<base>/<stream id>/<stream name>`), or the plain v1 method when the stream carried no
/// playback identity.
pub fn playback_method(base: &str, segment: &AvatarPlaybackSegment) -> String {
    match &segment.stream_id {
        Some(id) => format!("{base}/{id}/{}", stream_name(segment.playback)),
        None => base.to_owned(),
    }
}

/// `lk.playback_finished`'s payload, sent by the avatar worker's receiver.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FinishedPayload {
    #[serde(default)]
    pub playback_position: f64,
    #[serde(default)]
    pub interrupted: bool,
    #[serde(default)]
    pub synchronized_transcript: Option<String>,
}

/// What the avatar worker reads: audio, and where each segment ends.
pub type Received = lk_agents_abi::utils::video::AudioReceiverFrame;

pub use lk_agents_abi::voice::avatar::ReceiverEnd;

pub use lk_agents_abi::voice::avatar::ClearBufferListener;

/// With no chunk or trailer for this long, the last segment's trailer is assumed lost.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

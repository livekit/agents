//! Pure avatar segment identity, admission, framing and clear decisions.
use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use std::time::Duration;

pub mod output;
pub mod queue;
pub mod receiver;
pub mod runner;

pub const RPC_CLEAR_BUFFER: &str = "lk.clear_buffer";
pub const RPC_PLAYBACK_FINISHED: &str = "lk.playback_finished";
pub const RPC_PLAYBACK_STARTED: &str = "lk.playback_started";
pub const AUDIO_STREAM_TOPIC: &str = "lk.audio_stream";
pub const LOG: &str = "livekit.agents";

pub fn stream_name(segment: PlaybackSegment) -> String {
    format!("AUDIO_{}_{}", segment.attachment, segment.segment)
}

pub fn stream_segment(name: &str) -> std::result::Result<PlaybackSegment, ReceiverEnd> {
    let parsed = name.strip_prefix("AUDIO_").and_then(|name| name.split_once('_'));
    let Some((attachment, segment)) = parsed else {
        return Err(ReceiverEnd::BadStream("avatar stream name has no playback identity".into()));
    };
    let attachment = attachment
        .parse()
        .map_err(|_| ReceiverEnd::BadStream("invalid output attachment".into()))?;
    let segment =
        segment.parse().map_err(|_| ReceiverEnd::BadStream("invalid output segment".into()))?;
    Ok(PlaybackSegment { attachment, segment })
}

pub fn playback_method(
    base: &str,
    segment: &AvatarPlaybackSegment,
) -> std::result::Result<String, ReceiverEnd> {
    let id = segment
        .stream_id
        .as_ref()
        .ok_or_else(|| ReceiverEnd::BadStream("avatar report has no byte-stream id".into()))?;
    Ok(format!("{base}/{id}/{}", stream_name(segment.playback)))
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

/// Why reading stopped: closed by us (the end of iteration), or failed (the reference re-raises it
/// from iteration).
pub use lk_agents_abi::voice::avatar::ReceiverEnd;

pub use lk_agents_abi::voice::avatar::ClearBufferListener;

/// `STREAM_IDLE_TIMEOUT`: with no chunk or trailer for this long, the last segment's trailer is
/// assumed lost.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

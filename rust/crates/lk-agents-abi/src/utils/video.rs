//! Video as plain data, and the frames an avatar worker moves: records with their bytes, never the
//! host's objects. A frame's buffer is shared (`Bytes`), so moving it through the core copies
//! nothing; crossing in from a host `bytes` buffer copies nothing either, and crossing out is one
//! copy into the host's `bytes`.

use crate::AudioFrame;

/// `rtc.VideoFrame(width, height, type, data)`.
#[crate::host(handwritten(extract, emit, host = "livekit.rtc.VideoFrame", py))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// `rtc.VideoBufferType`, as the SDK numbers it.
    pub buffer_type: i32,
    pub data: bytes::Bytes,
}

/// A frame the SDK's synchronizer takes: audio or video.
#[crate::host(record(module = "avatar", extract, emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum MediaFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
}

/// What an avatar's video generator yields: a frame, or the end of a segment
/// (`AudioSegmentEnd`).
#[crate::host(record(module = "avatar", extract, emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum AvatarFrame {
    Audio(AudioFrame),
    Video(VideoFrame),
    #[host(host_type = "livekit.agents.voice.avatar.AudioSegmentEnd")]
    SegmentEnd,
}

/// An avatar audio receiver yields audio or the end of its segment, never video.
#[crate::host(record(module = "avatar", extract, emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum AudioReceiverFrame {
    Audio(AudioFrame),
    #[host(host_type = "livekit.agents.voice.avatar.AudioSegmentEnd")]
    SegmentEnd,
}

//! Video as plain data, and the frames an avatar worker moves: records with their bytes, never the
//! host's objects. A frame's buffer is shared (`Bytes`), so moving it through the core copies
//! nothing; crossing in from a host `bytes` buffer copies nothing either, and crossing out is one
//! copy into the host's `bytes`.

use crate::AudioFrame;

/// Marks the end of one avatar audio segment.
#[crate::host(record(module = "avatar", path = "livekit.agents.voice.avatar", marker, emit))]
#[derive(Debug, Clone, Copy)]
pub struct AudioSegmentEnd {}

/// `rtc.VideoFrame(width, height, type, data)`.
#[crate::host(record(
    module = "rtc_frames",
    name = "VideoFrame",
    host_type = "livekit.rtc.VideoFrame",
    boundary,
    extract,
    emit,
    error_owner = "VideoFrameFromPy"
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// `rtc.VideoBufferType`, as the SDK numbers it.
    #[host(rename = "type")]
    pub buffer_type: i32,
    #[host(read_as = Vec<u8>, extract_with = "py_values::video_bytes")]
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

use crate::AudioFrame;

// A frame's buffer is shared (`Bytes`): moving it through the core or in from a host `bytes`
// buffer copies nothing; crossing out is one copy into the host's `bytes`.

/// Marks the end of one avatar audio segment.
#[crate::host(record(module = "avatar", path = "livekit.agents.voice.avatar", marker, emit))]
#[derive(Debug, Clone, Copy)]
pub struct AudioSegmentEnd {}

/// `rtc.VideoBufferType`, crossing as the SDK's number.
#[crate::host(codec(extract, emit, host = "int", via = i32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoBufferType {
    Rgba,
    Abgr,
    Argb,
    Bgra,
    Rgb24,
    I420,
    I420a,
    I422,
    I444,
    I010,
    Nv12,
}

impl VideoBufferType {
    /// Every type, indexed by the SDK's number.
    pub const ALL: [VideoBufferType; 11] = [
        VideoBufferType::Rgba,
        VideoBufferType::Abgr,
        VideoBufferType::Argb,
        VideoBufferType::Bgra,
        VideoBufferType::Rgb24,
        VideoBufferType::I420,
        VideoBufferType::I420a,
        VideoBufferType::I422,
        VideoBufferType::I444,
        VideoBufferType::I010,
        VideoBufferType::Nv12,
    ];
}

impl From<VideoBufferType> for i32 {
    fn from(value: VideoBufferType) -> i32 {
        VideoBufferType::ALL.iter().position(|v| *v == value).expect("every type is in ALL") as i32
    }
}

impl TryFrom<i32> for VideoBufferType {
    type Error = crate::voice::rtc::UnknownWireValue;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        usize::try_from(value)
            .ok()
            .and_then(|i| VideoBufferType::ALL.get(i).copied())
            .ok_or(crate::voice::rtc::UnknownWireValue { kind: "VideoBufferType", value })
    }
}

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
    #[host(rename = "type")]
    pub buffer_type: VideoBufferType,
    #[host(read_as = Vec<u8>, extract_with = "py_values::video_bytes")]
    pub data: bytes::Bytes,
}

/// A host `rtc.VideoFrame` whose pixels stay in the host until an image of it is made. Two
/// handles are equal only when one is a clone of the other: each read of a host frame is its own.
#[crate::host(proxy(host = "livekit.rtc.VideoFrame"))]
pub trait RtcVideoFrame: std::any::Any + Send + Sync {
    /// The frame's pixels as RGBA: the frame itself when it already is, else its conversion.
    #[host(call, function = "livekit.agents._bindings.rgba_frame")]
    fn rgba(&self) -> Result<VideoFrame, crate::errors::HostError>;
}

impl PartialEq for dyn RtcVideoFrame {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self, other)
    }
}

impl Eq for dyn RtcVideoFrame {}

impl std::fmt::Debug for dyn RtcVideoFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RtcVideoFrame({:p})", self)
    }
}

/// A host frame cannot be persisted: durable media is supplied first.
#[cfg(feature = "serde")]
pub mod live_frame {
    use std::sync::Arc;

    use super::RtcVideoFrame;

    pub fn serialize<S: serde::Serializer>(
        _: &Arc<dyn RtcVideoFrame>,
        _: S,
    ) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("live media cannot be persisted; supply durable media first"))
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        _: D,
    ) -> Result<Arc<dyn RtcVideoFrame>, D::Error> {
        Err(serde::de::Error::custom("persisted state cannot contain a live media handle"))
    }
}

/// A frame with no host behind it, told apart by its number.
#[cfg(any(test, feature = "testkit"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeFrame(pub u64);

#[cfg(any(test, feature = "testkit"))]
impl FakeFrame {
    pub fn shared(id: u64) -> std::sync::Arc<dyn RtcVideoFrame> {
        std::sync::Arc::new(FakeFrame(id))
    }

    /// The number of a fake frame, `None` for any other.
    pub fn id_of(frame: &dyn RtcVideoFrame) -> Option<u64> {
        (frame as &dyn std::any::Any).downcast_ref::<FakeFrame>().map(|f| f.0)
    }
}

#[cfg(any(test, feature = "testkit"))]
impl RtcVideoFrame for FakeFrame {
    fn rgba(&self) -> Result<VideoFrame, crate::errors::HostError> {
        Err(crate::errors::HostError::from(format!("fake frame {} has no pixels", self.0)))
    }
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

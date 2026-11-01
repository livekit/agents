//! Reusable RTC source capabilities. Scheduling belongs to voice and execution to runtime.

use super::{RtcError, TrackSource};
use crate::utils::video::{MediaFrame, VideoFrame};
use std::any::Any;

#[crate::host(record(module = "rtc_wire", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoTrackOptions {
    pub name: String,
    pub source: TrackSource,
    pub width: u32,
    pub height: u32,
}

#[crate::host(proxy(host = "livekit.agents.voice.room_io.VideoSource"))]
pub trait VideoSource: Any + Send + Sync {
    #[host(call, name = "_core_capture")]
    fn capture(&self, frame: &VideoFrame) -> Result<(), RtcError>;
    #[host(await, name = "_core_close")]
    fn close(&self) -> crate::BoxFuture<'_, Result<(), RtcError>>;
}

/// The native synchronizer view returned by an avatar runner. Both media types use the shared
/// RTC sources; no language-specific avatar media manager participates in its lifetime.
#[crate::host(class(name = "AVSynchronizer"))]
pub trait AVSynchronizer: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn push(
        &self,
        frame: MediaFrame,
        #[host(default = None)] timestamp: Option<f64>,
    ) -> Result<(), RtcError>;
    #[host(await)]
    async fn clear_queue(&self) -> Result<(), RtcError>;
    #[host(await)]
    async fn wait_for_playout(&self) -> Result<(), RtcError>;
    #[host(call)]
    fn reset(&self);
    #[host(getter)]
    fn actual_fps(&self) -> f64;
    #[host(getter)]
    fn last_video_time(&self) -> f64;
    #[host(getter)]
    fn last_audio_time(&self) -> Result<f64, RtcError>;
    #[host(await)]
    async fn aclose(&self) -> Result<(), RtcError>;
}

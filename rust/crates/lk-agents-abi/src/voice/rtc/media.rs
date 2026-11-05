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

/// An `rtc.VideoSource`: frames captured as they come; pacing is the native synchronizer's.
#[crate::host(proxy(host = "livekit.rtc.VideoSource"))]
pub trait VideoSource: Any + Send + Sync {
    #[host(call, name = "capture_frame")]
    fn capture(&self, frame: &VideoFrame) -> Result<(), RtcError>;
    #[host(await, name = "aclose")]
    fn close(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), RtcError>>;
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

use super::PlaybackInfo;
use std::time::Duration;

#[crate::host(class(name = "PlayoutLedger"))]
pub trait PlayoutLedger: Sized + Send + Sync + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn add_pushed_samples(&self, samples_per_channel: u32, sample_rate: u32);
    #[host(call)]
    fn add_discarded(&self, duration: Duration);
    #[host(call)]
    fn snapshot(&self, queued: Duration) -> PlaybackInfo;
    #[host(call)]
    fn settle(&self) -> PlaybackInfo;
}

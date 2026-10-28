//! Host-facing source accounting; synchronization and storage belong to the runtime.

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

//! Runtime ownership of native avatar audio; SDK conversion stays in the host binding.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use lk_agents_abi::utils::spawner::TaskSet;
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, TokioSpawner};

use lk_agents_abi::voice::io::{AudioOutput, PlaybackTracker, Result};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, Closed};

pub struct DataStreamAudioOutput {
    output: Arc<lk_agents_voice::avatar::DataStreamAudioOutput>,
    spawner: TokioSpawner,
    cleanup: Arc<TaskSet>,
    cleanup_clock: Arc<dyn Clock>,
    incomplete: Arc<AtomicBool>,
}

impl DataStreamAudioOutput {
    pub async fn aclose(&self) {
        close_observed(&self.output, &*self.cleanup_clock, &self.incomplete).await;
    }

    pub fn from_native(
        output: lk_agents_voice::avatar::DataStreamAudioOutput,
        spawner: TokioSpawner,
        cleanup_clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            output: Arc::new(output),
            spawner,
            cleanup: Arc::default(),
            cleanup_clock,
            incomplete: Arc::default(),
        }
    }
}

async fn close_observed(
    output: &lk_agents_voice::avatar::DataStreamAudioOutput,
    clock: &dyn Clock,
    incomplete: &AtomicBool,
) {
    let outcome = lk_agents_abi::timeout(clock, HOST_TEARDOWN_TIMEOUT, output.aclose())
        .await
        .unwrap_or(Closed::TimedOut);
    if outcome != Closed::Clean && !incomplete.swap(true, Ordering::AcqRel) {
        log::error!("avatar output cleanup did not settle cleanly: {outcome:?}");
    }
}

impl Drop for DataStreamAudioOutput {
    fn drop(&mut self) {
        let output = Arc::clone(&self.output);
        let clock = Arc::clone(&self.cleanup_clock);
        let incomplete = Arc::clone(&self.incomplete);
        // The process executor owns this task independently of the host loop. TaskSet retains
        // its named ownership until settlement; the stream's close task survives the deadline.
        // This does not keep the SDK loop or process alive: explicit close while the room is
        // usable remains the primary path.
        self.cleanup.spawn(&self.spawner, "avatar_output_drop", async move {
            close_observed(&output, &*clock, &incomplete).await;
        });
    }
}

impl AudioOutput for DataStreamAudioOutput {
    fn label(&self) -> &str {
        self.output.label()
    }
    fn tracker(&self) -> &Arc<PlaybackTracker> {
        self.output.tracker()
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        self.output.capabilities()
    }
    fn sample_rate(&self) -> Option<u32> {
        self.output.sample_rate()
    }
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.output.capture(frame).await })
    }
    fn flush_segment(&self) {
        self.output.flush_segment();
    }
    fn clear_buffer(&self) {
        self.output.clear_buffer();
    }
    fn pause(&self) {
        self.output.pause();
    }
    fn resume(&self) {
        self.output.resume();
    }
}

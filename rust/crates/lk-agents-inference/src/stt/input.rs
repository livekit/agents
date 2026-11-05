//! A recognition stream's audio input: a `CatchUp` media queue its pump drains at the pace of
//! the provider, so a slow provider never fails the stream; ending the input drops the sender.

use lk_agents_abi::AudioFrame;
use lk_agents_media::media_queue::{self, Policy, Receiver, Sender, Stuck};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::telemetry;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::time::Duration;

/// Everything up to 10 s: a slow provider hears what was said, late.
pub(crate) const INPUT_POLICY: Policy = Policy::CatchUp(Duration::from_secs(10));

/// The `_FlushSentinel`, in order with the audio: the end of a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Flush;

pub(crate) struct AudioInput {
    audio: Mutex<Option<Sender<AudioFrame, Flush>>>,
    drops: AudioDrops,
}

impl AudioInput {
    pub(crate) fn new() -> (Self, Receiver<AudioFrame, Flush>) {
        let (audio, rx) = media_queue::audio(INPUT_POLICY);
        let drops = AudioDrops::new(telemetry::AudioConsumer::Stt, INPUT_POLICY);
        (AudioInput { audio: Mutex::new(Some(audio)), drops }, rx)
    }

    /// Measures drops under `cx`'s job.
    pub(crate) fn observe(&self, cx: telemetry::Cx) {
        self.drops.observe(cx);
    }

    /// `false` once the input ended or its pump is gone.
    pub(crate) fn push(&self, frame: AudioFrame) -> bool {
        let Some(audio) = self.open() else { return false };
        self.drops.record(audio.push(frame));
        true
    }

    /// `None` once the input ended or its pump is gone.
    pub(crate) fn flush(&self) -> Option<Result<(), Stuck>> {
        self.open().map(|audio| audio.event(Flush))
    }

    pub(crate) fn end(&self) {
        drop(self.audio.lock().take());
    }

    fn open(&self) -> Option<MappedMutexGuard<'_, Sender<AudioFrame, Flush>>> {
        MutexGuard::try_map(self.audio.lock(), |audio| {
            audio.as_mut().filter(|audio| !audio.is_closed())
        })
        .ok()
    }
}

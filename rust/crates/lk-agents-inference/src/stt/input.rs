//! A recognition stream's audio input: a `CatchUp` media queue its pump drains at the pace of
//! the provider, so a slow provider never fails the stream; ending the input drops the sender.

use lk_agents_abi::{AudioFrame, Clock};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Policy, Stuck};
use lk_agents_telemetry as telemetry;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::sync::Arc;
use std::time::Duration;

/// Everything up to 10 s: a slow provider hears what was said, late.
pub(crate) const INPUT_POLICY: Policy = Policy::CatchUp(Duration::from_secs(10));

/// The `_FlushSentinel`, in order with the audio: the end of a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Flush;

pub(crate) struct AudioInput {
    audio: Mutex<Option<AudioSender<Flush>>>,
}

impl AudioInput {
    pub(crate) fn new(clock: Arc<dyn Clock>) -> (Self, AudioReceiver<Flush>) {
        let (audio, rx) = media_queue::audio(INPUT_POLICY, telemetry::AudioConsumer::Stt, clock);
        (AudioInput { audio: Mutex::new(Some(audio)) }, rx)
    }

    /// Measures the queue under `cx`'s job.
    pub(crate) fn observe(&self, cx: telemetry::Cx) {
        if let Some(audio) = self.audio.lock().as_ref() {
            audio.observe(cx);
        }
    }

    /// `false` once the input ended or its pump is gone.
    pub(crate) fn push(&self, frame: AudioFrame) -> bool {
        let Some(audio) = self.open() else { return false };
        audio.push(frame);
        true
    }

    /// Filler, dropped before any other audio. `false` once the input ended or its pump is gone.
    pub(crate) fn filler(&self, frame: AudioFrame) -> bool {
        let Some(audio) = self.open() else { return false };
        audio.filler(frame);
        true
    }

    /// Pre-connect audio, kept in full. `false` once the input ended or its pump is gone.
    pub(crate) fn head(&self, frame: AudioFrame) -> bool {
        let Some(audio) = self.open() else { return false };
        audio.head(frame);
        true
    }

    /// `None` once the input ended or its pump is gone.
    pub(crate) fn flush(&self) -> Option<Result<(), Stuck>> {
        self.open().map(|audio| audio.event(Flush))
    }

    pub(crate) fn end(&self) {
        drop(self.audio.lock().take());
    }

    fn open(&self) -> Option<MappedMutexGuard<'_, AudioSender<Flush>>> {
        MutexGuard::try_map(self.audio.lock(), |audio| {
            audio.as_mut().filter(|audio| !audio.is_closed())
        })
        .ok()
    }
}

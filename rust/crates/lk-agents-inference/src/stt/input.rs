use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{AudioFrame, Clock};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Policy};
use lk_agents_net::source::InputGate;
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
    /// What a refused call answers: the recognition's failure, else its end.
    gate: InputGate,
}

impl AudioInput {
    pub(crate) fn new(clock: Arc<dyn Clock>, gate: InputGate) -> (Self, AudioReceiver<Flush>) {
        let (audio, rx) = media_queue::audio(INPUT_POLICY, telemetry::AudioConsumer::Stt, clock);
        (AudioInput { audio: Mutex::new(Some(audio)), gate }, rx)
    }

    /// Measures the queue under `cx`'s job.
    pub(crate) fn observe(&self, cx: telemetry::Cx) {
        if let Some(audio) = self.audio.lock().as_ref() {
            audio.observe(cx);
        }
    }

    pub(crate) fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.open()?.push(frame);
        Ok(())
    }

    /// Filler, dropped before any other audio.
    pub(crate) fn filler(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.open()?.filler(frame);
        Ok(())
    }

    /// Pre-connect audio, kept in full.
    pub(crate) fn head(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.open()?.head(frame);
        Ok(())
    }

    pub(crate) fn flush(&self) -> Result<(), ProviderError> {
        self.open()?.event(Flush).map_err(|stuck| {
            ApiError::new(lk_agents_abi::errors::ErrorKind::InvalidRequest, stuck.to_string())
                .into()
        })
    }

    pub(crate) fn end(&self) {
        self.gate.end();
        drop(self.audio.lock().take());
    }

    /// The queue while the input is open and its pump reads it; else the gate's answer.
    fn open(&self) -> Result<MappedMutexGuard<'_, AudioSender<Flush>>, ProviderError> {
        self.gate.admit()?;
        MutexGuard::try_map(self.audio.lock(), |audio| {
            audio.as_mut().filter(|audio| !audio.is_closed())
        })
        .map_err(|unopen| {
            drop(unopen);
            self.gate.refused()
        })
    }
}

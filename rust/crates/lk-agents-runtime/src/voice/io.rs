use super::session::Timebase;
use super::transcription::{TranscriptSyncConfig, TranscriptSynchronizer};
use lk_agents_abi::voice::io::{AudioOutput, TranscriptionOutput};
use lk_agents_abi::voice::room_owner::RoomIo;
use lk_agents_abi::{HOST_TEARDOWN_TIMEOUT, Spawner};
use std::sync::Arc;

/// A supplied or RTC output pair, with the wrapper that must close before its physical sinks.
#[derive(Clone)]
pub(super) struct SessionOutputs {
    pub audio: Option<Arc<dyn AudioOutput>>,
    pub text: Option<Arc<dyn TranscriptionOutput>>,
    synchronizer: Option<Arc<TranscriptSynchronizer>>,
    time: Timebase,
}

impl SessionOutputs {
    pub fn new(
        room: &dyn RoomIo,
        audio: Option<Arc<dyn AudioOutput>>,
        text: Option<Arc<dyn TranscriptionOutput>>,
        speed: f64,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let external_audio = audio.is_some();
        let audio = audio.or_else(|| room.audio_output());
        let text = text.or_else(|| room.transcription_output());
        let (audio, text, synchronizer) = match (audio, text) {
            (Some(audio), Some(text)) if room.wants_transcript_sync(external_audio) => {
                let sync = TranscriptSynchronizer::new(TranscriptSyncConfig {
                    audio_next: audio,
                    text_next: Some(text),
                    speed,
                    clock: time.clock(),
                    spawner,
                });
                (Some(sync.audio_output()), Some(sync.transcription_output()), Some(sync))
            }
            (audio, text) => (audio, text, None),
        };
        Self { audio, text, synchronizer, time }
    }

    pub async fn aclose(&self) {
        if let Some(sync) = &self.synchronizer
            && self.time.timeout(HOST_TEARDOWN_TIMEOUT, sync.aclose()).await.is_err()
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, timeout_ms = HOST_TEARDOWN_TIMEOUT.as_millis() as u64; "transcript synchronizer cleanup timed out");
        }
    }
}

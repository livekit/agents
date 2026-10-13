//! Recognition: every `stream()` the product opens is its own invocation with its own event
//! channel. A reopen is a new `RequestId`; the old one stays addressable so a late delivery to
//! it is a recorded, rejected attempt rather than a silent channel swap. Microphone audio the
//! driver forwards is recorded, never interpreted; recognition events come only from the case.
use std::sync::Arc;

use lk_agents_abi::{
    AudioBuffer, LanguageCode, SpeechEvent, Stt, SttCapabilities, SttSink, SttSource, Ts,
    errors::ProviderError,
    utils::stream::{Closed, StreamSource},
};
use tokio::sync::mpsc;

use super::{Control, Forward, Registry, Slot};
use crate::{
    input::{Input, Speaker, UserText},
    record::{Detail, Kind},
};

pub struct FakeStt {
    registry: Arc<Registry>,
    /// How long opening a stream takes, on the session's clock: a real recogniser connects
    /// before it listens, and a driver that reopens or closes during that window has an open
    /// in flight to deal with.
    connect: std::time::Duration,
}

impl FakeStt {
    pub fn new(registry: Arc<Registry>, connect: std::time::Duration) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect })
    }
}

struct Recognition {
    slot: Slot,
    registry: Arc<Registry>,
    events: mpsc::UnboundedReceiver<Result<SpeechEvent, ProviderError>>,
}

#[async_trait::async_trait]
impl Stt for FakeStt {
    fn label(&self) -> String {
        "fake::stt".into()
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, interim_results: true, ..Default::default() }
    }

    async fn recognize(
        &self,
        _: AudioBuffer<'_>,
        _: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        unreachable!("streaming fixture")
    }

    async fn stream(
        &self,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        if !self.connect.is_zero() {
            tokio::time::sleep(self.connect).await;
        }
        let (sender, events) = mpsc::unbounded_channel();
        let opened = self.registry.now();
        let slot =
            self.registry.register(Kind::Stt, Detail::None, Control::Stt(sender, opened, None));
        Some(Ok((
            Box::new(Forward { slot, registry: self.registry.clone() }),
            Box::new(Recognition { slot, registry: self.registry.clone(), events }),
        )))
    }
}

#[async_trait::async_trait]
impl StreamSource<SpeechEvent> for Recognition {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        self.events.recv().await.transpose()
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(self: Box<Self>, _: Ts) -> Closed {
        self.registry.mark_closed(self.slot);
        Closed::Clean
    }
}

/// Where a transcript sits in the stream's audio: a real recogniser stamps its words with the
/// seconds of stream it had heard when they began and ended, and the driver's speaker filter
/// locates the utterance's audio window by exactly those.
pub(super) struct Clock {
    pub opened: Ts,
    pub started: Option<Ts>,
    pub now: Ts,
}

impl Clock {
    fn seconds(&self, at: Ts) -> f64 {
        at.saturating_duration_since(self.opened).as_secs_f64()
    }
}

/// One recognition event onto a registered stream. True when the product still reads it.
pub(super) fn deliver(
    sender: &mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>,
    what: &Input,
    clock: Clock,
) -> bool {
    use lk_agents_abi::SpeechEventType as T;
    if let Input::SttFail { .. } = what {
        let error = ProviderError::Local(lk_agents_abi::errors::ApiError::new(
            "recogniser failed mid-stream",
        ));
        return sender.send(Err(error)).is_ok();
    }
    let (kind, text, speaker) = match what {
        Input::SttStart { .. } => (T::StartOfSpeech, None, None),
        Input::SttInterim { text: UserText(t), speaker, .. } => {
            (T::InterimTranscript, Some(t), *speaker)
        }
        Input::SttPreflight { text: UserText(t), speaker, .. } => {
            (T::PreflightTranscript, Some(t), *speaker)
        }
        Input::SttFinal { text: UserText(t), speaker, .. } => {
            (T::FinalTranscript, Some(t), *speaker)
        }
        Input::SttEnd { .. } => (T::EndOfSpeech, None, None),
        _ => return false,
    };
    let mut ev = SpeechEvent::new(kind);
    if let Some(text) = text {
        let mut data = lk_agents_abi::SpeechData::new("en", text.clone());
        data.speaker_id = speaker.map(|Speaker(n)| format!("speaker-{n}"));
        data.start_time = clock.seconds(clock.started.unwrap_or(clock.now));
        data.end_time = clock.seconds(clock.now);
        ev.alternatives.push(data);
    }
    sender.send(Ok(ev)).is_ok()
}

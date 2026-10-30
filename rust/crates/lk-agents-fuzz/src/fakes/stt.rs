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
use tokio::sync::{Notify, mpsc};

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
    /// What its streams are recorded as: the session's, or AMD's own.
    detail: Detail,
    /// `STTCapabilities.keyterms`: whether keyterm detection runs over it.
    keyterms: bool,
}

impl FakeStt {
    pub fn new(registry: Arc<Registry>, connect: std::time::Duration) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect, detail: Detail::None, keyterms: false })
    }

    /// A recogniser that takes keyterms, as most streaming providers do.
    pub fn taking_keyterms(registry: Arc<Registry>, connect: std::time::Duration) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect, detail: Detail::None, keyterms: true })
    }

    /// `AMD(stt=..)`: a recogniser the detector feeds through its own listening gate.
    pub fn for_amd(registry: Arc<Registry>, connect: std::time::Duration) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect, detail: Detail::AmdStt, keyterms: false })
    }
}

struct Recognition {
    slot: Slot,
    registry: Arc<Registry>,
    events: mpsc::UnboundedReceiver<Result<SpeechEvent, ProviderError>>,
    ended: Arc<Notify>,
}

impl Stt for FakeStt {
    fn label(&self) -> String {
        "fake::stt".into()
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            streaming: true,
            interim_results: true,
            keyterms: self.keyterms,
            ..Default::default()
        }
    }

    fn recognize(
        &self,
        _: AudioBuffer<'_>,
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!("streaming fixture") })
    }

    fn stream(
        &self,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move {
            if !self.connect.is_zero() {
                tokio::time::sleep(self.connect).await;
            }
            let (sender, events) = mpsc::unbounded_channel();
            let ended = Arc::new(Notify::new());
            let opened = self.registry.now();
            let slot = self.registry.register(
                Kind::Stt,
                self.detail.clone(),
                Control::Stt(sender, opened, None),
            );
            Ok(Some((
                Box::new(Forward {
                    slot,
                    registry: self.registry.clone(),
                    ended: Some(ended.clone()),
                }) as SttSink,
                Box::new(Recognition { slot, registry: self.registry.clone(), events, ended })
                    as SttSource,
            )))
        })
    }
}

impl StreamSource<SpeechEvent> for Recognition {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            // A real stream ends once its input has: what was already produced first, then the end.
            // Without it a stream the driver replaced stays open, and a case keeps speaking into it.
            tokio::select! {
                biased;
                event = self.events.recv() => event.transpose(),
                () = self.ended.notified() => Ok(None),
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

/// Where a transcript sits in the stream's audio: a real recogniser stamps its words with the
/// seconds of stream it had heard when they began and ended.
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
        Input::SttUsage { .. } => (T::RecognitionUsage, None, None),
        _ => return false,
    };
    let mut ev = SpeechEvent::new(kind);
    if kind == T::RecognitionUsage {
        // The audio the stream has heard so far, as the providers bill it.
        ev.recognition_usage = Some(lk_agents_abi::RecognitionUsage {
            audio_duration: Some(clock.seconds(clock.now)),
            ..Default::default()
        });
    }
    if let Some(text) = text {
        let mut data = lk_agents_abi::SpeechData::new("en", text.clone());
        data.speaker_id = speaker.map(|Speaker(n)| format!("speaker-{n}"));
        data.start_time = clock.seconds(clock.started.unwrap_or(clock.now));
        data.end_time = clock.seconds(clock.now);
        ev.alternatives.push(data);
    }
    sender.send(Ok(ev)).is_ok()
}

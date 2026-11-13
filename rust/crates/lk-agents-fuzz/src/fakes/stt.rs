use std::sync::Arc;

use lk_agents_abi::{
    AudioFrame, LanguageCode, SpeechEvent, Stt, SttCapabilities, SttSink, SttSource, Ts,
    errors::ProviderError,
    utils::stream::{Closed, StreamOutput},
};
use tokio::sync::{Notify, mpsc};

use super::{
    Control, Forward, Registry, Slot,
    hazard::{Site, strike},
};
use crate::{
    input::{Input, Speaker, UserText},
    record::{Detail, Kind},
};

// A reopened stream stays addressable, so a late delivery to it is a recorded, rejected attempt
// rather than a silent channel swap. Audio is recorded, never interpreted. As a real provider's,
// a stream whose input ended still delivers the utterance it is in, and closes after its end.

pub struct FakeStt {
    registry: Arc<Registry>,
    /// Connect delay, so a reopen or close can land while an open is in flight.
    connect: std::time::Duration,
    /// The session's streams, or AMD's own.
    detail: Detail,
    /// `STTCapabilities.keyterms`.
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
    hazard: super::hazard::StreamHazard,
    input_ended: bool,
    /// Between a start of speech and its end.
    speaking: bool,
}

impl Stt for FakeStt {
    fn label(&self) -> String {
        "fake::stt".into()
    }

    fn failed(&self, _error: ProviderError, recoverable: bool) {
        self.registry.record_failed(Kind::Stt, recoverable);
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities {
            streaming: true,
            interim_results: true,
            keyterms: self.keyterms,
            // the corpus models send their own `recognition_usage` (`Input::SttUsage`)
            reports_usage: true,
            ..Default::default()
        }
    }

    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!("streaming fixture") })
    }

    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            if !self.connect.is_zero() {
                tokio::time::sleep(self.connect).await;
            }
            if let Some(fault) = self.registry.hazard(Site::SttOpen) {
                return strike(fault, Site::SttOpen).await;
            }
            let (sender, events) = mpsc::unbounded_channel();
            let ended = Arc::new(Notify::new());
            let opened = self.registry.now();
            let slot = self.registry.register(
                Kind::Stt,
                self.detail.clone(),
                Control::Stt(sender, opened, None),
            );
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(Forward {
                    slot,
                    registry: self.registry.clone(),
                    ended: Some(ended.clone()),
                }) as SttSink,
                Box::new(Recognition {
                    slot,
                    registry: self.registry.clone(),
                    events,
                    ended,
                    hazard: self.registry.stream_hazard(Site::SttNext, Site::SttClose),
                    input_ended: false,
                    speaking: false,
                }) as SttSource,
            )))
        })
    }
}

impl StreamOutput<SpeechEvent> for Recognition {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::SttNext, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return strike(fault, Site::SttNext).await;
            }
            // A real stream ends once its input has and its utterance ended; otherwise a replaced
            // stream stays open.
            loop {
                if self.input_ended && !self.speaking {
                    return Ok(None);
                }
                let event = tokio::select! {
                    biased;
                    event = self.events.recv() => event,
                    () = self.ended.notified(), if !self.input_ended => {
                        self.input_ended = true;
                        continue;
                    }
                };
                if let Some(Ok(ev)) = &event {
                    match ev.kind {
                        lk_agents_abi::SpeechEventType::StartOfSpeech => self.speaking = true,
                        lk_agents_abi::SpeechEventType::EndOfSpeech => self.speaking = false,
                        _ => {}
                    }
                }
                return event.transpose();
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::SttClose, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return self.registry.strike_close(fault, Site::SttClose, deadline).await;
            }
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for Recognition {}

/// A real recogniser stamps words with the seconds of stream heard when they began and ended.
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
        ev.recognition_usage = Some(Box::new(lk_agents_abi::RecognitionUsage {
            audio_duration: Some(clock.seconds(clock.now)),
            ..Default::default()
        }));
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

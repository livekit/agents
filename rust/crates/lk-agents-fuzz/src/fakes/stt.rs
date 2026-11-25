use std::sync::Arc;

use lk_agents_abi::{
    AudioFrame, LanguageCode, SpeechEvent, SpeechEventType, Stt, SttCapabilities, SttSink,
    SttSource, Ts,
    errors::ProviderError,
    utils::stream::{Closed, StreamOutput},
};
use lk_agents_runtime::inference::stt::normalizer::SpeechNormalizer;
use tokio::sync::{Notify, mpsc};

use super::{
    Control, Forward, Registry, Slot,
    hazard::{Site, strike},
};
use crate::{
    input::{Input, Speaker, UserText},
    record::{Detail, Kind, Recognised, SttReport},
};

// A reopened stream stays addressable, so a late delivery to it is a recorded, rejected attempt
// rather than a silent channel swap. Audio is recorded, never interpreted. As a real provider's,
// a stream whose input ended still delivers the speech it is in, and ends once that speech has.

pub struct FakeStt {
    registry: Arc<Registry>,
    /// Connect delay, so a reopen or close can land while an open is in flight.
    connect: std::time::Duration,
    detail: Detail,
    /// `STTCapabilities.keyterms`.
    keyterms: bool,
    reports: SttSpeech,
}

impl FakeStt {
    pub fn new(
        registry: Arc<Registry>,
        connect: std::time::Duration,
        reports: SttSpeech,
    ) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect, detail: Detail::None, keyterms: false, reports })
    }

    /// A recogniser that takes keyterms, as most streaming providers do.
    pub fn taking_keyterms(
        registry: Arc<Registry>,
        connect: std::time::Duration,
        reports: SttSpeech,
    ) -> Arc<Self> {
        Arc::new(FakeStt { registry, connect, detail: Detail::None, keyterms: true, reports })
    }

    /// AMD's own recogniser: its streams open as `Detail::AmdStt`, apart from the session's.
    pub fn for_amd(registry: Arc<Registry>, reports: SttSpeech) -> Arc<Self> {
        let connect = std::time::Duration::ZERO;
        Arc::new(FakeStt { registry, connect, detail: Detail::AmdStt, keyterms: false, reports })
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
    /// The stream ended, and said so.
    finished: bool,
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
            transcribes_after_speech_ends: self.reports.transcribes_after_speech_ends,
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
            let provider = Provider {
                sender,
                opened: self.registry.now(),
                started: None,
                reports: self.reports,
                speech: SpeechNormalizer::new(
                    self.reports.provider_vad,
                    self.reports.transcribes_after_speech_ends,
                ),
            };
            let slot = self.registry.register(
                Kind::Stt,
                self.detail.clone(),
                Control::Stt(Box::new(provider)),
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
                    finished: false,
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
            // A real stream ends once its input has and its speech ended, all it said read;
            // otherwise a replaced stream stays open. The provider is gone from then on.
            loop {
                if self.input_ended && !self.speaking && self.events.is_empty() {
                    if !std::mem::replace(&mut self.finished, true) {
                        self.events.close();
                        let finished = SttReport::Finished;
                        self.registry.record_recognition(self.slot, finished, Vec::new());
                    }
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
                        SpeechEventType::StartOfSpeech => self.speaking = true,
                        SpeechEventType::EndOfSpeech => self.speaking = false,
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

/// How the provider reports speech, as a plugin declares it: whether it detects speech itself
/// (`provider_vad`, its own start and end of speech) and whether it transcribes a speech only
/// after it ended (`STTCapabilities.transcribes_after_speech_ends`, as openai and mistral do).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SttSpeech {
    pub provider_vad: bool,
    pub transcribes_after_speech_ends: bool,
}

impl Default for SttSpeech {
    fn default() -> Self {
        SttSpeech { provider_vad: true, transcribes_after_speech_ends: false }
    }
}

/// The provider's side of one recognition stream. What the case reports goes through the
/// production speech rules (`SpeechNormalizer`) on its way to the reader, exactly as a plugin's reports
/// do: the stream's starts and ends of speech, its utterance numbers and the words a connection
/// boundary promotes are the machine's, never the fake's.
pub struct Provider {
    sender: mpsc::UnboundedSender<Result<SpeechEvent, ProviderError>>,
    /// When the stream opened, and when its utterance began: the instants a transcript's
    /// `start_time`/`end_time` are measured from.
    opened: Ts,
    started: Option<Ts>,
    reports: SttSpeech,
    speech: SpeechNormalizer,
}

impl Provider {
    /// Whether the stream's speech is over and owes no words: nothing the case still has to
    /// release.
    pub(crate) fn settled(&self) -> bool {
        self.speech.settled()
    }

    /// One report from the provider. Whether the reader still takes the stream, and what the
    /// provider reported with what the stream made of it, if it reported anything.
    pub(super) fn deliver(
        &mut self,
        what: &Input,
        now: Ts,
    ) -> (bool, Option<(SttReport, Vec<Recognised>)>) {
        let seconds = |at: Ts| at.saturating_duration_since(self.opened).as_secs_f64();
        let words = |text: &UserText, speaker: &Option<Speaker>| {
            let mut transcript = lk_agents_abi::stt::events::Transcript::new("en", text.0.clone());
            transcript.speaker_id = speaker.map(|Speaker(n)| format!("speaker-{n}"));
            transcript.start_time = seconds(self.started.unwrap_or(now));
            transcript.end_time = seconds(now);
            transcript
        };
        if self.sender.is_closed() {
            return (false, None);
        }
        let mut out = Vec::new();
        let report = match what {
            Input::SttUsage { .. } => {
                // A usage report passes the speech rules by, as a plugin `send`s it.
                let mut ev = SpeechEvent::new(SpeechEventType::RecognitionUsage);
                // The audio the stream has heard so far, as the providers bill it.
                ev.recognition_usage = Some(Box::new(lk_agents_abi::RecognitionUsage {
                    audio_duration: Some(seconds(now)),
                    ..Default::default()
                }));
                return (self.sender.send(Ok(ev)).is_ok(), None);
            }
            Input::SttStart { .. } => {
                self.started = Some(now);
                if !self.reports.provider_vad {
                    return (true, None);
                }
                // as most plugins report it: no onset of its own, the start's arrival is
                self.speech.started(None, &mut out);
                SttReport::Started
            }
            Input::SttEnd { .. } => {
                if !self.reports.provider_vad {
                    return (true, None);
                }
                self.speech.ended(&mut out);
                SttReport::Ended
            }
            Input::SttInterim { text, speaker, .. } => {
                self.speech.transcript(words(text, speaker), &mut out);
                SttReport::Interim(text.0.clone())
            }
            Input::SttPreflight { text, speaker, .. } => {
                let transcript = lk_agents_abi::stt::events::Transcript {
                    preflight: true,
                    ..words(text, speaker)
                };
                self.speech.transcript(transcript, &mut out);
                SttReport::Preflight(text.0.clone())
            }
            Input::SttFinal { text, speaker, .. } => {
                let transcript = lk_agents_abi::stt::events::Transcript {
                    r#final: true,
                    ..words(text, speaker)
                };
                self.speech.transcript(transcript, &mut out);
                SttReport::Final(text.0.clone())
            }
            // The attempt ends: its open speech ends there, whether it reconnects or gives up.
            Input::SttDrop { .. } => {
                self.speech.boundary(&mut out);
                SttReport::Dropped
            }
            Input::SttFail { .. } => {
                self.speech.boundary(&mut out);
                SttReport::Failed
            }
            _ => return (false, None),
        };
        let emitted = out.iter().filter_map(recognised).collect();
        let mut accepted = true;
        for ev in out {
            accepted &= self.sender.send(Ok(ev)).is_ok();
        }
        if let Input::SttFail { .. } = what {
            let error = ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "recogniser failed mid-stream",
            ));
            accepted &= self.sender.send(Err(error)).is_ok();
        }
        (accepted, Some((report, emitted)))
    }
}

fn recognised(ev: &SpeechEvent) -> Option<Recognised> {
    let text = || ev.alternatives.first().map(|a| a.text.clone()).unwrap_or_default();
    Some(match ev.kind {
        SpeechEventType::StartOfSpeech => Recognised::Start,
        SpeechEventType::InterimTranscript => Recognised::Interim(text()),
        SpeechEventType::PreflightTranscript => Recognised::Preflight(text()),
        SpeechEventType::FinalTranscript => Recognised::Final(text()),
        SpeechEventType::EndOfSpeech => Recognised::End,
        _ => return None,
    })
}

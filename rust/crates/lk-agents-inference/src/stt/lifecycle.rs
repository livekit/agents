//! STT completion policy. Audio EOF asks for finalization; neither socket EOF nor a quiet
//! period proves that the provider delivered the final transcript.

use super::{machine::TranscriptMachine, wire::Incoming};
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{SpeechEvent, Ts};
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum SttEvent {
    Start,
    AudioWritten(f64),
    InputEof,
    Message(Incoming),
    ServerEof,
    Wake,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SttEffect {
    SttStreamOpen { stream_id: u64, stt: ResourceId },
    SttFinalize { stream_id: u64 },
    Deliver(SpeechEvent),
    EndOfSpeech,
    Finished(Result<(), ApiError>),
}

#[derive(Debug)]
pub struct SttProtocol {
    stream_id: u64,
    stt: ResourceId,
    transcript: TranscriptMachine,
    finalization_timeout: Duration,
    total_deadline: Option<Ts>,
    finalization_deadline: Option<Ts>,
    started: bool,
    input_ended: bool,
    finalized: bool,
    pending_transcript: bool,
    finished: bool,
    produced_output: bool,
}

impl SttProtocol {
    pub fn new(
        stream_id: u64,
        stt: ResourceId,
        request_id: String,
        language: Option<&str>,
        finalization_timeout: Duration,
        total_deadline: Option<Ts>,
    ) -> Self {
        Self {
            stream_id,
            stt,
            transcript: TranscriptMachine::new(request_id, language),
            finalization_timeout,
            total_deadline,
            finalization_deadline: None,
            started: false,
            input_ended: false,
            finalized: false,
            pending_transcript: false,
            finished: false,
            produced_output: false,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.transcript.is_speaking()
    }

    pub fn replay_safe(&self) -> bool {
        !self.produced_output && !self.pending_transcript && !self.input_ended
    }

    pub fn wake_at(&self) -> Option<Ts> {
        self.total_deadline.into_iter().chain(self.finalization_deadline).min()
    }

    pub fn handle(&mut self, event: SttEvent, now: Ts) -> Vec<SttEffect> {
        if self.finished {
            return Vec::new();
        }
        if self.wake_at().is_some_and(|deadline| now >= deadline) {
            self.finished = true;
            let phase = if self.total_deadline.is_some_and(|deadline| now >= deadline) {
                TimeoutPhase::Total
            } else {
                TimeoutPhase::Read
            };
            return vec![SttEffect::Finished(
                Err(ApiError::timed_out(phase).with_retryable(false)),
            )];
        }
        match event {
            SttEvent::Start if !self.started => {
                self.started = true;
                vec![SttEffect::SttStreamOpen { stream_id: self.stream_id, stt: self.stt }]
            }
            SttEvent::AudioWritten(duration) => {
                self.pending_transcript = true;
                self.transcript.record_audio(duration);
                Vec::new()
            }
            SttEvent::InputEof if !self.input_ended => {
                self.input_ended = true;
                self.finalization_deadline = Some(now.saturating_add(self.finalization_timeout));
                vec![SttEffect::SttFinalize { stream_id: self.stream_id }]
            }
            SttEvent::Message(message) => {
                match &message {
                    Incoming::SessionFinalized if self.input_ended => self.finalized = true,
                    Incoming::FinalTranscript(_) => self.pending_transcript = false,
                    Incoming::SessionClosed
                        if !self.input_ended || !self.finalized || self.pending_transcript =>
                    {
                        return self.fail("STT session closed before transcript completion");
                    }
                    _ => {}
                }
                let processed = self.transcript.process(message);
                self.produced_output |= processed.events.iter().any(|event| {
                    matches!(
                        event.kind,
                        lk_agents_abi::SpeechEventType::InterimTranscript
                            | lk_agents_abi::SpeechEventType::PreflightTranscript
                            | lk_agents_abi::SpeechEventType::FinalTranscript
                    )
                });
                let mut effects: Vec<_> =
                    processed.events.into_iter().map(SttEffect::Deliver).collect();
                if processed.end_of_speech {
                    effects.push(SttEffect::EndOfSpeech);
                }
                if let Some(error) = processed.error {
                    self.finished = true;
                    effects.push(SttEffect::Finished(Err(error)));
                } else if self.input_ended && self.finalized && !self.pending_transcript {
                    self.finished = true;
                    effects.push(SttEffect::Finished(Ok(())));
                }
                effects
            }
            SttEvent::ServerEof => self.fail("STT server EOF before transcript completion"),
            _ => Vec::new(),
        }
    }

    fn fail(&mut self, message: &str) -> Vec<SttEffect> {
        self.finished = true;
        vec![SttEffect::Finished(Err(
            ApiError::connection(Some(message.into())).with_retryable(false)
        ))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::Clock;

    #[test]
    fn stt_completion_vectors_require_finalization_and_transcript_and_bound_settlement() {
        let now = lk_agents_abi::VirtualClock::new().now();
        let make =
            || SttProtocol::new(1, ResourceId(2), "r".into(), None, Duration::from_secs(5), None);
        let mut stream = make();
        assert!(matches!(stream.handle(SttEvent::Start, now)[0], SttEffect::SttStreamOpen { .. }));
        stream.handle(SttEvent::AudioWritten(0.1), now);
        assert!(matches!(stream.handle(SttEvent::InputEof, now)[0], SttEffect::SttFinalize { .. }));
        assert!(stream.handle(SttEvent::Message(Incoming::SessionFinalized), now).is_empty());
        assert!(matches!(stream.handle(SttEvent::ServerEof, now)[0], SttEffect::Finished(Err(_))));
        let mut stream = make();
        stream.handle(SttEvent::AudioWritten(0.1), now);
        stream.handle(SttEvent::InputEof, now);
        stream.handle(SttEvent::Message(Incoming::SessionFinalized), now);
        let final_transcript =
            super::super::wire::decode(r#"{"type":"final_transcript","transcript":"hello"}"#)
                .unwrap();
        let effects = stream.handle(SttEvent::Message(final_transcript), now);
        assert!(effects.iter().any(|effect| matches!(effect, SttEffect::Deliver(event) if event.kind == lk_agents_abi::SpeechEventType::FinalTranscript)));
        assert_eq!(effects.last(), Some(&SttEffect::Finished(Ok(()))));
        let mut stream = make();
        stream.handle(SttEvent::InputEof, now);
        assert!(stream.handle(SttEvent::Message(Incoming::Unknown), now).is_empty());
        assert!(matches!(
            stream.handle(SttEvent::Wake, now.saturating_add(Duration::from_secs(5)))[0],
            SttEffect::Finished(Err(_))
        ));
        let mut stream = make();
        stream.handle(SttEvent::InputEof, now);
        assert_eq!(
            stream.handle(SttEvent::Message(Incoming::SessionFinalized), now),
            vec![SttEffect::Finished(Ok(()))]
        );
    }
}

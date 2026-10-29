//! STT completion policy. Audio EOF asks for finalization. Some providers acknowledge it with
//! `session.finalized`; others (Deepgram, AssemblyAI) answer with at most a final transcript and
//! keep the socket open, so a quiet period after the input ended completes the stream: short after
//! a final transcript, `finalization_timeout` otherwise. Socket EOF before completion fails.

use super::{machine::TranscriptMachine, wire::Incoming};
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::{SpeechEvent, Ts};
use std::time::Duration;

/// Quiet period after a post-EOF final transcript that completes a stream with no acknowledgment.
pub const FINAL_TRANSCRIPT_QUIET: Duration = Duration::from_secs(3);

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
    finished: bool,
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
            finished: false,
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.transcript.is_speaking()
    }

    /// A reconnect sends only new audio, so it repeats no delivered transcript; once the input
    /// ended there is nothing left to send and the pending transcript would silently be lost.
    pub fn replay_safe(&self) -> bool {
        !self.input_ended
    }

    pub fn wake_at(&self) -> Option<Ts> {
        self.total_deadline.into_iter().chain(self.finalization_deadline).min()
    }

    pub fn handle(&mut self, event: SttEvent, now: Ts) -> Vec<SttEffect> {
        if self.finished {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            self.finished = true;
            return vec![SttEffect::Finished(Err(
                ApiError::timed_out(TimeoutPhase::Total).with_retryable(false)
            ))];
        }
        if self.finalization_deadline.is_some_and(|deadline| now >= deadline) {
            self.finished = true;
            return vec![SttEffect::Finished(Ok(()))];
        }
        match event {
            SttEvent::Start if !self.started => {
                self.started = true;
                vec![SttEffect::SttStreamOpen { stream_id: self.stream_id, stt: self.stt }]
            }
            SttEvent::AudioWritten(duration) => {
                self.transcript.record_audio(duration);
                Vec::new()
            }
            SttEvent::InputEof if !self.input_ended => {
                self.input_ended = true;
                self.finalization_deadline = Some(now.saturating_add(self.finalization_timeout));
                vec![SttEffect::SttFinalize { stream_id: self.stream_id }]
            }
            SttEvent::Message(message) => {
                let quiet = match &message {
                    Incoming::SessionClosed => {
                        return self.fail("STT session closed before transcript completion");
                    }
                    Incoming::FinalTranscript(_) => FINAL_TRANSCRIPT_QUIET,
                    Incoming::InterimTranscript(_) | Incoming::PreflightTranscript(_) => {
                        self.finalization_timeout
                    }
                    _ => Duration::ZERO,
                };
                if self.input_ended && !quiet.is_zero() {
                    self.finalization_deadline = Some(now.saturating_add(quiet));
                }
                let acknowledged =
                    self.input_ended && matches!(message, Incoming::SessionFinalized);
                let processed = self.transcript.process(message);
                let mut effects: Vec<_> =
                    processed.events.into_iter().map(SttEffect::Deliver).collect();
                if processed.end_of_speech {
                    effects.push(SttEffect::EndOfSpeech);
                }
                if let Some(error) = processed.error {
                    self.finished = true;
                    effects.push(SttEffect::Finished(Err(error)));
                } else if acknowledged {
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
        vec![SttEffect::Finished(Err(ApiError::connection(Some(message.into()))))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::Clock;

    // Catches: a stream whose input ended waiting for an acknowledgment the gateway never sends
    // (Deepgram and AssemblyAI keep the socket open with no `session.finalized` after
    // `session.finalize`), failing every `stt.stream()` read to its end with a read timeout.
    #[test]
    fn input_eof_completes_on_acknowledgment_or_quiet_period() {
        let now = lk_agents_abi::VirtualClock::new().now();
        let at = |secs| now.saturating_add(Duration::from_secs(secs));
        let make =
            || SttProtocol::new(1, ResourceId(2), "r".into(), None, Duration::from_secs(5), None);
        let final_transcript = || {
            let text = r#"{"type":"final_transcript","transcript":"hello"}"#;
            SttEvent::Message(super::super::wire::decode(text).unwrap())
        };

        let mut stream = make();
        assert!(matches!(stream.handle(SttEvent::Start, now)[0], SttEffect::SttStreamOpen { .. }));
        stream.handle(SttEvent::AudioWritten(0.1), now);
        assert!(matches!(stream.handle(SttEvent::InputEof, now)[0], SttEffect::SttFinalize { .. }));
        let effects = stream.handle(final_transcript(), at(1));
        assert!(!effects.iter().any(|effect| matches!(effect, SttEffect::Finished(_))));
        assert!(stream.handle(SttEvent::Wake, at(3)).is_empty());
        assert_eq!(stream.handle(SttEvent::Wake, at(4)), vec![SttEffect::Finished(Ok(()))]);

        let mut stream = make();
        stream.handle(SttEvent::InputEof, now);
        assert!(stream.handle(SttEvent::Wake, at(4)).is_empty());
        assert_eq!(stream.handle(SttEvent::Wake, at(5)), vec![SttEffect::Finished(Ok(()))]);

        let mut stream = make();
        stream.handle(SttEvent::InputEof, now);
        assert_eq!(
            stream.handle(SttEvent::Message(Incoming::SessionFinalized), now),
            vec![SttEffect::Finished(Ok(()))]
        );

        let mut stream = make();
        stream.handle(SttEvent::InputEof, now);
        assert!(matches!(stream.handle(SttEvent::ServerEof, now)[0], SttEffect::Finished(Err(_))));
    }
}

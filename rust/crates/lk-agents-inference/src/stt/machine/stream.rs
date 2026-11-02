//! STT completion policy. Audio EOF asks for finalization. Some providers acknowledge it with
//! `session.finalized`; others (Deepgram, AssemblyAI) answer with at most a final transcript and
//! keep the socket open, so a quiet period after the input ended completes the stream: short after
//! a final transcript, `finalization_timeout` otherwise. Socket EOF before completion fails.

use super::transcript::TranscriptMachine;
use crate::effects::SttStreamEffect;
use crate::stt::wire::Incoming;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;
use std::time::Duration;

/// Quiet period after a post-EOF final transcript that completes a stream with no acknowledgment.
pub const FINAL_TRANSCRIPT_QUIET: Duration = Duration::from_secs(3);

#[derive(Debug, Clone)]
pub enum SttStreamEvent {
    Start,
    AudioWritten(f64),
    InputEnded,
    Message(Box<Incoming>),
    ServerEof,
    Wake,
}

#[derive(Debug)]
pub struct SttStream {
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

impl SttStream {
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

    pub fn handle(&mut self, event: SttStreamEvent, now: Ts) -> Vec<SttStreamEffect> {
        if self.finished {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            self.finished = true;
            return vec![SttStreamEffect::Finished(Err(
                ApiError::timed_out(TimeoutPhase::Total).with_retryable(false)
            ))];
        }
        if self.finalization_deadline.is_some_and(|deadline| now >= deadline) {
            self.finished = true;
            return vec![SttStreamEffect::Finished(Ok(()))];
        }
        match event {
            SttStreamEvent::Start if !self.started => {
                self.started = true;
                vec![SttStreamEffect::OpenSttStream { stream_id: self.stream_id, stt: self.stt }]
            }
            SttStreamEvent::AudioWritten(duration) => {
                self.transcript.record_audio(duration);
                Vec::new()
            }
            SttStreamEvent::InputEnded if !self.input_ended => {
                self.input_ended = true;
                self.finalization_deadline = Some(now.saturating_add(self.finalization_timeout));
                vec![SttStreamEffect::SttFinalize { stream_id: self.stream_id }]
            }
            SttStreamEvent::Message(message) => {
                let message = *message;
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
                    processed.events.into_iter().map(SttStreamEffect::Deliver).collect();
                if processed.end_of_speech {
                    effects.push(SttStreamEffect::EndOfSpeech);
                }
                if let Some(error) = processed.error {
                    self.finished = true;
                    effects.push(SttStreamEffect::Finished(Err(error)));
                } else if acknowledged {
                    self.finished = true;
                    effects.push(SttStreamEffect::Finished(Ok(())));
                }
                effects
            }
            SttStreamEvent::ServerEof => self.fail("STT server EOF before transcript completion"),
            _ => Vec::new(),
        }
    }

    fn fail(&mut self, message: &str) -> Vec<SttStreamEffect> {
        self.finished = true;
        vec![SttStreamEffect::Finished(Err(ApiError::connection(Some(message.into()))))]
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
            || SttStream::new(1, ResourceId(2), "r".into(), None, Duration::from_secs(5), None);
        let final_transcript = || {
            let text = r#"{"type":"final_transcript","transcript":"hello"}"#;
            SttStreamEvent::Message(Box::new(crate::stt::wire::decode(text).unwrap()))
        };

        let mut stream = make();
        assert!(matches!(
            stream.handle(SttStreamEvent::Start, now)[0],
            SttStreamEffect::OpenSttStream { .. }
        ));
        stream.handle(SttStreamEvent::AudioWritten(0.1), now);
        assert!(matches!(
            stream.handle(SttStreamEvent::InputEnded, now)[0],
            SttStreamEffect::SttFinalize { .. }
        ));
        let effects = stream.handle(final_transcript(), at(1));
        assert!(!effects.iter().any(|effect| matches!(effect, SttStreamEffect::Finished(_))));
        assert!(stream.handle(SttStreamEvent::Wake, at(3)).is_empty());
        assert_eq!(
            stream.handle(SttStreamEvent::Wake, at(4)),
            vec![SttStreamEffect::Finished(Ok(()))]
        );

        let mut stream = make();
        stream.handle(SttStreamEvent::InputEnded, now);
        assert!(stream.handle(SttStreamEvent::Wake, at(4)).is_empty());
        assert_eq!(
            stream.handle(SttStreamEvent::Wake, at(5)),
            vec![SttStreamEffect::Finished(Ok(()))]
        );

        let mut stream = make();
        stream.handle(SttStreamEvent::InputEnded, now);
        assert_eq!(
            stream.handle(SttStreamEvent::Message(Box::new(Incoming::SessionFinalized)), now),
            vec![SttStreamEffect::Finished(Ok(()))]
        );

        let mut stream = make();
        stream.handle(SttStreamEvent::InputEnded, now);
        assert!(matches!(
            stream.handle(SttStreamEvent::ServerEof, now)[0],
            SttStreamEffect::Finished(Err(_))
        ));
    }
}

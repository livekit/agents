//! TtsSynthesis protocol decisions. The adapter owns sockets, bounded delivery and cleanup.

use crate::effects::TtsSynthesisEffect;
use std::{collections::BTreeSet, time::Duration};

use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::metrics::base::ConnectionAcquired;
use lk_agents_abi::{SynthesizedAudio, Ts};

use super::wire::Incoming;

pub enum TtsSynthesisEvent {
    Start,
    Text(String),
    InputEnded,
    InputSent { final_flush: bool, started_time: f64 },
    Received(Incoming),
    Audio { session_id: Option<String>, data: Vec<u8> },
    Failed(ApiError),
    ServerEof,
    Cancel,
    Wake,
}

pub struct TtsSynthesis {
    request_id: String,
    segment_id: String,
    sample_rate: u32,
    read_timeout: Option<Duration>,
    response_deadline: Option<Ts>,
    total_deadline: Option<Ts>,
    started_time: Option<f64>,
    input_ended: bool,
    final_flush_sent: bool,
    terminal: bool,
    segments: BTreeSet<String>,
    /// The synthesis's connection, until its first frame carries it.
    connection: Option<ConnectionAcquired>,
}

impl TtsSynthesis {
    pub fn new(
        request_id: String,
        sample_rate: u32,
        read_timeout: Option<Duration>,
        total_deadline: Option<Ts>,
    ) -> Self {
        Self {
            segment_id: request_id.clone(),
            request_id,
            sample_rate,
            read_timeout,
            response_deadline: None,
            total_deadline,
            started_time: None,
            input_ended: false,
            final_flush_sent: false,
            terminal: false,
            segments: Default::default(),
            connection: None,
        }
    }

    /// The connection the synthesis runs on: its first frame reports it.
    pub fn connected(&mut self, connection: ConnectionAcquired) {
        self.connection = Some(connection);
    }

    pub fn total_deadline(&self) -> Option<Ts> {
        self.total_deadline
    }

    pub fn deadline(&self) -> Option<Ts> {
        self.total_deadline.into_iter().chain(self.response_deadline).min()
    }

    pub fn delivery_resumed(&mut self, blocked_at: Ts, now: Ts) {
        if let Some(deadline) = &mut self.response_deadline {
            *deadline = deadline.saturating_add(now.saturating_duration_since(blocked_at));
        }
    }

    pub fn handle(&mut self, event: TtsSynthesisEvent, now: Ts) -> Vec<TtsSynthesisEffect> {
        if self.terminal {
            return Vec::new();
        }
        if self.deadline().is_some_and(|deadline| now >= deadline)
            && !matches!(event, TtsSynthesisEvent::Cancel)
        {
            let phase = if self.total_deadline.is_some_and(|deadline| now >= deadline) {
                TimeoutPhase::Total
            } else {
                TimeoutPhase::Read
            };
            return self.fail(ApiError::timed_out(phase));
        }
        match event {
            TtsSynthesisEvent::Start => vec![TtsSynthesisEffect::TtsSynthesisStart],
            TtsSynthesisEvent::Text(text) if !self.input_ended => {
                vec![TtsSynthesisEffect::TtsTextSend(text)]
            }
            TtsSynthesisEvent::Text(_) => {
                self.fail(ApiError::new("TTS text received after input EOF"))
            }
            TtsSynthesisEvent::InputEnded => {
                if self.input_ended {
                    return Vec::new();
                }
                self.input_ended = true;
                vec![TtsSynthesisEffect::TtsFinalFlush]
            }
            TtsSynthesisEvent::InputSent { final_flush, started_time } => {
                self.started_time.get_or_insert(started_time);
                self.final_flush_sent |= final_flush;
                if final_flush || self.response_deadline.is_none() {
                    self.response_deadline =
                        self.read_timeout.map(|timeout| now.saturating_add(timeout));
                }
                Vec::new()
            }
            TtsSynthesisEvent::Audio { session_id, data } => {
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                if !data.len().is_multiple_of(2) {
                    return self.fail(ApiError::new("TTS returned truncated PCM audio"));
                }
                self.response_deadline = if self.final_flush_sent {
                    self.read_timeout.map(|timeout| now.saturating_add(timeout))
                } else {
                    None
                };
                vec![TtsSynthesisEffect::TtsAudio(self.audio(data, false))]
            }
            TtsSynthesisEvent::Received(Incoming::OutputAlignment { session_id, tokens }) => {
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                self.progress(now);
                self.segments.insert(self.segment_id.clone());
                vec![TtsSynthesisEffect::TtsAlignment {
                    segment_id: self.segment_id.clone(),
                    tokens,
                }]
            }
            TtsSynthesisEvent::Received(Incoming::Done { session_id }) => {
                if !self.final_flush_sent {
                    return self.fail(ApiError::new("TTS completed before final flush"));
                }
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                let mut effects = Vec::new();
                let segment_id = self.segment_id.clone();
                let pending_ids: Vec<_> =
                    self.segments.iter().filter(|id| **id != segment_id).cloned().collect();
                for id in pending_ids {
                    self.segment_id = id;
                    effects.push(TtsSynthesisEffect::TtsAudio(self.audio(Vec::new(), true)));
                }
                self.segment_id = segment_id;
                effects.push(TtsSynthesisEffect::TtsAudio(self.audio(Vec::new(), true)));
                self.terminal = true;
                effects.push(TtsSynthesisEffect::Finished(Ok(())));
                effects
            }
            TtsSynthesisEvent::Received(Incoming::SessionCreated { session_id }) => {
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                Vec::new()
            }
            TtsSynthesisEvent::Received(Incoming::Error { raw }) => {
                self.fail(ApiError::new(format!("LiveKit Inference TTS returned error: {raw}")))
            }
            TtsSynthesisEvent::Received(Incoming::Malformed { detail }) => {
                self.fail(ApiError::new(detail))
            }
            TtsSynthesisEvent::Failed(error) => self.fail(error),
            TtsSynthesisEvent::ServerEof => {
                self.fail(ApiError::new("TTS gateway closed before synthesis completion"))
            }
            TtsSynthesisEvent::Cancel => {
                self.terminal = true;
                vec![TtsSynthesisEffect::TtsCancel]
            }
            // Session-created and heartbeat frames do not satisfy outstanding synthesis work.
            TtsSynthesisEvent::Received(_) | TtsSynthesisEvent::Wake => Vec::new(),
        }
    }

    fn progress(&mut self, now: Ts) {
        if self.response_deadline.is_some() {
            self.response_deadline = self.read_timeout.map(|timeout| now.saturating_add(timeout));
        }
    }

    fn segment(&mut self, id: Option<String>) -> Result<(), ApiError> {
        if let Some(id) = id {
            if id.is_empty() {
                return Err(ApiError::new("TTS returned an empty segment identity"));
            }
            self.segment_id = id;
        }
        Ok(())
    }

    fn audio(&mut self, data: Vec<u8>, is_final: bool) -> SynthesizedAudio {
        self.segments.insert(self.segment_id.clone());
        SynthesizedAudio {
            is_final,
            segment_id: self.segment_id.clone(),
            started_time: self.started_time,
            connection: self.connection.take(),
            ..SynthesizedAudio::new(
                crate::vad::machine::mono(data, self.sample_rate),
                self.request_id.clone(),
            )
        }
    }

    fn fail(&mut self, mut error: ApiError) -> Vec<TtsSynthesisEffect> {
        error.request_id.get_or_insert_with(|| self.request_id.clone());
        self.terminal = true;
        vec![TtsSynthesisEffect::Finished(Err(error)), TtsSynthesisEffect::TtsCancel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthesis() -> TtsSynthesis {
        TtsSynthesis::new("request".into(), 24_000, Some(Duration::from_secs(1)), None)
    }

    #[test]
    fn empty_input_requires_the_final_flush_and_the_provider_completion() {
        let mut core = synthesis();
        assert!(matches!(
            core.handle(TtsSynthesisEvent::InputEnded, Ts::ZERO).as_slice(),
            [TtsSynthesisEffect::TtsFinalFlush]
        ));
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: true, started_time: 0.0 },
            Ts::ZERO,
        );
        let effects =
            core.handle(TtsSynthesisEvent::Received(Incoming::Done { session_id: None }), Ts::ZERO);
        assert!(
            matches!(effects.as_slice(), [TtsSynthesisEffect::TtsAudio(audio), TtsSynthesisEffect::Finished(Ok(()))] if audio.is_final)
        );
        assert!(core.handle(TtsSynthesisEvent::ServerEof, Ts::ZERO).is_empty());
    }

    #[test]
    fn heartbeats_cannot_settle_response_and_later_segments_rearm_the_deadline() {
        let mut core = synthesis();
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: false, started_time: 0.0 },
            Ts::ZERO,
        );
        core.handle(
            TtsSynthesisEvent::Received(Incoming::Unknown { session_id: None }),
            Ts::from_nanos(900_000_000),
        );
        assert!(matches!(
            core.handle(TtsSynthesisEvent::Wake, Ts::from_nanos(1_000_000_000)).as_slice(),
            [TtsSynthesisEffect::Finished(Err(_)), TtsSynthesisEffect::TtsCancel]
        ));
        let mut core = synthesis();
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: false, started_time: 0.0 },
            Ts::ZERO,
        );
        core.handle(
            TtsSynthesisEvent::Audio { session_id: Some("first".into()), data: vec![0; 2] },
            Ts::from_nanos(500_000_000),
        );
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: false, started_time: 0.0 },
            Ts::from_nanos(1_000_000_000),
        );
        assert!(core.handle(TtsSynthesisEvent::Wake, Ts::from_nanos(1_500_000_000)).is_empty());
        assert!(matches!(
            core.handle(TtsSynthesisEvent::Wake, Ts::from_nanos(2_000_000_000)).as_slice(),
            [TtsSynthesisEffect::Finished(Err(_)), TtsSynthesisEffect::TtsCancel]
        ));
    }

    #[test]
    fn delivered_audio_keeps_its_identity_and_provider_failure_is_terminal() {
        let mut core = synthesis();
        for segment in ["first", "second"] {
            let effects = core.handle(
                TtsSynthesisEvent::Audio { session_id: Some(segment.into()), data: vec![0; 2] },
                Ts::ZERO,
            );
            assert!(
                matches!(effects.as_slice(), [TtsSynthesisEffect::TtsAudio(audio)] if audio.segment_id == segment && audio.request_id == "request")
            );
        }
        assert!(matches!(
            core.handle(
                TtsSynthesisEvent::Received(Incoming::Error { raw: "provider failed".into() }),
                Ts::ZERO
            )
            .as_slice(),
            [TtsSynthesisEffect::Finished(Err(_)), TtsSynthesisEffect::TtsCancel]
        ));
        assert!(core.handle(TtsSynthesisEvent::Start, Ts::ZERO).is_empty());
        assert!(
            core.handle(TtsSynthesisEvent::Received(Incoming::Done { session_id: None }), Ts::ZERO)
                .is_empty()
        );
    }
    #[test]
    fn segment_progress_cannot_extend_the_total_deadline() {
        let mut core = TtsSynthesis::new(
            "request".into(),
            24_000,
            Some(Duration::from_secs(1)),
            Some(Ts::from_nanos(1_500_000_000)),
        );
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: false, started_time: 0.0 },
            Ts::ZERO,
        );
        core.handle(
            TtsSynthesisEvent::Audio { session_id: None, data: vec![0; 2] },
            Ts::from_nanos(500_000_000),
        );
        core.handle(
            TtsSynthesisEvent::InputSent { final_flush: false, started_time: 1.0 },
            Ts::from_nanos(1_000_000_000),
        );
        assert!(
            matches!(core.handle(TtsSynthesisEvent::Wake, Ts::from_nanos(1_500_000_000)).as_slice(),
            [TtsSynthesisEffect::Finished(Err(error)), TtsSynthesisEffect::TtsCancel] if error.phase == Some(TimeoutPhase::Total) && error.request_id.as_deref() == Some("request"))
        );
    }
}

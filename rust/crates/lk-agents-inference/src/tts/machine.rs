//! Synthesis protocol decisions. The adapter owns sockets, bounded delivery and cleanup.

use std::time::Duration;

use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::tts::markup::drop_bracket_cues;
use lk_agents_abi::{SynthesizedAudio, TimedString, Ts};

use super::wire::Incoming;

pub enum Event {
    Start,
    Text(String),
    EndInput,
    InputSent { final_flush: bool, started_time: f64 },
    Received(Incoming),
    Audio { session_id: Option<String>, data: Vec<u8> },
    Failed(ApiError),
    ServerEof,
    Cancel,
    Tick,
}

pub enum Effect {
    TtsSynthesisStart,
    TtsTextSend(String),
    TtsFinalFlush,
    TtsAudio(SynthesizedAudio),
    TtsFailed(ApiError),
    TtsCompleted,
    TtsCancel,
}

pub struct Synthesis {
    request_id: String,
    segment_id: String,
    sample_rate: u32,
    expressive: bool,
    read_timeout: Option<Duration>,
    response_deadline: Option<Ts>,
    total_deadline: Option<Ts>,
    started_time: Option<f64>,
    input_ended: bool,
    final_flush_sent: bool,
    terminal: bool,
    segments: std::collections::BTreeMap<String, (Vec<TimedString>, Vec<TimedString>)>,
}

impl Synthesis {
    pub fn new(
        request_id: String,
        sample_rate: u32,
        expressive: bool,
        read_timeout: Option<Duration>,
        total_deadline: Option<Ts>,
    ) -> Self {
        Self {
            segment_id: request_id.clone(),
            request_id,
            sample_rate,
            expressive,
            read_timeout,
            response_deadline: None,
            total_deadline,
            started_time: None,
            input_ended: false,
            final_flush_sent: false,
            terminal: false,
            segments: Default::default(),
        }
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

    pub fn handle(&mut self, event: Event, now: Ts) -> Vec<Effect> {
        if self.terminal {
            return Vec::new();
        }
        if self.deadline().is_some_and(|deadline| now >= deadline)
            && !matches!(event, Event::Cancel)
        {
            let phase = if self.total_deadline.is_some_and(|deadline| now >= deadline) {
                TimeoutPhase::Total
            } else {
                TimeoutPhase::Read
            };
            return self.fail(ApiError::timed_out(phase));
        }
        match event {
            Event::Start => vec![Effect::TtsSynthesisStart],
            Event::Text(text) if !self.input_ended => vec![Effect::TtsTextSend(text)],
            Event::Text(_) => self.fail(ApiError::new("TTS text received after input EOF")),
            Event::EndInput => {
                if self.input_ended {
                    return Vec::new();
                }
                self.input_ended = true;
                vec![Effect::TtsFinalFlush]
            }
            Event::InputSent { final_flush, started_time } => {
                self.started_time.get_or_insert(started_time);
                self.final_flush_sent |= final_flush;
                if final_flush || self.response_deadline.is_none() {
                    self.response_deadline =
                        self.read_timeout.map(|timeout| now.saturating_add(timeout));
                }
                Vec::new()
            }
            Event::Audio { session_id, data } => {
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
                vec![Effect::TtsAudio(self.audio(data, false))]
            }
            Event::Received(Incoming::OutputAlignment { session_id, tokens }) => {
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                self.progress(now);
                let (pending, held) = self.segments.entry(self.segment_id.clone()).or_default();
                pending.extend(if self.expressive {
                    drop_bracket_cues(tokens, held, false)
                } else {
                    tokens
                });
                Vec::new()
            }
            Event::Received(Incoming::Done { session_id }) => {
                if !self.final_flush_sent {
                    return self.fail(ApiError::new("TTS completed before final flush"));
                }
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                let mut effects = Vec::new();
                let segment_id = self.segment_id.clone();
                let pending_ids: Vec<_> =
                    self.segments.keys().filter(|id| **id != segment_id).cloned().collect();
                for id in pending_ids {
                    self.segment_id = id;
                    effects.push(Effect::TtsAudio(self.audio(Vec::new(), true)));
                }
                self.segment_id = segment_id;
                effects.push(Effect::TtsAudio(self.audio(Vec::new(), true)));
                self.terminal = true;
                effects.push(Effect::TtsCompleted);
                effects
            }
            Event::Received(Incoming::SessionCreated { session_id }) => {
                if let Err(error) = self.segment(session_id) {
                    return self.fail(error);
                }
                Vec::new()
            }
            Event::Received(Incoming::Error { raw }) => {
                self.fail(ApiError::new(format!("LiveKit Inference TTS returned error: {raw}")))
            }
            Event::Received(Incoming::Malformed { detail }) => self.fail(ApiError::new(detail)),
            Event::Failed(error) => self.fail(error),
            Event::ServerEof => {
                self.fail(ApiError::new("TTS gateway closed before synthesis completion"))
            }
            Event::Cancel => {
                self.terminal = true;
                vec![Effect::TtsCancel]
            }
            // Session-created and heartbeat frames do not satisfy outstanding synthesis work.
            Event::Received(_) | Event::Tick => Vec::new(),
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
        let (pending, held) = self.segments.entry(self.segment_id.clone()).or_default();
        if is_final {
            pending.extend(drop_bracket_cues(Vec::new(), held, true));
        }
        SynthesizedAudio {
            is_final,
            segment_id: self.segment_id.clone(),
            started_time: self.started_time,
            timed_transcript: std::mem::take(pending),
            ..SynthesizedAudio::new(
                crate::vad::machine::mono(data, self.sample_rate),
                self.request_id.clone(),
            )
        }
    }

    fn fail(&mut self, mut error: ApiError) -> Vec<Effect> {
        error.request_id.get_or_insert_with(|| self.request_id.clone());
        self.terminal = true;
        vec![Effect::TtsFailed(error), Effect::TtsCancel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthesis() -> Synthesis {
        Synthesis::new("request".into(), 24_000, false, Some(Duration::from_secs(1)), None)
    }

    #[test]
    fn empty_input_requires_the_final_flush_and_the_provider_completion() {
        let mut core = synthesis();
        assert!(matches!(
            core.handle(Event::EndInput, Ts::ZERO).as_slice(),
            [Effect::TtsFinalFlush]
        ));
        core.handle(Event::InputSent { final_flush: true, started_time: 0.0 }, Ts::ZERO);
        let effects = core.handle(Event::Received(Incoming::Done { session_id: None }), Ts::ZERO);
        assert!(
            matches!(effects.as_slice(), [Effect::TtsAudio(audio), Effect::TtsCompleted] if audio.is_final)
        );
        assert!(core.handle(Event::ServerEof, Ts::ZERO).is_empty());
    }

    #[test]
    fn heartbeats_cannot_settle_response_and_later_segments_rearm_the_deadline() {
        let mut core = synthesis();
        core.handle(Event::InputSent { final_flush: false, started_time: 0.0 }, Ts::ZERO);
        core.handle(
            Event::Received(Incoming::Unknown { session_id: None }),
            Ts::from_nanos(900_000_000),
        );
        assert!(matches!(
            core.handle(Event::Tick, Ts::from_nanos(1_000_000_000)).as_slice(),
            [Effect::TtsFailed(_), Effect::TtsCancel]
        ));
        let mut core = synthesis();
        core.handle(Event::InputSent { final_flush: false, started_time: 0.0 }, Ts::ZERO);
        core.handle(
            Event::Audio { session_id: Some("first".into()), data: vec![0; 2] },
            Ts::from_nanos(500_000_000),
        );
        core.handle(
            Event::InputSent { final_flush: false, started_time: 0.0 },
            Ts::from_nanos(1_000_000_000),
        );
        assert!(core.handle(Event::Tick, Ts::from_nanos(1_500_000_000)).is_empty());
        assert!(matches!(
            core.handle(Event::Tick, Ts::from_nanos(2_000_000_000)).as_slice(),
            [Effect::TtsFailed(_), Effect::TtsCancel]
        ));
    }

    #[test]
    fn delivered_audio_keeps_its_identity_and_provider_failure_is_terminal() {
        let mut core = synthesis();
        for segment in ["first", "second"] {
            let effects = core.handle(
                Event::Audio { session_id: Some(segment.into()), data: vec![0; 2] },
                Ts::ZERO,
            );
            assert!(
                matches!(effects.as_slice(), [Effect::TtsAudio(audio)] if audio.segment_id == segment && audio.request_id == "request")
            );
        }
        assert!(matches!(
            core.handle(
                Event::Received(Incoming::Error { raw: "provider failed".into() }),
                Ts::ZERO
            )
            .as_slice(),
            [Effect::TtsFailed(_), Effect::TtsCancel]
        ));
        assert!(core.handle(Event::Start, Ts::ZERO).is_empty());
        assert!(
            core.handle(Event::Received(Incoming::Done { session_id: None }), Ts::ZERO).is_empty()
        );
    }
    #[test]
    fn segment_progress_cannot_extend_the_total_deadline() {
        let mut core = Synthesis::new(
            "request".into(),
            24_000,
            false,
            Some(Duration::from_secs(1)),
            Some(Ts::from_nanos(1_500_000_000)),
        );
        core.handle(Event::InputSent { final_flush: false, started_time: 0.0 }, Ts::ZERO);
        core.handle(
            Event::Audio { session_id: None, data: vec![0; 2] },
            Ts::from_nanos(500_000_000),
        );
        core.handle(
            Event::InputSent { final_flush: false, started_time: 1.0 },
            Ts::from_nanos(1_000_000_000),
        );
        assert!(matches!(core.handle(Event::Tick, Ts::from_nanos(1_500_000_000)).as_slice(),
            [Effect::TtsFailed(error), Effect::TtsCancel] if error.phase == Some(TimeoutPhase::Total) && error.request_id.as_deref() == Some("request")));
    }
}

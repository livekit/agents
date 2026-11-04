//! The user turn's spans (`telemetry.md`, "The user turn"). The activity owns one
//! [`UserTurnSpan`] per user turn, from its first evidence to its end: while it is heard, while
//! decided turns wait for their hook, and while the hook runs. Dropping one unended is a bug the
//! `Drop` assertion of its spans reports.

use super::participant_attrs;
use crate::audio_recognition::{EndOfTurnInfo, EouDetected, UserTurnCommitVerdict};
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::{BoolString, Content, Json, Measured, Seconds, SpanRef};
use lk_agents_abi::telemetry::errors::SpanError;
use lk_agents_abi::telemetry::spans::user_turn::{
    EouDetection, EouDetectionEnd, EouOutcome, EouPrediction, EouWait, EouWaitEnd,
    OnUserTurnCompleted, Rearmed, StopResponse, Superseded, UserResumed, UserSpeaking,
    UserSpeakingEnd, UserTurn, UserTurnEnd, UserTurnOrigin, UserTurnOutcome,
};
use lk_agents_abi::voice::events::OverlappingSpeechEvent;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_core::telemetry::span::{Span, Trace};

/// Every way a decided turn ends; a new verdict is a new arm here.
impl From<UserTurnCommitVerdict> for UserTurnOutcome {
    fn from(verdict: UserTurnCommitVerdict) -> Self {
        match verdict {
            UserTurnCommitVerdict::Reply | UserTurnCommitVerdict::LateTranscript => {
                UserTurnOutcome::Committed
            }
            UserTurnCommitVerdict::TooFewWords => UserTurnOutcome::TooFewWords,
            UserTurnCommitVerdict::Empty => UserTurnOutcome::Empty,
            UserTurnCommitVerdict::RealtimeBackchannel => UserTurnOutcome::Backchannel,
            UserTurnCommitVerdict::Dropped { .. } => UserTurnOutcome::Dropped,
        }
    }
}

/// The STT a turn is heard with: its plugin's model and provider, as 1.8.10 records them.
#[derive(Debug, Clone, Default)]
pub(crate) struct Stt {
    pub(crate) model: Option<String>,
    pub(crate) provider: Option<String>,
}

/// What triggered the end-of-turn countdown (`lk.eou.source`) and the delay it waits.
#[derive(Clone, Copy)]
pub(crate) struct Endpointing {
    pub(crate) source: &'static str,
    pub(crate) delay: f64,
}

impl Endpointing {
    fn decl(self, started_at: Ts) -> EouWait {
        EouWait {
            started_at,
            source: self.source.to_owned(),
            endpointing_delay: Seconds(self.delay),
        }
    }
}

/// `lk.eou.source` of a manual commit.
pub(crate) const MANUAL: &str = "manual";

struct Speaking {
    span: Span<UserSpeaking>,
    verdict: Option<Verdict>,
}

#[derive(Clone, Copy)]
struct Verdict {
    interruption: bool,
    probability: Option<f64>,
    total: f64,
    prediction: f64,
    delay: f64,
}

struct Wait {
    span: Span<EouWait>,
    started: Ts,
    endpointing: Endpointing,
    rearms: u32,
    not_committed: u32,
    /// The prediction the countdown waits on.
    detection: Option<Span<EouDetection>>,
}

/// What the decision knew of the turn, written when it ends.
#[derive(Default)]
struct Decided {
    transcript: String,
    confidence: Option<f64>,
    transcription_delay: Option<f64>,
    end_of_turn_delay: Option<f64>,
}

/// One user turn's `user_turn` span and the spans inside it: its speech segments
/// (`user_speaking`), its end-of-turn wait (`eou_wait`, `eou_detection`) and its hook
/// (`on_user_turn_completed`).
pub(crate) struct UserTurnSpan {
    /// `None` once ended.
    span: Option<Span<UserTurn>>,
    speaking: Option<Speaking>,
    wait: Option<Wait>,
    hook: Option<Span<OnUserTurnCompleted>>,
    resumes: u32,
    request_ids: Vec<String>,
    stt_events: Vec<serde_json::Value>,
    decided: Decided,
}

impl UserTurnSpan {
    /// The turn's first evidence, at `started_at`.
    pub(crate) fn open(
        trace: &mut Trace,
        started_at: Ts,
        origin: UserTurnOrigin,
        participant: Option<&ParticipantInfo>,
        stt: &Stt,
    ) -> Self {
        let decl = UserTurn {
            started_at,
            origin,
            participant: participant.map(participant_attrs),
            stt_model: stt.model.clone(),
            stt_provider: stt.provider.clone(),
        };
        UserTurnSpan {
            span: Some(trace.open(decl)),
            speaking: None,
            wait: None,
            hook: None,
            resumes: 0,
            request_ids: Vec::new(),
            stt_events: Vec::new(),
            decided: Decided::default(),
        }
    }

    /// A turn whose span another decision of the same turn holds: it records nothing.
    pub(crate) fn none() -> Self {
        UserTurnSpan {
            span: None,
            speaking: None,
            wait: None,
            hook: None,
            resumes: 0,
            request_ids: Vec::new(),
            stt_events: Vec::new(),
            decided: Decided::default(),
        }
    }

    /// Typed user input: a turn of its own, committed on arrival. Returns its span.
    pub(crate) fn typed(
        trace: &mut Trace,
        participant: Option<&ParticipantInfo>,
        text: &str,
    ) -> Option<SpanRef> {
        let now = trace.now();
        let mut turn = Self::open(trace, now, UserTurnOrigin::Text, participant, &Stt::default());
        turn.decided.transcript = text.to_owned();
        let span = turn.context(trace);
        turn.end(trace, UserTurnOutcome::Committed);
        span
    }

    /// What the turn's chat item and its hook run under.
    pub(crate) fn context(&self, trace: &Trace) -> Option<SpanRef> {
        trace.context(self.span.as_ref()?)
    }

    /// A speech onset at `at`: a new segment, resuming the turn the decision waits on.
    pub(crate) fn speech_started(
        &mut self,
        trace: &mut Trace,
        at: Ts,
        participant: Option<&ParticipantInfo>,
    ) {
        if let Some(wait) = self.wait.take() {
            trace.event(&wait.span, UserResumed { at });
            self.resumes += 1;
            end_wait(trace, at, wait, EouOutcome::UserResumed);
        }
        let Some(turn) = &self.span else { return };
        if self.speaking.is_none() {
            let decl =
                UserSpeaking { started_at: at, participant: participant.map(participant_attrs) };
            self.speaking = Some(Speaking { span: trace.open_in(turn, decl), verdict: None });
        }
    }

    /// The end of speech at `at` starts the countdown to the turn's decision.
    pub(crate) fn speech_ended(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        if let Some(speaking) = self.speaking.take() {
            end_speaking(trace, at, speaking);
        }
        self.trigger(trace, at, endpointing);
    }

    /// A later trigger (a final transcript, a manual commit) restarts the countdown of the turn
    /// waiting on its decision: the wait is re-armed, or opened by a manual commit.
    pub(crate) fn retriggered(&mut self, trace: &mut Trace, endpointing: Endpointing) {
        if self.wait.is_some() || endpointing.source == MANUAL {
            let now = trace.now();
            self.trigger(trace, now, endpointing);
        }
    }

    /// A countdown trigger: it opens the wait, or re-arms the open one.
    fn trigger(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        match &mut self.wait {
            Some(wait) => {
                wait.rearms += 1;
                wait.endpointing = endpointing;
                let source = endpointing.source.to_owned();
                trace.event(&wait.span, Rearmed { at, source });
                trace.set(&wait.span, endpointing.decl(wait.started));
            }
            None => {
                let Some(turn) = &self.span else { return };
                let span = trace.open_in(turn, endpointing.decl(at));
                self.wait = Some(Wait {
                    span,
                    started: at,
                    endpointing,
                    rearms: 0,
                    not_committed: 0,
                    detection: None,
                });
            }
        }
    }

    /// The countdown's end-of-turn detection: it waits on the detector, or resolved.
    pub(crate) fn detection(
        &mut self,
        trace: &mut Trace,
        detected: &EouDetected,
        chat_ctx: serde_json::Value,
    ) {
        let Some(wait) = self.wait.as_mut() else { return };
        let (prediction, delay, from_cache, language) = match detected {
            // One detection at a time: a new countdown's replaces the last.
            EouDetected::Awaiting => {
                if let Some(span) = wait.detection.take() {
                    let ended_at = trace.now();
                    trace.end(span, EouDetectionEnd { ended_at, prediction: None });
                }
                wait.detection = Some(trace.open_in(&wait.span, EouDetection));
                return;
            }
            EouDetected::Resolved { prediction, delay, from_cache, language } => {
                (prediction, delay, from_cache, language)
            }
        };
        let span = match wait.detection.take() {
            Some(span) => span,
            None => trace.open_in(&wait.span, EouDetection),
        };
        wait.endpointing.delay = delay.as_secs_f64();
        trace.set(&wait.span, wait.endpointing.decl(wait.started));
        let prediction = prediction.map(|p| EouPrediction {
            chat_ctx: Content(Json(chat_ctx)),
            endpointing_delay: Seconds(delay.as_secs_f64()),
            language: Some(language.clone().unwrap_or_default()),
            source: wait.endpointing.source.to_owned(),
            from_cache: *from_cache,
            probability: p.probability,
            unlikely_threshold: p.threshold,
            detection_delay: p.detection_delay.map(Seconds),
        });
        let ended_at = trace.now();
        trace.end(span, EouDetectionEnd { ended_at, prediction });
    }

    /// A transcript the turn was heard with.
    pub(crate) fn transcript(&mut self, received_at: f64, kind: &str, text: &str, request: &str) {
        if !request.is_empty() && !self.request_ids.iter().any(|r| r == request) {
            self.request_ids.push(request.to_owned());
        }
        self.stt_events.push(serde_json::json!({
            "received_at": received_at,
            "type": kind,
            "transcript_length": text.chars().count(),
        }));
    }

    /// An overlap verdict on the speech being heard; the last one wins.
    pub(crate) fn overlap(&mut self, e: &OverlappingSpeechEvent) {
        let Some(speaking) = self.speaking.as_mut() else { return };
        speaking.verdict = Some(Verdict {
            interruption: e.is_interruption,
            probability: Some(e.probability),
            total: e.total_duration,
            prediction: e.prediction_duration,
            delay: e.detection_delay,
        });
    }

    /// The overlap ended with no verdict: the speech did not interrupt.
    pub(crate) fn overlap_ended(&mut self) {
        let Some(speaking) = self.speaking.as_mut() else { return };
        speaking.verdict.get_or_insert(Verdict {
            interruption: false,
            probability: None,
            total: 0.0,
            prediction: 0.0,
            delay: 0.0,
        });
    }

    /// The turn was decided: its segment and its wait end, and what the decision knew is kept for
    /// its end. Returns the outcome it ends with: a turn with no reply ends now, a committed one
    /// when it is answered.
    pub(crate) fn decide(
        &mut self,
        trace: &mut Trace,
        info: &EndOfTurnInfo,
        verdict: UserTurnCommitVerdict,
    ) -> UserTurnOutcome {
        let now = trace.now();
        if let Some(speaking) = self.speaking.take() {
            end_speaking(trace, now, speaking);
        }
        let outcome = UserTurnOutcome::from(verdict);
        if let Some(mut wait) = self.wait.take() {
            let committed = outcome == UserTurnOutcome::Committed;
            // The decision did not commit the turn: 1.8.10's `on_end_of_turn` refusal.
            wait.not_committed += u32::from(!committed);
            let outcome = if committed { EouOutcome::Committed } else { EouOutcome::Dropped };
            end_wait(trace, now, wait, outcome);
        }
        self.decided = Decided {
            transcript: info.new_transcript.clone(),
            confidence: Some(info.transcript_confidence),
            transcription_delay: info.metrics.transcription_delay.map(|d| d.as_secs_f64()),
            end_of_turn_delay: info.metrics.end_of_turn_delay.map(|d| d.as_secs_f64()),
        };
        outcome
    }

    /// `on_user_turn_completed` runs for the committed turn: its span, which the host runs it
    /// under.
    pub(crate) fn hook_issued(&mut self, trace: &mut Trace, agent: &str) -> Option<SpanRef> {
        let turn = self.span.as_ref()?;
        let span = trace.open_in(turn, OnUserTurnCompleted { agent_label: agent.into() });
        let context = trace.context(&span);
        if let Some(previous) = self.hook.replace(span) {
            trace.end(previous, ());
        }
        context
    }

    /// The hook returned, raised or timed out: it ends, and the turn with it.
    pub(crate) fn hook_finished(
        &mut self,
        trace: &mut Trace,
        failure: Option<&dyn SpanError>,
        stop_response: bool,
    ) {
        if let Some(hook) = self.hook.take() {
            if stop_response {
                trace.event(&hook, StopResponse);
            }
            if let Some(failure) = failure {
                trace.fail(&hook, failure);
            }
            trace.end(hook, ());
        }
        self.end(trace, UserTurnOutcome::Committed);
    }

    /// The turn ends with `outcome`, everything inside it first. Ending twice is a no-op.
    pub(crate) fn end(&mut self, trace: &mut Trace, outcome: UserTurnOutcome) {
        let Some(span) = self.span.take() else { return };
        let now = trace.now();
        if let Some(speaking) = self.speaking.take() {
            end_speaking(trace, now, speaking);
        }
        if let Some(wait) = self.wait.take() {
            end_wait(trace, now, wait, EouOutcome::Dropped);
        }
        if let Some(hook) = self.hook.take() {
            trace.end(hook, ());
        }
        let d = std::mem::take(&mut self.decided);
        let end = UserTurnEnd {
            outcome,
            transcript: Content(d.transcript),
            confidence: d.confidence,
            transcription_delay: Measured(d.transcription_delay.map(Seconds)),
            end_of_turn_delay: Measured(d.end_of_turn_delay.map(Seconds)),
            provider_request_ids: std::mem::take(&mut self.request_ids),
            stt_events: Json(serde_json::Value::Array(std::mem::take(&mut self.stt_events))),
            resumes: (self.resumes > 0).then_some(self.resumes),
        };
        trace.end(span, end);
    }
}

fn end_speaking(trace: &mut Trace, ended_at: Ts, speaking: Speaking) {
    let v = speaking.verdict;
    let end = UserSpeakingEnd {
        ended_at,
        is_interruption: v.map(|v| BoolString(v.interruption)),
        probability: v.and_then(|v| v.probability),
        total_duration: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.total)),
        prediction_duration: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.prediction)),
        detection_delay: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.delay)),
    };
    trace.end(speaking.span, end);
}

fn end_wait(trace: &mut Trace, at: Ts, wait: Wait, outcome: EouOutcome) {
    // The detector was still deciding: the wait's end supersedes it.
    if let Some(detection) = wait.detection {
        trace.event(&detection, Superseded { at, outcome });
        trace.end(detection, EouDetectionEnd { ended_at: at, prediction: None });
    }
    let end = EouWaitEnd {
        ended_at: at,
        outcome,
        wait_duration: Seconds(at.saturating_duration_since(wait.started).as_secs_f64()),
        rearm_count: wait.rearms,
        not_committed_count: wait.not_committed,
    };
    trace.end(wait.span, end);
}

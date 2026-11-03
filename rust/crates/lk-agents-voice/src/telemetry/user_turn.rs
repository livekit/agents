//! The user turn's spans (`telemetry.md`, "The user turn"): opened by the turn's first evidence,
//! backdated to it, and ended once with its [`UserTurnOutcome`].

use super::participant_attrs;
use crate::agent_session::HookId;
use crate::audio_recognition::{EndOfTurnInfo, EouDetected, TurnId, UserTurnCommitVerdict};
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::{BoolString, Content, Json, Measured, Seconds, SpanRef};
use lk_agents_abi::telemetry::spans::session::AgentSession;
use lk_agents_abi::telemetry::spans::user_turn::{
    EouDetection, EouOutcome, EouPrediction, EouWait, EouWaitEnd, OnUserTurnCompleted, Rearmed,
    StopResponse, Superseded, UserResumed, UserSpeaking, UserSpeakingEnd, UserTurn, UserTurnEnd,
    UserTurnOrigin, UserTurnOutcome,
};
use lk_agents_abi::voice::events::OverlappingSpeechEvent;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_core::telemetry::span::{ManualSpan, SpanFailure, Trace};
use std::collections::BTreeMap;

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
    fn decl(self) -> EouWait {
        EouWait { source: self.source.to_owned(), endpointing_delay: Seconds(self.delay) }
    }
}

struct Speaking {
    span: ManualSpan<UserSpeaking>,
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
    span: ManualSpan<EouWait>,
    started: Ts,
    endpointing: Endpointing,
    rearms: u32,
    not_committed: u32,
    /// The prediction the countdown waits on.
    detection: Option<ManualSpan<EouDetection>>,
}

/// What the decision knew of the turn, written when it ends.
#[derive(Default)]
struct Decided {
    transcript: String,
    confidence: Option<f64>,
    transcription_delay: Option<f64>,
    end_of_turn_delay: Option<f64>,
}

struct Turn {
    span: ManualSpan<UserTurn>,
    speaking: Option<Speaking>,
    wait: Option<Wait>,
    hook: Option<(HookId, ManualSpan<OnUserTurnCompleted>)>,
    resumes: u32,
    request_ids: Vec<String>,
    stt_events: Vec<serde_json::Value>,
    decided: Decided,
}

/// The turn gathering evidence, and the decided ones still running their hook.
#[derive(Default)]
pub(crate) struct UserTurns {
    current: Option<Turn>,
    decided: BTreeMap<TurnId, Turn>,
    /// Every committed turn's span, for the chat items it produced.
    spans: BTreeMap<TurnId, SpanRef>,
    /// The linked participant the turns are heard from.
    pub(crate) participant: Option<ParticipantInfo>,
}

impl UserTurns {
    fn current(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        at: Ts,
        origin: UserTurnOrigin,
        stt: &Stt,
    ) -> &mut Turn {
        let participant = &self.participant;
        self.current.get_or_insert_with(|| {
            let decl = UserTurn {
                origin,
                participant: participant.as_ref().map(participant_attrs),
                stt_model: stt.model.clone(),
                stt_provider: stt.provider.clone(),
            };
            Turn::new(root.child_at(trace, at, decl))
        })
    }

    /// A speech onset at `at`: it opens the turn, or resumes the one waiting on its decision.
    pub(crate) fn speech_started(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        at: Ts,
        origin: UserTurnOrigin,
        stt: &Stt,
    ) {
        let participant = self.participant.as_ref().map(participant_attrs);
        let turn = self.current(trace, root, at, origin, stt);
        if let Some(wait) = turn.wait.take() {
            wait.span.event(trace, at, UserResumed);
            turn.resumes += 1;
            end_wait(trace, at, wait, EouOutcome::UserResumed);
        }
        if turn.speaking.is_none() {
            let span = turn.span.child_at(trace, at, UserSpeaking { participant });
            turn.speaking = Some(Speaking { span, verdict: None });
        }
    }

    /// The end of speech at `at` starts the countdown to the turn's decision.
    pub(crate) fn speech_ended(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        let Some(turn) = self.current.as_mut() else { return };
        if let Some(speaking) = turn.speaking.take() {
            end_speaking(trace, at, speaking);
        }
        turn.trigger(trace, at, endpointing);
    }

    /// A later trigger (a final transcript, a manual commit) restarts the countdown of the turn
    /// waiting on its decision: the wait is re-armed, or opened by a manual commit.
    pub(crate) fn retriggered(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        let Some(turn) = self.current.as_mut() else { return };
        if turn.wait.is_some() || endpointing.source == MANUAL {
            turn.trigger(trace, at, endpointing);
        }
    }

    /// The countdown's end-of-turn detection: it waits on the detector, or resolved.
    pub(crate) fn detection(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        detected: &EouDetected,
        chat_ctx: serde_json::Value,
    ) {
        let Some(turn) = self.current.as_mut() else { return };
        let Some(wait) = turn.wait.as_mut() else { return };
        let (prediction, delay, from_cache, language) = match detected {
            // One detection at a time: a new countdown's replaces the last.
            EouDetected::Awaiting => {
                if let Some(span) = wait.detection.take() {
                    span.end(trace, now, None);
                }
                wait.detection = Some(wait.span.child(trace, now, EouDetection));
                return;
            }
            EouDetected::Resolved { prediction, delay, from_cache, language } => {
                (prediction, delay, from_cache, language)
            }
        };
        let span = match wait.detection.take() {
            Some(span) => span,
            None => wait.span.child(trace, now, EouDetection),
        };
        wait.endpointing.delay = delay.as_secs_f64();
        wait.span.set(trace, wait.endpointing.decl());
        let end = prediction.map(|p| EouPrediction {
            chat_ctx: Content(Json(chat_ctx)),
            endpointing_delay: Seconds(delay.as_secs_f64()),
            language: Some(language.clone().unwrap_or_default()),
            source: wait.endpointing.source.to_owned(),
            from_cache: *from_cache,
            probability: p.probability,
            unlikely_threshold: p.threshold,
            detection_delay: p.detection_delay.map(Seconds),
        });
        span.end(trace, now, end);
    }

    /// A transcript: it opens a turn no speech edge opened.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcript(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        received_at: f64,
        kind: &str,
        text: &str,
        request_id: &str,
        stt: &Stt,
    ) {
        if text.is_empty() && self.current.is_none() {
            return;
        }
        let turn = self.current(trace, root, now, UserTurnOrigin::Transcript, stt);
        if !request_id.is_empty() && !turn.request_ids.iter().any(|r| r == request_id) {
            turn.request_ids.push(request_id.to_owned());
        }
        turn.stt_events.push(serde_json::json!({
            "received_at": received_at,
            "type": kind,
            "transcript_length": text.chars().count(),
        }));
    }

    /// An overlap verdict on the speech being heard; the last one wins.
    pub(crate) fn overlap(&mut self, e: &OverlappingSpeechEvent) {
        let Some(speaking) = self.current.as_mut().and_then(|t| t.speaking.as_mut()) else {
            return;
        };
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
        let Some(speaking) = self.current.as_mut().and_then(|t| t.speaking.as_mut()) else {
            return;
        };
        speaking.verdict.get_or_insert(Verdict {
            interruption: false,
            probability: None,
            total: 0.0,
            prediction: 0.0,
            delay: 0.0,
        });
    }

    /// The turn was decided: a turn with no reply ends now, a committed one when its hook returns.
    pub(crate) fn decided(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        info: &EndOfTurnInfo,
        verdict: UserTurnCommitVerdict,
        stt: &Stt,
    ) {
        if self.decided.contains_key(&info.turn_id) {
            return;
        }
        let at = info.metrics.started_speaking_at.unwrap_or(now);
        self.current(trace, root, at, UserTurnOrigin::Transcript, stt);
        let Some(mut turn) = self.current.take() else { return };
        if let Some(speaking) = turn.speaking.take() {
            end_speaking(trace, now, speaking);
        }
        let outcome = UserTurnOutcome::from(verdict);
        if let Some(mut wait) = turn.wait.take() {
            let committed = outcome == UserTurnOutcome::Committed;
            // The decision did not commit the turn: 1.8.10's `on_end_of_turn` refusal.
            wait.not_committed += u32::from(!committed);
            let outcome = if committed { EouOutcome::Committed } else { EouOutcome::Dropped };
            end_wait(trace, now, wait, outcome);
        }
        turn.decided = Decided {
            transcript: info.new_transcript.clone(),
            confidence: Some(info.transcript_confidence),
            transcription_delay: info.metrics.transcription_delay.map(|d| d.as_secs_f64()),
            end_of_turn_delay: info.metrics.end_of_turn_delay.map(|d| d.as_secs_f64()),
        };
        if outcome == UserTurnOutcome::Committed {
            self.spans.insert(info.turn_id, turn.span.context());
            self.decided.insert(info.turn_id, turn);
        } else {
            end_turn(trace, now, turn, outcome);
        }
    }

    /// `on_user_turn_completed` runs for a committed turn.
    pub(crate) fn hook_issued(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        turn: TurnId,
        hook: HookId,
        agent: &str,
    ) -> Option<SpanRef> {
        let turn = self.decided.get_mut(&turn)?;
        let span = turn.span.child(trace, now, OnUserTurnCompleted { agent_label: agent.into() });
        let context = span.context();
        turn.hook = Some((hook, span));
        Some(context)
    }

    /// The hook returned: it ends, and the turn with it.
    pub(crate) fn hook_finished(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        turn: TurnId,
        failure: Option<&dyn SpanFailure>,
        stop_response: bool,
    ) {
        let Some(mut turn) = self.decided.remove(&turn) else { return };
        if let Some((_, hook)) = turn.hook.take() {
            if stop_response {
                hook.event(trace, now, StopResponse);
            }
            if let Some(failure) = failure {
                hook.fail(trace, now, failure);
            }
            hook.end(trace, now, ());
        }
        end_turn(trace, now, turn, UserTurnOutcome::Committed);
    }

    /// A decided turn ends without (or before) its hook.
    pub(crate) fn retire(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        turn: TurnId,
        outcome: UserTurnOutcome,
    ) {
        if let Some(turn) = self.decided.remove(&turn) {
            end_turn(trace, now, turn, outcome);
        }
    }

    /// The turn being heard ends undecided.
    pub(crate) fn abandon(&mut self, trace: &mut Trace, now: Ts, outcome: UserTurnOutcome) {
        if let Some(turn) = self.current.take() {
            end_turn(trace, now, turn, outcome);
        }
    }

    /// Typed user input: a turn of its own, committed on arrival.
    pub(crate) fn typed(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        text: &str,
    ) -> SpanRef {
        let decl = UserTurn {
            origin: UserTurnOrigin::Text,
            participant: self.participant.as_ref().map(participant_attrs),
            stt_model: None,
            stt_provider: None,
        };
        let mut turn = Turn::new(root.child(trace, now, decl));
        turn.decided.transcript = text.to_owned();
        let span = turn.span.context();
        end_turn(trace, now, turn, UserTurnOutcome::Committed);
        span
    }

    /// The span of committed turn `turn`.
    pub(crate) fn span_of(&self, turn: TurnId) -> Option<SpanRef> {
        self.spans.get(&turn).copied()
    }

    /// The session closed: every open turn ends `closed`.
    pub(crate) fn close(&mut self, trace: &mut Trace, now: Ts) {
        self.abandon(trace, now, UserTurnOutcome::Closed);
        for (_, turn) in std::mem::take(&mut self.decided) {
            end_turn(trace, now, turn, UserTurnOutcome::Closed);
        }
    }
}

fn end_speaking(trace: &mut Trace, at: Ts, speaking: Speaking) {
    let v = speaking.verdict;
    let end = UserSpeakingEnd {
        is_interruption: v.map(|v| BoolString(v.interruption)),
        probability: v.and_then(|v| v.probability),
        total_duration: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.total)),
        prediction_duration: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.prediction)),
        detection_delay: v.filter(|v| v.probability.is_some()).map(|v| Seconds(v.delay)),
    };
    speaking.span.end(trace, at, end);
}

/// `lk.eou.source` of a manual commit.
pub(crate) const MANUAL: &str = "manual";

impl Turn {
    fn new(span: ManualSpan<UserTurn>) -> Self {
        Turn {
            span,
            speaking: None,
            wait: None,
            hook: None,
            resumes: 0,
            request_ids: Vec::new(),
            stt_events: Vec::new(),
            decided: Decided::default(),
        }
    }

    /// A countdown trigger: it opens the wait, or re-arms the open one.
    fn trigger(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        match &mut self.wait {
            Some(wait) => {
                wait.rearms += 1;
                wait.endpointing = endpointing;
                let source = endpointing.source.to_owned();
                wait.span.event(trace, at, Rearmed { source });
                wait.span.set(trace, endpointing.decl());
            }
            None => {
                let span = self.span.child_at(trace, at, endpointing.decl());
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
}

fn end_wait(trace: &mut Trace, at: Ts, wait: Wait, outcome: EouOutcome) {
    // The detector was still deciding: the wait's end supersedes it.
    if let Some(detection) = wait.detection {
        detection.event(trace, at, Superseded { outcome });
        detection.end(trace, at, None);
    }
    let end = EouWaitEnd {
        outcome,
        wait_duration: Seconds(at.saturating_duration_since(wait.started).as_secs_f64()),
        rearm_count: wait.rearms,
        not_committed_count: wait.not_committed,
    };
    wait.span.end(trace, at, end);
}

fn end_turn(trace: &mut Trace, now: Ts, mut turn: Turn, outcome: UserTurnOutcome) {
    if let Some(speaking) = turn.speaking.take() {
        end_speaking(trace, now, speaking);
    }
    if let Some(wait) = turn.wait.take() {
        end_wait(trace, now, wait, EouOutcome::Dropped);
    }
    if let Some((_, hook)) = turn.hook.take() {
        hook.end(trace, now, ());
    }
    let d = turn.decided;
    let end = UserTurnEnd {
        outcome,
        transcript: Content(d.transcript),
        confidence: d.confidence,
        transcription_delay: Measured(d.transcription_delay.map(Seconds)),
        end_of_turn_delay: Measured(d.end_of_turn_delay.map(Seconds)),
        provider_request_ids: turn.request_ids,
        stt_events: Json(serde_json::Value::Array(turn.stt_events)),
        resumes: (turn.resumes > 0).then_some(turn.resumes),
    };
    turn.span.end(trace, now, end);
}

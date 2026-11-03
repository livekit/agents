//! The user turn's spans (`telemetry.md`, "The user turn"): opened by the turn's first evidence,
//! backdated to it, and ended once with its [`UserTurnOutcome`].

use crate::agent_session::HookId;
use crate::audio_recognition::{EndOfTurnInfo, TurnId, UserTurnCommitVerdict};
use crate::events::EotPredictionEvent;
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::{BoolString, Content, Json, Measured, Seconds};
use lk_agents_abi::telemetry::spans::session::AgentSession;
use lk_agents_abi::telemetry::spans::user_turn::{
    EouDetection, EouOutcome, EouPrediction, EouWait, EouWaitEnd, OnUserTurnCompleted,
    StopResponse, UserResumed, UserSpeaking, UserSpeakingEnd, UserTurn, UserTurnEnd,
    UserTurnOrigin, UserTurnOutcome,
};
use lk_agents_abi::voice::events::OverlappingSpeechEvent;
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

/// The STT a turn is heard with.
#[derive(Debug, Clone, Default)]
pub(crate) struct Stt {
    pub(crate) model: Option<String>,
    pub(crate) provider: Option<String>,
}

/// What decides the end of the turn, as `eou_wait` records it.
pub(crate) struct Endpointing {
    pub(crate) source: &'static str,
    pub(crate) delay: f64,
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
    not_committed: u32,
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
        self.current.get_or_insert_with(|| {
            let decl = UserTurn {
                origin,
                participant: None,
                stt_model: stt.model.clone(),
                stt_provider: stt.provider.clone(),
            };
            Turn {
                span: root.child_at(trace, at, decl),
                speaking: None,
                wait: None,
                hook: None,
                resumes: 0,
                request_ids: Vec::new(),
                stt_events: Vec::new(),
                decided: Decided::default(),
            }
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
        let turn = self.current(trace, root, at, origin, stt);
        if let Some(wait) = turn.wait.take() {
            wait.span.event(trace, at, UserResumed);
            turn.resumes += 1;
            end_wait(trace, at, wait, EouOutcome::UserResumed);
        }
        if turn.speaking.is_none() {
            let span = turn.span.child_at(trace, at, UserSpeaking { participant: None });
            turn.speaking = Some(Speaking { span, verdict: None });
        }
    }

    /// The end of speech at `at` starts the wait for the turn's decision.
    pub(crate) fn speech_ended(&mut self, trace: &mut Trace, at: Ts, endpointing: Endpointing) {
        let Some(turn) = self.current.as_mut() else { return };
        if let Some(speaking) = turn.speaking.take() {
            end_speaking(trace, at, speaking);
        }
        if turn.wait.is_none() {
            let decl = EouWait {
                source: endpointing.source.to_owned(),
                endpointing_delay: Seconds(endpointing.delay),
            };
            let span = turn.span.child_at(trace, at, decl);
            turn.wait = Some(Wait { span, started: at, not_committed: 0 });
        }
    }

    /// A transcript: it opens a turn no speech edge opened.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcript(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        received_at: f64,
        is_final: bool,
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
        let kind = if is_final { "final_transcript" } else { "interim_transcript" };
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

    /// One end-of-turn prediction, from its request to `now`.
    pub(crate) fn prediction(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        e: &EotPredictionEvent,
        endpointing: Endpointing,
    ) {
        let Some(turn) = self.current.as_mut() else { return };
        let started =
            now.saturating_sub(std::time::Duration::from_secs_f64(e.inference_duration.max(0.0)));
        let wait = turn.wait.get_or_insert_with(|| {
            let decl = EouWait {
                source: endpointing.source.to_owned(),
                endpointing_delay: Seconds(endpointing.delay),
            };
            Wait { span: turn.span.child_at(trace, started, decl), started, not_committed: 0 }
        });
        if e.probability < e.threshold {
            wait.not_committed += 1;
        }
        let detection = wait.span.child_at(trace, started, EouDetection);
        let prediction = EouPrediction {
            chat_ctx: Content(Json(serde_json::Value::Null)),
            endpointing_delay: Seconds(e.delay.as_secs_f64()),
            language: None,
            source: endpointing.source.to_owned(),
            from_cache: false,
            probability: e.probability,
            unlikely_threshold: Some(e.threshold),
            detection_delay: (e.inference_duration > 0.0).then_some(Seconds(e.inference_duration)),
        };
        detection.end(trace, now, Some(prediction));
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
        if let Some(wait) = turn.wait.take() {
            let committed = outcome == UserTurnOutcome::Committed;
            end_wait(
                trace,
                now,
                wait,
                if committed { EouOutcome::Committed } else { EouOutcome::Dropped },
            );
        }
        turn.decided = Decided {
            transcript: info.new_transcript.clone(),
            confidence: Some(info.transcript_confidence),
            transcription_delay: info.metrics.transcription_delay.map(|d| d.as_secs_f64()),
            end_of_turn_delay: info.metrics.end_of_turn_delay.map(|d| d.as_secs_f64()),
        };
        if outcome == UserTurnOutcome::Committed {
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
    ) {
        let Some(turn) = self.decided.get_mut(&turn) else { return };
        let span = turn.span.child(trace, now, OnUserTurnCompleted { agent_label: agent.into() });
        turn.hook = Some((hook, span));
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
    ) {
        let decl = UserTurn {
            origin: UserTurnOrigin::Text,
            participant: None,
            stt_model: None,
            stt_provider: None,
        };
        let span = root.child(trace, now, decl);
        let mut turn = Turn {
            span,
            speaking: None,
            wait: None,
            hook: None,
            resumes: 0,
            request_ids: Vec::new(),
            stt_events: Vec::new(),
            decided: Decided::default(),
        };
        turn.decided.transcript = text.to_owned();
        end_turn(trace, now, turn, UserTurnOutcome::Committed);
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

fn end_wait(trace: &mut Trace, at: Ts, wait: Wait, outcome: EouOutcome) {
    let end = EouWaitEnd {
        outcome,
        wait_duration: Seconds(at.saturating_duration_since(wait.started).as_secs_f64()),
        rearm_count: 0,
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

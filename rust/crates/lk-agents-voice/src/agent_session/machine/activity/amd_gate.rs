//! The session's AMD reply gate: it owns the hold, AMD owns only the policy.
//!
//! While an AMD run is attached, the session holds speech authorization by default and replies
//! to a committed turn only on that turn's decision. It owns the live facts: which turn is the
//! latest, whether the participant speaks, and how long they have been silent.
//!
//! - **State**: [`Gate`] on the activity, its [`Hold`] read by the scheduling pass
//!   ([`Gate::authorizes`]).
//! - **Inputs**: [`AmdCommand::Attach`], [`AmdAnswer`] (`VoiceEvent::Amd`), the turn path's
//!   commit, hook return and exits, participant speech boundaries, and `Timer::AmdSilence`.
//! - **Effects**: [`AmdObservation`]s as `VoiceEffect::Amd`, `GateStep` and
//!   `SpeechStep::AuthorizationHeld` traces.
//! - **Terminal**: `Detach` settles the pending turns, removes the gate and reports `Detached`.
//!   Later commits are not gated, and a new run can attach.

use crate::agent::{Source, Visibility};
use crate::agent_session::effects::{Timer, VoiceEffect};
use crate::agent_session::machine::activity::scheduling::{PreemptiveDiscard, SpeechId, Status};
use crate::agent_session::options::{AgentSessionOptions, ConfigLayer, OverrideLayer};
use crate::agent_session::{Answer, RequestId, VoiceDriver};
use crate::amd::{
    AmdAnswer, AmdAttachRefused, AmdCommand, AmdObservation, AmdRunId, DetachCancel, Opening,
    ReplyDecision, SpeechObservation, ToolCallRecord, TurnEnd, UserSpeechObservation,
};
use crate::audio_recognition::{EndOfTurnInfo, TurnEdits, TurnId};
use crate::events::SessionEvent;
use crate::generation::ReplySource;
use crate::trace::{AnswerDrop, GateStep, InterruptCause, OpeningDrop, SpeechStep};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{ChatItem, Ts};
use std::collections::BTreeSet;
use std::time::Duration;

/// The override layer that forces interruptions on while a run is attached. Host layers are
/// minted from 1 upward by the session that holds them, so this id never collides.
const AMD_OVERRIDE_LAYER: u64 = u64::MAX;

/// What authorization waits on while a run is attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hold {
    /// AMD attached; no turn has been allowed yet.
    Entry,
    /// The latest turn has no decision applied yet.
    Undecided { turn: TurnId },
    /// The latest turn was denied, or ended without a reply (`skip_reply`, `StopResponse`). It
    /// stays held until the next commit.
    Denied { turn: TurnId },
    /// An allowed reply waits until the participant has been silent for `needed`, measured from
    /// the session's latest speech boundary. `deadline` is the armed expiry, `None` while the
    /// participant speaks.
    Silence { turn: TurnId, needed: Duration, deadline: Option<Ts> },
    /// Authorization is free.
    Open,
}

/// The attached run's gate.
pub(crate) struct Gate {
    run: AmdRunId,
    /// Set by `Activated`.
    active: bool,
    /// The latest committed turn and its decision, once known.
    latest: Option<Latest>,
    /// The latest turn, back from its hook before its decision. At most one.
    parked: Option<ParkedTurn>,
    hold: Hold,
    /// Turns committed under the gate that have not ended.
    turns: BTreeSet<TurnId>,
    /// The latest allowed turn's reply and the decision it was made under.
    reply: Option<GateReply>,
    /// The opening, the only speech that bypasses the hold, until it ends.
    opening: Option<OpeningSpeech>,
    /// The latest participant speech boundary the silence hold measures from.
    boundary: Option<Ts>,
    /// A boundary came since the previous commit.
    boundary_since_commit: bool,
    /// The `AgentBusy` last reported; `None` before the first.
    busy: Option<bool>,
}

struct Latest {
    turn: TurnId,
    /// A decision that arrived while the turn was still in its customer hook.
    decision: Option<ReplyDecision>,
    /// The turn's decision was applied, or the turn ended without one.
    settled: bool,
}

/// The latest turn, back from its customer hook, waiting for its decision.
pub(crate) struct ParkedTurn {
    info: EndOfTurnInfo,
    edits: TurnEdits,
}

struct GateReply {
    turn: TurnId,
    speech: SpeechId,
    decision: ReplyDecision,
}

struct OpeningSpeech {
    speech: SpeechId,
    played: bool,
}

impl Gate {
    fn new(run: AmdRunId) -> Self {
        Gate {
            run,
            active: false,
            latest: None,
            parked: None,
            hold: Hold::Entry,
            turns: BTreeSet::new(),
            reply: None,
            opening: None,
            boundary: None,
            boundary_since_commit: false,
            busy: None,
        }
    }

    /// The scheduler may authorize `speech`: the hold is open, or it is the opening.
    pub(crate) fn authorizes(&self, speech: SpeechId) -> bool {
        self.hold == Hold::Open || self.opening.as_ref().is_some_and(|o| o.speech == speech)
    }

    fn latest_turn(&self) -> Option<TurnId> {
        self.latest.as_ref().map(|latest| latest.turn)
    }

    pub(crate) fn parked_turn(&self) -> Option<TurnId> {
        self.parked.as_ref().map(|parked| parked.info.turn_id)
    }

    pub(crate) fn parked_info(&self) -> Option<&EndOfTurnInfo> {
        self.parked.as_ref().map(|parked| &parked.info)
    }

    pub(crate) fn parked_info_mut(&mut self) -> Option<&mut EndOfTurnInfo> {
        self.parked.as_mut().map(|parked| &mut parked.info)
    }
}

impl super::AgentActivity {
    pub(crate) fn amd_attached(&self) -> bool {
        self.gate.is_some()
    }
}

// Attach, answers and detach.
impl VoiceDriver {
    pub(crate) fn on_amd(&mut self, now: Ts, cmd: AmdCommand) {
        match cmd {
            AmdCommand::Attach { run, call } => self.attach_amd(now, run, call),
        }
    }

    fn attach_amd(&mut self, now: Ts, run: AmdRunId, call: Option<RequestId>) {
        let refused = if self.session.closing.is_some() {
            Some(AmdAttachRefused::Closing)
        } else if self.activity.gate.is_some() {
            Some(AmdAttachRefused::AlreadyAttached)
        } else if self.activity.in_window() {
            Some(AmdAttachRefused::HandoffInFlight)
        } else if self.activity.voice_tasks.running() {
            Some(AmdAttachRefused::TaskRunning)
        // Realtime commit paths bypass the gate until implementation step 4.
        } else if self.activity.realtime().is_some() {
            Some(AmdAttachRefused::Realtime)
        } else {
            None
        };
        if let Some(why) = refused {
            self.record_trace(GateStep::AttachRefused { run, why });
            return self.answer(call, Answer::AmdAttach(Err(why)));
        }
        self.activity.gate = Some(Gate::new(run));
        self.record_trace(GateStep::Attached { run });
        self.record_trace(SpeechStep::AuthorizationHeld(Hold::Entry));
        // Interruptions are forced on, and restored by releasing the layer at detach.
        let mut options = AgentSessionOptions::default();
        options.turn.interruption.enabled = Some(true);
        let layer = OverrideLayer { id: AMD_OVERRIDE_LAYER, options };
        let diff = self.session.config.push_override(layer);
        self.apply_config(now, ConfigLayer::Override, diff);
        // No participant speech exists before activation, on either side.
        self.route_session_audio();
        // A speech queued or playing before the run, such as a greeting, is watched too.
        let speaking = self.session.states.agent_state() == AgentState::Speaking;
        let current = self.activity.speech_scheduler.current_speech();
        let existing: Vec<_> = (self.activity.speech_scheduler.speech_ids())
            .filter_map(|id| {
                let s = &self.activity.speech_scheduler;
                let state = s.speech(id).filter(|h| !h.done())?;
                let kind = s.kind(id)?;
                let live = matches!(s.status(id)?, Status::Queued | Status::Current);
                let playing = speaking && current == Some(id) && !state.interrupted();
                live.then_some((id, kind, s.add_to_chat_ctx(id), playing))
            })
            .collect();
        for (speech, kind, add_to_chat_ctx, playing) in existing {
            let event = SpeechObservation::Created { kind, add_to_chat_ctx, playing };
            self.observe_amd(AmdObservation::Speech { speech, event, at: now });
        }
        self.report_agent_busy(now);
        self.answer(call, Answer::AmdAttach(Ok(())));
    }

    pub(crate) fn on_amd_answer(&mut self, now: Ts, answer: AmdAnswer) {
        let (run, turn) = match &answer {
            AmdAnswer::Activated { run, .. } | AmdAnswer::Detach { run, .. } => (*run, None),
            AmdAnswer::ReplyDecided { run, turn, .. }
            | AmdAnswer::InsertUserItem { run, turn, .. } => (*run, Some(*turn)),
        };
        if self.activity.gate.as_ref().map(|gate| gate.run) != Some(run) {
            return self.record_trace(GateStep::AnswerDropped {
                run,
                turn,
                why: AnswerDrop::OtherRun,
            });
        }
        match answer {
            AmdAnswer::Activated { opening, .. } => self.amd_activated(now, opening),
            AmdAnswer::ReplyDecided { turn, decision, .. } => self.amd_decided(now, turn, decision),
            // Only a realtime model without user transcription needs it; realtime sessions are
            // refused at attach until implementation step 4.
            AmdAnswer::InsertUserItem { turn, .. } => self.record_trace(GateStep::AnswerDropped {
                run,
                turn: Some(turn),
                why: AnswerDrop::Unsupported,
            }),
            AmdAnswer::Detach { pending, cancel, .. } => self.amd_detach(now, pending, cancel),
        }
    }

    /// The activation instant: participant audio flows from here, and the opening is checked
    /// against what the session knows now.
    fn amd_activated(&mut self, now: Ts, opening: Option<Opening>) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        let run = gate.run;
        if gate.active {
            return self.record_trace(GateStep::AnswerDropped {
                run,
                turn: None,
                why: AnswerDrop::AlreadyActive,
            });
        }
        gate.active = true;
        self.record_trace(GateStep::Activated { run });
        self.route_session_audio();
        let Some(opening) = opening else { return };
        if self.session.states.user_speaking() {
            return self.record_trace(GateStep::OpeningDropped { why: OpeningDrop::UserSpeaking });
        }
        let speech = match opening {
            Opening::Text(_)
                if self.session.tts.is_none() && self.session.outputs.plays_audio() =>
            {
                return self.record_trace(GateStep::OpeningDropped { why: OpeningDrop::NoVoice });
            }
            Opening::Instructions(_) if self.session.llm.is_none() => {
                return self.record_trace(GateStep::OpeningDropped { why: OpeningDrop::NoModel });
            }
            Opening::Text(text) => {
                use crate::agent_session::machine::activity::scheduling::SpeechKind;
                use crate::agent_session::machine::activity::speech::SpeechSpec;
                let spec = SpeechSpec::new(SpeechKind::Say, ReplySource::Text(text), true);
                self.start_speech(now, spec)
            }
            Opening::Instructions(instructions) => {
                let none = Some(lk_agents_abi::ToolChoice::Forbidden);
                let (user, chat_ctx, tools) = (None, None, None);
                let instructions = Some(instructions);
                self.generate_reply_speech(
                    now,
                    user,
                    instructions,
                    Some(true),
                    none,
                    chat_ctx,
                    tools,
                    false,
                    None,
                )
            }
        };
        // Scheduling runs after this transition, so the bypass is in place before it.
        if let Some(gate) = self.activity.gate.as_mut() {
            gate.opening = Some(OpeningSpeech { speech, played: false });
        }
        self.record_trace(GateStep::OpeningBypass { speech });
    }

    /// The decision for `turn`: kept only for the latest committed turn, applied at once to a
    /// parked one.
    fn amd_decided(&mut self, now: Ts, turn: TurnId, decision: ReplyDecision) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        let (run, allow) = (gate.run, decision.allow);
        let why = match gate.latest.as_mut() {
            Some(latest) if latest.turn != turn => Some(AnswerDrop::NotLatest),
            None => Some(AnswerDrop::NotLatest),
            Some(latest) if latest.settled || latest.decision.is_some() => {
                Some(AnswerDrop::Duplicate)
            }
            Some(latest) => {
                latest.decision = Some(decision.clone());
                None
            }
        };
        if let Some(why) = why {
            return self.record_trace(GateStep::AnswerDropped { run, turn: Some(turn), why });
        }
        let parked = gate.parked.take_if(|parked| parked.info.turn_id == turn);
        let decision = parked.as_ref().and_then(|_| {
            let latest = gate.latest.as_mut()?;
            latest.settled = true;
            latest.decision.take()
        });
        self.record_trace(GateStep::Decided { run, turn, allow });
        if let (Some(ParkedTurn { info, edits }), Some(decision)) = (parked, decision) {
            self.apply_reply_decision(now, info, edits, decision, true);
        }
    }

    /// AMD completed (spec §12.3): cancel what it forbids, restore the session, settle the
    /// pending turns, free authorization, remove the gate.
    fn amd_detach(&mut self, now: Ts, pending: ReplyDecision, cancel: DetachCancel) {
        // 1. A cancel that forbids replies takes every speech with no playout yet.
        if cancel == DetachCancel::ForbidReplies {
            self.cancel_unplayed_speeches(now);
        }
        // 2. The session's own settings and rules come back.
        let diff = self.session.config.release_override(AMD_OVERRIDE_LAYER);
        self.apply_config(now, ConfigLayer::Override, diff);
        // 3. Settle turns: a parked turn now; one in its hook carries the decision to
        // `finish_turn`.
        let parked = self.activity.gate.as_mut().and_then(|gate| gate.parked.take());
        if let Some(ParkedTurn { info, edits }) = parked {
            self.record_trace(GateStep::Settled { turn: info.turn_id, allow: pending.allow });
            self.apply_reply_decision(now, info, edits, pending.clone(), false);
        }
        let Some(gate) = self.activity.gate.as_ref() else { return };
        let in_hook: Vec<TurnId> = (self.activity.pending_turn.iter().map(|t| t.info.turn_id))
            .chain(self.activity.waiting_turns.iter().map(|t| t.info.turn_id))
            .filter(|turn| gate.turns.contains(turn))
            .collect();
        for turn in in_hook {
            self.record_trace(GateStep::Settled { turn, allow: pending.allow });
            self.activity.amd_settled.insert(turn, pending.clone());
        }
        // 4. Free authorization: a reply held for silence plays at once.
        self.set_hold(now, Hold::Open);
        self.clear_deadline_if_set(Timer::AmdSilence);
        // 5. The gate is gone: later commits are not gated.
        let Some(gate) = self.activity.gate.take() else { return };
        let run = gate.run;
        self.route_session_audio();
        self.record_trace(GateStep::Detached { run });
        self.record_effect(VoiceEffect::Amd { run, observation: AmdObservation::Detached { run } });
    }

    /// `_cancel_pending_speeches`: the preemptive generation, every queued or held speech, and
    /// the current one unless it plays or is paused.
    fn cancel_unplayed_speeches(&mut self, now: Ts) {
        self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
        let paused = self.activity.paused_speech.paused_handle();
        let speaking = self.session.states.agent_state() == AgentState::Speaking;
        let s = &self.activity.speech_scheduler;
        let unplayed: Vec<SpeechId> = s
            .speeches()
            .filter(|(id, handle)| {
                !handle.done()
                    && !handle.interrupted()
                    && match s.status(*id) {
                        Some(Status::Queued) => true,
                        Some(Status::Current) => !speaking && paused != Some(*id),
                        _ => false,
                    }
            })
            .map(|(id, _)| id)
            .collect();
        for speech in unplayed {
            self.force_interrupt_speech(now, speech, InterruptCause::Superseded);
        }
    }

    /// Session STT and VAD hear the participant unless a run is attached and not yet active.
    fn route_session_audio(&mut self) {
        let discarding = self.activity.gate.as_ref().is_some_and(|gate| !gate.active);
        let enabled = self.input.audio_enabled && !discarding;
        if self.input.amd_discard != discarding {
            self.input.amd_discard = discarding;
            let command = crate::data_plane::AudioInputRouterCommand::SetInputEnabled(enabled);
            self.record_effect(VoiceEffect::UpdateAudioInputRouting(command));
        }
    }
}

// The turn path.
impl VoiceDriver {
    /// A replying turn committed under the gate: the older turn is superseded locally,
    /// authorization is held for this one, and the current speech is interrupted.
    pub(crate) fn gate_commit(&mut self, now: Ts, info: &EndOfTurnInfo) {
        let speaking = self.session.states.user_speaking();
        let Some(gate) = self.activity.gate.as_mut() else { return };
        let (run, turn) = (gate.run, info.turn_id);
        // Rule 3: a parked turn is denied, and a reply not yet authorized is cancelled.
        let parked = gate.parked.take();
        let held = gate.reply.take().filter(|reply| {
            let s = &self.activity.speech_scheduler;
            s.status(reply.speech) == Some(Status::Queued)
                && s.speech(reply.speech).is_some_and(|h| !h.done() && !h.interrupted())
        });
        if let Some(ParkedTurn { info: older, edits }) = parked {
            self.record_trace(GateStep::Superseded { turn: older.turn_id, speech: None });
            self.deny_turn(now, older, edits, TurnEnd::Superseded);
        }
        if let Some(reply) = held {
            let speech = Some(reply.speech);
            self.record_trace(GateStep::Superseded { turn: reply.turn, speech });
            self.force_interrupt_speech(now, reply.speech, InterruptCause::Superseded);
        }
        let Some(gate) = self.activity.gate.as_mut() else { return };
        // The silence hold measures from the speech's end; with no edge since the last commit,
        // from the end the endpointing delay implies.
        if !speaking && !gate.boundary_since_commit {
            gate.boundary =
                Some(now.saturating_sub(info.metrics.end_of_turn_delay.unwrap_or_default()));
        }
        gate.boundary_since_commit = false;
        gate.latest = Some(Latest { turn, decision: None, settled: false });
        gate.turns.insert(turn);
        let active = gate.active;
        self.record_trace(GateStep::Committed { run, turn });
        self.observe_amd(AmdObservation::TurnCommitted {
            turn,
            transcript: info.new_transcript.clone(),
            eot_delay: info.metrics.end_of_turn_delay,
            skip_reply: info.skip_reply,
            speaking,
            agent: self.activity.agent.clone(),
            at: now,
        });
        // Before activation no turn is bound: the hold stays at its entry.
        if active {
            self.set_hold(now, Hold::Undecided { turn });
        }
        // The commit interrupt does not wait for an older turn's hook.
        if !info.skip_reply {
            self.interrupt_for_user_turn(now);
        }
    }

    /// The hook returned (or there was none): a turn settled by `Detach` replies on its own;
    /// the latest gated turn replies on its decision or parks for it. `Some` hands the turn back
    /// to the ungated path.
    pub(crate) fn gate_reply(
        &mut self,
        now: Ts,
        info: EndOfTurnInfo,
        edits: TurnEdits,
    ) -> Option<(EndOfTurnInfo, TurnEdits)> {
        if let Some(decision) = self.activity.amd_settled.remove(&info.turn_id) {
            self.apply_reply_decision(now, info, edits, decision, false);
            return None;
        }
        let Some(gate) = self.activity.gate.as_mut() else { return Some((info, edits)) };
        let Some(latest) =
            gate.latest.as_mut().filter(|latest| latest.turn == info.turn_id && !latest.settled)
        else {
            return Some((info, edits));
        };
        match latest.decision.take() {
            Some(decision) => {
                latest.settled = true;
                self.apply_reply_decision(now, info, edits, decision, true);
            }
            None => {
                let turn = info.turn_id;
                gate.parked = Some(ParkedTurn { info, edits });
                self.record_trace(GateStep::Parked { turn });
            }
        }
        None
    }

    /// Apply a decision to its turn. `gated`: the current gate holds for it; else a decision
    /// `Detach` settled, applied with no gate.
    fn apply_reply_decision(
        &mut self,
        now: Ts,
        info: EndOfTurnInfo,
        edits: TurnEdits,
        decision: ReplyDecision,
        gated: bool,
    ) {
        let turn = info.turn_id;
        if !decision.allow {
            if gated {
                self.set_hold_for(now, turn, Hold::Denied { turn });
            }
            return self.deny_turn(now, info, edits, TurnEnd::Denied);
        }
        // As the ungated reply path: an unheard late reply goes, the current speech is cut.
        if info.late {
            self.supersede_unheard_reply(now);
        }
        self.interrupt_for_user_turn(now);
        let Some(speech) = self.reply_to_turn(now, &info, edits, Some(&decision)) else {
            return self.gate_turn_ended(turn, TurnEnd::Dropped);
        };
        // A speech fixes its interruptibility at creation, and the gate forces it on while
        // attached: an adopted preemptive generation takes the value a new reply would have now,
        // the restored one after `Detach`.
        let interruptible = self.session.config.effective.turn.interruption.enabled();
        let s = &self.activity.speech_scheduler;
        if s.speech(speech).is_some_and(|h| h.allow_interruptions() != interruptible) {
            self.set_allow_interruptions(speech, interruptible);
        }
        self.observe_amd(AmdObservation::ReplyCreated { turn, speech });
        self.gate_turn_ended(turn, TurnEnd::Replied);
        if !gated {
            return;
        }
        let Some(gate) = self.activity.gate.as_mut() else { return };
        gate.reply = Some(GateReply { turn, speech, decision: decision.clone() });
        // An allow before activation binds no turn: authorization stays held from entry.
        if !gate.active {
            return;
        }
        let hold = match decision.silence.filter(|needed| !needed.is_zero()) {
            Some(needed) => Hold::Silence { turn, needed, deadline: None },
            None => Hold::Open,
        };
        self.set_hold_for(now, turn, hold);
        self.arm_silence(now);
    }

    /// The turn's words are committed with no reply. A denied turn's preemptive generation goes;
    /// a superseded one leaves it to the newer turn, whose reply decides whether it fits.
    fn deny_turn(&mut self, now: Ts, info: EndOfTurnInfo, edits: TurnEdits, how: TurnEnd) {
        let turn = info.turn_id;
        if self.activity.last_turn.is_none_or(|last| last < turn) {
            self.activity.last_turn = Some(turn);
        }
        if how != TurnEnd::Superseded {
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
        }
        let user = self.user_turn_item(&info, edits.user_text.as_deref());
        self.commit_item(ChatItem::Message(user), Source::UserTurn { turn }, Visibility::Both);
        self.gate_turn_ended(turn, how);
    }

    /// A gated turn left the activity. One that leaves without a reply keeps authorization
    /// held until the next commit.
    pub(crate) fn gate_turn_ended(&mut self, turn: TurnId, how: TurnEnd) {
        self.activity.amd_settled.remove(&turn);
        let Some(gate) = self.activity.gate.as_mut() else { return };
        if !gate.turns.remove(&turn) {
            return;
        }
        if let Some(latest) = gate.latest.as_mut().filter(|latest| latest.turn == turn) {
            latest.settled = true;
        }
        self.observe_amd(AmdObservation::TurnEnded { turn, how });
        if !matches!(how, TurnEnd::Replied | TurnEnd::Superseded) {
            let now = self.step.now;
            self.set_hold_for(now, turn, Hold::Denied { turn });
        }
    }

    /// The close: a parked turn's words are kept with no reply, and every gated turn ends.
    pub(crate) fn gate_close(&mut self, now: Ts) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        let parked = gate.parked.take();
        if let Some(ParkedTurn { info, edits }) = parked {
            self.deny_turn(now, info, edits, TurnEnd::Abandoned);
        }
        let turns: Vec<TurnId> =
            self.activity.gate.iter().flat_map(|gate| gate.turns.iter().copied()).collect();
        for turn in turns {
            self.gate_turn_ended(turn, TurnEnd::Abandoned);
        }
    }

    /// A correction remade the latest gated turn's unheard reply: the replacement carries the
    /// same decision. `None` when the gate does not allow a remake for `turn`.
    pub(crate) fn gate_remake(&self, turn: TurnId) -> Option<Option<ReplyDecision>> {
        let Some(gate) = self.activity.gate.as_ref() else { return Some(None) };
        let reply = gate.reply.as_ref().filter(|reply| reply.turn == turn)?;
        let held_or_open = match gate.hold {
            Hold::Silence { turn: held, .. } => held == turn,
            Hold::Open => true,
            Hold::Entry | Hold::Undecided { .. } | Hold::Denied { .. } => false,
        };
        (gate.latest_turn() == Some(turn) && held_or_open).then(|| Some(reply.decision.clone()))
    }

    pub(crate) fn gate_remade(&mut self, turn: TurnId, speech: SpeechId) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        if let Some(reply) = gate.reply.as_mut().filter(|reply| reply.turn == turn) {
            reply.speech = speech;
            self.observe_amd(AmdObservation::ReplyCreated { turn, speech });
        }
    }

    /// The reply's tools under a decision that allows DTMF: the advertised ones plus AMD's
    /// built-in `send_dtmf_events`, unless the agent already has a tool with that id. `None`
    /// keeps the advertised tools.
    pub(crate) fn gate_toolset(
        &self,
        decision: &ReplyDecision,
    ) -> Option<std::sync::Arc<crate::tool_executor::Toolset>> {
        if !decision.dtmf {
            return None;
        }
        let advertised = self.activity.advertised_toolset(&self.session.tools);
        if advertised.specs.iter().any(|spec| spec.name == crate::amd::SEND_DTMF_TOOL) {
            return None;
        }
        let mut toolset = (*advertised).clone();
        toolset.specs.push(crate::amd::send_dtmf_tool());
        Some(std::sync::Arc::new(toolset))
    }

    /// The gated reply `speech` is the gate's: the one a remake may replace.
    pub(crate) fn gate_reply_speech(&self, turn: TurnId) -> Option<SpeechId> {
        let gate = self.activity.gate.as_ref()?;
        gate.reply.as_ref().filter(|reply| reply.turn == turn).map(|reply| reply.speech)
    }
}

// The hold and its silence.
impl VoiceDriver {
    /// Replace the hold, reporting silence entry and exit to AMD and the change to the journal.
    fn set_hold(&mut self, now: Ts, hold: Hold) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        let old = std::mem::replace(&mut gate.hold, hold);
        if old == hold {
            return;
        }
        self.record_trace(SpeechStep::AuthorizationHeld(hold));
        let silence = |h: Hold| match h {
            Hold::Silence { turn, .. } => Some(turn),
            _ => None,
        };
        let (was, is) = (silence(old), silence(hold));
        if was != is {
            if let Some(turn) = was {
                self.clear_deadline_if_set(Timer::AmdSilence);
                self.observe_amd(AmdObservation::SilenceHold { turn, held: false, at: now });
            }
            if let Some(turn) = is {
                self.observe_amd(AmdObservation::SilenceHold { turn, held: true, at: now });
            }
        }
    }

    /// Only the latest turn moves the hold: an older turn's late end leaves it alone.
    fn set_hold_for(&mut self, now: Ts, turn: TurnId, hold: Hold) {
        let latest = self.activity.gate.as_ref().and_then(Gate::latest_turn);
        let active = self.activity.gate.as_ref().is_some_and(|gate| gate.active);
        if latest == Some(turn) && active {
            self.set_hold(now, hold);
        }
    }

    /// Arm the silence deadline from the latest boundary; disarm it while the participant
    /// speaks. A silence already long enough opens the hold now.
    fn arm_silence(&mut self, now: Ts) {
        let speaking = self.session.states.user_speaking();
        let Some(gate) = self.activity.gate.as_ref() else { return };
        let Hold::Silence { turn, needed, .. } = gate.hold else { return };
        let deadline = (!speaking).then(|| gate.boundary.unwrap_or(now).saturating_add(needed));
        match deadline {
            Some(at) if at <= now => self.set_hold(now, Hold::Open),
            Some(at) => {
                self.set_hold(now, Hold::Silence { turn, needed, deadline: Some(at) });
                self.set_deadline(Timer::AmdSilence, at);
            }
            None => {
                self.set_hold(now, Hold::Silence { turn, needed, deadline: None });
                self.clear_deadline_if_set(Timer::AmdSilence);
            }
        }
    }

    /// `Timer::AmdSilence`: opens the hold only for the deadline armed, while the turn is still
    /// the latest and the participant silent.
    pub(crate) fn amd_silence_elapsed(&mut self, now: Ts) {
        let speaking = self.session.states.user_speaking();
        let Some(gate) = self.activity.gate.as_ref() else { return };
        let due = matches!(gate.hold, Hold::Silence { turn, deadline: Some(at), .. }
            if at <= now && gate.latest_turn() == Some(turn));
        if due && !speaking {
            self.set_hold(now, Hold::Open);
        }
    }
}

// Observations.
impl VoiceDriver {
    fn observe_amd(&mut self, observation: AmdObservation) {
        if let Some(run) = self.activity.gate.as_ref().map(|gate| gate.run) {
            self.record_effect(VoiceEffect::Amd { run, observation });
        }
    }

    pub(crate) fn observe_speech(&mut self, speech: SpeechId, event: SpeechObservation) {
        if self.activity.gate.is_none() {
            return;
        }
        let at = self.step.now;
        if let (SpeechObservation::PlayoutAdvanced, Some(gate)) = (&event, &mut self.activity.gate)
            && let Some(opening) = gate.opening.as_mut().filter(|o| o.speech == speech)
        {
            opening.played = true;
        }
        if let (SpeechObservation::Done { .. }, Some(gate)) = (&event, &mut self.activity.gate) {
            gate.opening.take_if(|o| o.speech == speech);
        }
        self.observe_amd(AmdObservation::Speech { speech, event, at });
    }

    pub(crate) fn observe_reply_item(&mut self, speech: SpeechId, item: &ChatItem) {
        if self.activity.gate.is_none() {
            return;
        }
        let ChatItem::Message(message) = item else { return };
        if message.role != lk_agents_abi::Role::Assistant {
            return;
        }
        let event = SpeechObservation::ItemAdded {
            text: message.raw_text_content(),
            interrupted: message.interrupted,
        };
        self.observe_speech(speech, event);
    }

    /// Participant speech boundaries, finals and executed tools, from the published events the
    /// user state follows. Boundaries move the silence hold; a start before the opening plays
    /// cancels it.
    pub(crate) fn gate_session_event(&mut self, now: Ts, ev: &SessionEvent) {
        let Some(gate) = self.activity.gate.as_mut() else { return };
        // Before activation no participant speech exists for either side.
        let active = gate.active;
        let observation = match ev {
            SessionEvent::StartOfSpeech(e) if active => {
                gate.boundary = Some(e.at);
                gate.boundary_since_commit = true;
                let opening = gate.opening.take_if(|o| !o.played).map(|o| o.speech);
                if let Some(speech) = opening {
                    self.record_trace(GateStep::OpeningCancelled { speech });
                    self.force_interrupt_speech(now, speech, InterruptCause::Superseded);
                }
                self.arm_silence(now);
                UserSpeechObservation::Started { at: e.at }
            }
            SessionEvent::EndOfSpeech(e) if active => {
                gate.boundary = Some(e.at);
                gate.boundary_since_commit = true;
                self.arm_silence(now);
                UserSpeechObservation::Ended { at: e.at }
            }
            SessionEvent::UserInputTranscribed(t) if active && t.is_final => {
                let text = t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default();
                if text.trim().is_empty() {
                    return;
                }
                UserSpeechObservation::Final { text, at: now }
            }
            SessionEvent::FunctionToolsExecuted(e) => {
                let calls = e
                    .function_calls
                    .iter()
                    .map(|call| ToolCallRecord {
                        call: call.clone(),
                        output: (e.function_call_outputs.iter())
                            .find(|output| output.call_id == call.call_id)
                            .cloned(),
                    })
                    .collect();
                return self.observe_amd(AmdObservation::ToolsExecuted { calls });
            }
            _ => return,
        };
        self.observe_amd(AmdObservation::UserSpeech(observation));
    }

    /// At the end of each step: `AgentBusy` when it changed (and once at attach).
    pub(crate) fn report_agent_busy(&mut self, now: Ts) {
        let Some(gate) = self.activity.gate.as_ref() else { return };
        let previous = gate.busy;
        let a = &self.activity;
        let s = &a.speech_scheduler;
        // Python's `_is_agent_busy`: a speech other than the preemptive generation, a paused
        // speech or a pending false interruption, `speaking`/`thinking`, or a turn not yet
        // replied to (in its hook, behind one, or parked).
        let busy = s.speech_ids().any(|id| s.status(id) != Some(Status::Preemptive))
            || a.paused_speech.paused_handle().is_some()
            || self.deadline(Timer::FalseInterruption).is_some()
            || matches!(
                self.session.states.agent_state(),
                AgentState::Speaking | AgentState::Thinking
            )
            || a.pending_turn.is_some()
            || !a.waiting_turns.is_empty()
            || gate.parked.is_some();
        if previous == Some(busy) {
            return;
        }
        if let Some(gate) = self.activity.gate.as_mut() {
            gate.busy = Some(busy);
        }
        self.observe_amd(AmdObservation::AgentBusy { busy, at: now });
    }
}

#[cfg(test)]
#[path = "amd_gate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "amd_gate_rule_tests.rs"]
mod rule_tests;

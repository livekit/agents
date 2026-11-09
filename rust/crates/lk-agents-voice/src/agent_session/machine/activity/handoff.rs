use super::Cx;
use crate::agent::TurnEdit;
use crate::agent_session::machine::activity::drain::DrainEvent;
use crate::agent_session::machine::activity::hooks::Exit;
use crate::agent_session::machine::activity::scheduling::PauseCause;
use crate::agent_session::machine::coordination::{PauseResult, VoiceCallback};
use crate::agent_session::{
    AgentUpdate, Answer, HandoffRefused, HookId, SpeechCommand, VoiceDriver,
};
use crate::audio_recognition::EndOfTurnInfo;
use crate::trace::HandoffStep;
use lk_agents_abi::Ts;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandoffId(pub u64);

/// The switches asked for and the one in progress, and the user turn held while one runs.
#[derive(Default)]
pub(crate) struct Handoff {
    next: u64,
    in_progress: Option<InProgress>,
    /// Switches land in the order they were asked for.
    queue: VecDeque<(HandoffId, Box<AgentUpdate>, Option<crate::events::RunId>)>,
    /// A user turn decided while a switch runs, delivered to the arriving agent once.
    pub(crate) held_user_turn: Option<HeldUserTurn>,
    /// The host's `say()` / `generate_reply()` asked while the leaving agent drains, in order.
    /// Replayed on the agent the switch leaves driving, or refused by the close.
    pub(crate) held_speech: Vec<SpeechCommand>,
}

struct InProgress {
    id: HandoffId,
    update: Box<AgentUpdate>,
    phase: Phase,
    run: Option<crate::events::RunId>,
}

enum Phase {
    /// The leaving agent may still speak its goodbye.
    Exiting { hook: HookId },
    /// Scheduling paused: the swap lands once every speech has finished.
    Draining,
}

/// A turn kept for the arriving agent. `edits: None`: the arriving agent runs its own
/// `on_user_turn_completed`; `Some`: it already answered, so the turn is decided exactly once.
pub(crate) struct HeldUserTurn {
    pub(crate) info: EndOfTurnInfo,
    pub(crate) edits: Option<Vec<TurnEdit>>,
}

pub(crate) enum HandoffEvent {
    /// `update_agent()`, by the host or a tool's handoff.
    Asked {
        update: Box<AgentUpdate>,
        run: Option<crate::events::RunId>,
    },
    Paused(PauseResult),
    /// The swap landed: the next queued switch begins.
    Next,
    /// The close: the switch in progress and every queued one refused, and a held turn dropped,
    /// since no agent arrives.
    Closing,
}

impl Handoff {
    pub(crate) fn arriving(&self) -> Option<&str> {
        self.in_progress.as_ref().map(|handoff| handoff.update.agent.as_str())
    }

    pub(crate) fn has_run_work(&self, run: crate::events::RunId) -> bool {
        self.in_progress.as_ref().is_some_and(|handoff| handoff.run == Some(run))
            || self.queue.iter().any(|(_, _, member)| *member == Some(run))
    }

    #[must_use]
    pub(super) fn handle(&mut self, ev: HandoffEvent, cx: Cx<'_>) -> Vec<VoiceCallback> {
        let mut callbacks = Vec::new();
        self.handle_callbacks(ev, cx, &mut callbacks);
        callbacks
    }

    fn handle_callbacks(&mut self, ev: HandoffEvent, mut cx: Cx<'_>, out: &mut Vec<VoiceCallback>) {
        match ev {
            HandoffEvent::Asked { update, run } => self.asked(update, run, &mut cx, out),
            HandoffEvent::Paused(result) => {
                if !matches!(self.in_progress, Some(InProgress { phase: Phase::Draining, .. })) {
                    return;
                }
                if result == PauseResult::Settled {
                    return out.push(VoiceCallback::ApplyHandoff);
                }
                // Refused: the agent that blocked the held turn takes it, so it is kept exactly
                // once, and a drain that joined the switch runs on it. Under a close it drops.
                let Some((id, update)) = self.take_in_progress() else { return };
                out.push(VoiceCallback::ReplayHeldSpeech);
                if cx.closing {
                    return refuse(Some(id), update, HandoffRefused::Closing, out);
                }
                refuse(Some(id), update, HandoffRefused::DrainRefused, out);
                out.push(VoiceCallback::DeliverHeldTurn);
                out.push(VoiceCallback::AgentStays);
            }
            HandoffEvent::Next => self.start_next(&mut cx, out),
            HandoffEvent::Closing => {
                if let Some((id, update)) = self.take_in_progress() {
                    refuse(Some(id), update, HandoffRefused::Closing, out);
                }
                out.extend(self.refuse_queued());
                self.held_user_turn = None;
                out.push(VoiceCallback::ReplayHeldSpeech);
            }
        }
    }

    /// The common hook owner finished the exit this handoff waits on.
    pub(super) fn exit_hook_finished(&mut self, hook: HookId) -> Option<VoiceCallback> {
        if let Some(InProgress { phase: phase @ Phase::Exiting { .. }, .. }) = &mut self.in_progress
            && matches!(phase, Phase::Exiting { hook: h } if *h == hook)
        {
            *phase = Phase::Draining;
            return Some(VoiceCallback::PauseScheduling(PauseCause::Handoff));
        }
        None
    }

    pub(crate) fn switching(&self) -> bool {
        self.in_progress.is_some()
    }

    /// The leaving agent drains: its scheduling is paused until the swap.
    pub(crate) fn draining(&self) -> bool {
        matches!(self.in_progress, Some(InProgress { phase: Phase::Draining, .. }))
    }

    pub(crate) fn in_progress(&self) -> Option<&AgentUpdate> {
        self.in_progress.as_ref().map(|h| &*h.update)
    }

    /// Every switch it holds: the one in progress, then the queued ones.
    pub(crate) fn updates(&self) -> impl Iterator<Item = &AgentUpdate> {
        let queued = self.queue.iter().map(|(_, update, _)| &**update);
        self.in_progress().into_iter().chain(queued)
    }

    pub(crate) fn take_in_progress(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
        self.in_progress.take().map(|InProgress { id, update, .. }| (id, update))
    }

    #[must_use]
    pub(crate) fn refuse_queued(&mut self) -> Vec<VoiceCallback> {
        let mut callbacks = Vec::new();
        while let Some((id, update, _)) = self.queue.pop_front() {
            refuse(Some(id), update, HandoffRefused::Closing, &mut callbacks);
        }
        callbacks
    }

    fn mint(&mut self) -> HandoffId {
        self.next += 1;
        HandoffId(self.next)
    }

    /// Refused when closing, queued behind a switch in progress, begun otherwise.
    fn asked(
        &mut self,
        update: Box<AgentUpdate>,
        run: Option<crate::events::RunId>,
        cx: &mut Cx<'_>,
        out: &mut Vec<VoiceCallback>,
    ) {
        let id = self.mint();
        let queued = self.switching() || !self.queue.is_empty();
        if cx.closing {
            return refuse(Some(id), update, HandoffRefused::Closing, out);
        }
        if queued {
            self.queue.push_back((id, update, run));
            return;
        }
        self.start(id, update, run, cx, out);
    }

    fn start_next(&mut self, cx: &mut Cx<'_>, out: &mut Vec<VoiceCallback>) {
        if self.switching() {
            return;
        }
        if let Some((id, update, run)) = self.queue.pop_front() {
            self.start(id, update, run, cx, out);
        }
    }

    /// The first phase: the leaving agent's `on_exit`, or straight to the drain when it already
    /// ran.
    fn start(
        &mut self,
        id: HandoffId,
        update: Box<AgentUpdate>,
        run: Option<crate::events::RunId>,
        cx: &mut Cx<'_>,
        out: &mut Vec<VoiceCallback>,
    ) {
        let exit = cx.leave.claim();
        let phase = match exit.hook() {
            Some(hook) => Phase::Exiting { hook },
            None => Phase::Draining,
        };
        self.in_progress = Some(InProgress { id, update, phase, run });
        if exit == Exit::Done {
            out.push(VoiceCallback::PauseScheduling(PauseCause::Handoff));
        }
    }
}

/// `id` is the handoff's when it was begun (`None` for a command refused before it was).
fn refuse(
    id: Option<HandoffId>,
    update: Box<AgentUpdate>,
    cause: HandoffRefused,
    out: &mut Vec<VoiceCallback>,
) {
    out.push(VoiceCallback::RefuseHandoff { id, update, cause });
}

impl VoiceDriver {
    pub(crate) fn refuse_handoff(
        &mut self,
        _id: Option<HandoffId>,
        mut update: Box<AgentUpdate>,
        cause: HandoffRefused,
    ) {
        self.answer(update.call.take(), Answer::Handoff(Err(cause)));
    }

    /// The swap: the closing re-check, the agent installed with its two records, the caller
    /// answered, `on_enter` issued, the held turn delivered, the next queued switch begun.
    pub(crate) fn perform_handoff(&mut self, now: Ts) {
        let Some((_id, mut update)) = self.activity.handoff.take_in_progress() else {
            return;
        };
        if self.session.closing.is_some() {
            self.replay_held_speech(now);
            self.refuse_handoff(Some(_id), update, HandoffRefused::Closing);
            for callback in self.activity.handoff.refuse_queued() {
                self.defer_callback(callback);
            }
            return;
        }
        let call = update.call.take();
        self.update_agent(now, *update);
        self.drain_step(DrainEvent::Leaving);
        self.activity.on_exit.swapped();
        self.answer(call, Answer::Handoff(Ok(())));
        // Before `on_enter`, whose speech comes after what was asked during the switch.
        self.replay_held_speech(now);
        self.issue_on_enter(now);
        self.defer_callback(VoiceCallback::DeliverHeldTurn);
        self.defer_callback(VoiceCallback::StartNextHandoff);
    }

    /// The host's speech held over the drain, asked again of the agent now driving: scheduled, or
    /// refused as any speech is under a close.
    pub(crate) fn replay_held_speech(&mut self, now: Ts) {
        for cmd in std::mem::take(&mut self.activity.handoff.held_speech) {
            self.on_speech(now, cmd);
        }
    }

    pub(crate) fn deliver_held_turn(&mut self, now: Ts) {
        if self.activity.in_window() {
            return;
        }
        if let Some(HeldUserTurn { info, edits }) = self.activity.handoff.held_user_turn.take() {
            match edits {
                None => self.on_end_of_turn(now, info),
                Some(edits) => self.finish_turn(now, info, edits),
            }
        }
    }

    /// The close: pending handoffs refused; a running callback is abandoned, not cancelled.
    pub(crate) fn abandon_handoffs(&mut self) {
        // A drain already pausing is answered by the pause's refusal instead.
        self.drain_step(DrainEvent::Closing);
        self.dtmf_step(crate::workflows::machine::dtmf::DtmfTaskEvent::Stop);
        self.handoff_step(HandoffEvent::Closing);
        self.activity.on_exit.abandon();
        self.activity.tools.pending.clear();
        let tools = &mut self.activity.tools;
        tools.reply.take_if(|r| matches!(r, crate::tool_executor::ToolReply::Rendering { .. }));
        self.abandon_hooks();
    }

    /// A fresh park consumes the turn in the machine, so the next countdown cannot commit its words
    /// twice. A second turn in the window is folded in, so the arriving agent decides one turn.
    pub(crate) fn hold_user_turn(
        &mut self,
        now: Ts,
        mut info: EndOfTurnInfo,
        edits: Option<Vec<TurnEdit>>,
    ) -> bool {
        if !self.activity.in_window() {
            return false;
        }
        if edits.is_none() {
            let taken = crate::audio_recognition::UserTurnCommitDecision {
                verdict: crate::audio_recognition::UserTurnCommitVerdict::Reply,
                committed: true,
                cancel_false_interruption: false,
            };
            let eff = self.activity.audio_recognition.on_commit_verdict(now, &info, &taken);
            self.apply_recognition_effects(eff);
        }
        let edits = match self.activity.handoff.held_user_turn.take() {
            Some(HeldUserTurn { info: earlier, edits: earlier_edits }) => {
                // The merged park is the undecided turn; the fresh park's id stays.
                if edits.is_some() && earlier_edits.is_none() {
                    info.turn_id = earlier.turn_id;
                }
                let mut words = earlier.new_transcript.trim().to_owned();
                if !info.new_transcript.trim().is_empty() {
                    if !words.is_empty() {
                        words.push(' ');
                    }
                    words.push_str(info.new_transcript.trim());
                }
                info.new_transcript = words;
                if info.metrics.started_speaking_at.is_none() {
                    info.metrics.started_speaking_at = earlier.metrics.started_speaking_at;
                }
                // The earlier answer was for the earlier words alone.
                None
            }
            None => edits,
        };
        self.activity.handoff.held_user_turn = Some(HeldUserTurn { info, edits });
        true
    }

    /// Reconfigure the running session for a new agent; unchanged streams keep running.
    pub(crate) fn update_agent(&mut self, now: Ts, update: AgentUpdate) {
        let (stt_reopened, vad_reopened) = (update.stt.is_given(), update.vad.is_given());
        let AgentUpdate {
            agent,
            instructions,
            dynamic_instructions,
            toolset,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            options,
            expressive,
            output_nodes,
            ..
        } = update;
        self.session.expressive.agent = expressive;
        if let Some(reason) = self.activity.refuse_turn_limit() {
            self.record_trace(crate::trace::TurnStep::TurnLimitSkipped(reason));
        }
        // A reply in flight keeps the instructions and tools it snapshotted.
        let count = toolset.specs.len();
        let tool_names: std::collections::BTreeSet<String> =
            toolset.specs.iter().map(|t| t.name.clone()).collect();
        let previous_agent = self.activity.set_agent(
            agent,
            instructions,
            dynamic_instructions,
            output_nodes,
            toolset,
        );
        self.observe_experts();
        let effects = self.session.apply_providers(llm, tts);
        for effect in effects {
            self.record_effect(effect);
        }

        let replaced_ctx = chat_ctx.is_some();
        let agent = self.activity.agent.clone();
        match chat_ctx {
            Some(items) => {
                self.session.chat_ctx.update_agent_chat_ctx(items, &agent);
            }
            None => {
                let from = self.session.chat_ctx.agent().to_owned();
                let items = self.session.chat_ctx.hand_to(&agent, &tool_names);
                if from != agent {
                    self.publish_event(crate::events::SessionEvent::ContextCarried {
                        from,
                        to: agent.clone(),
                        items,
                    });
                }
            }
        }

        self.record_handoff((!previous_agent.is_empty()).then_some(previous_agent));

        let legs = self.reopen_legs(stt, vad);

        // An STT that refused to open is still the old one.
        self.session.config.legs.stt = legs.stt;
        self.session.config.legs.vad = legs.vad;
        let diff = self.session.config.set_agent(options);
        self.apply_config(now, crate::agent_session::options::ConfigLayer::Agent, diff);

        // Ungated by the `mutable_*` capabilities: a refusal on the wire is visible, a skipped
        // push is not.
        let ctx = replaced_ctx.then(|| self.session.chat_ctx.for_realtime());
        self.push_activity(ctx);

        self.resume_scheduling(now);
        let legs = crate::trace::ProviderPresence {
            stt: legs.stt,
            vad: legs.vad,
            llm: self.session.llm.is_some(),
            tts: self.session.tts.is_some(),
            realtime: self.activity.realtime().is_some(),
        };
        self.record_trace(HandoffStep::AgentUpdated {
            agent: self.activity.agent.clone(),
            tools: count,
            chat_ctx: replaced_ctx,
            stt_reopened,
            vad_reopened,
            legs,
        });
    }
}

impl super::AgentActivity {
    /// The transition window: from a handoff request through the swap.
    pub(crate) fn in_window(&self) -> bool {
        self.handoff.switching()
    }
}

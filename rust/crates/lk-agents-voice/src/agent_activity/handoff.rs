//! `update_agent()`: switching agents, from the leaving agent's `on_exit` through the drain to the
//! swap, in the order the switches were asked for, with the user turn held across it.

use crate::agent::TurnEdit;
use crate::agent_activity::drain::DrainEvent;
use crate::agent_activity::on_exit::Exit;
use crate::agent_activity::scheduling::PauseCause;
use crate::agent_session::effects::{Out, Timer};
use crate::agent_session::signals::{Cx, PauseResult, Signal};
use crate::agent_session::{
    AgentUpdate, Answer, Driver, HandoffRefused, HookId, HookKind, SpeechCommand,
};
use crate::audio_recognition::EndOfTurnInfo;
use crate::run_result::Owned;
use crate::trace::{HandoffStep, HookEnd};
use lk_agents_abi::Ts;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandoffId(pub u64);

/// The switches asked for and the one in progress, and the user turn held while one runs.
#[derive(Default)]
pub(crate) struct Handoff {
    next: u64,
    in_progress: Option<InProgress>,
    /// The order switches were asked for, which is the order they land in (each `update_agent`
    /// chains behind the one before).
    queue: VecDeque<(HandoffId, Box<AgentUpdate>)>,
    /// A user turn decided while a switch runs, delivered to the arriving agent once.
    pub(crate) held_user_turn: Option<HeldUserTurn>,
    /// The host's `say()` / `generate_reply()` asked while the leaving agent drains, in order: the
    /// reference schedules them on `_next_activity`. Replayed on the agent the switch leaves
    /// driving, or into the close's refusal.
    pub(crate) held_speech: Vec<SpeechCommand>,
}

struct InProgress {
    id: HandoffId,
    update: Box<AgentUpdate>,
    phase: Phase,
}

enum Phase {
    /// The leaving agent may still speak its goodbye.
    Exiting { hook: HookId },
    /// `PauseScheduling { cause: Handoff }` asked: the swap lands once every speech has finished.
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
    Asked(Box<AgentUpdate>),
    OnExitEnded(HookId),
    Paused(PauseResult),
    /// The swap landed: the next queued switch begins.
    Next,
    /// The close: the switch in progress and every queued one refused, and a held turn dropped,
    /// since no agent arrives.
    Closing,
}

impl Handoff {
    pub(crate) fn handle(&mut self, ev: HandoffEvent, mut cx: Cx<'_>, out: &mut Out) {
        match ev {
            HandoffEvent::Asked(update) => self.asked(update, &mut cx, out),
            HandoffEvent::OnExitEnded(hook) => {
                if let Some(InProgress { phase: phase @ Phase::Exiting { .. }, .. }) =
                    &mut self.in_progress
                    && matches!(phase, Phase::Exiting { hook: h } if *h == hook)
                {
                    *phase = Phase::Draining;
                    out.signal(Signal::PauseScheduling(PauseCause::Handoff));
                }
            }
            HandoffEvent::Paused(result) => {
                if !matches!(self.in_progress, Some(InProgress { phase: Phase::Draining, .. })) {
                    return;
                }
                if result == PauseResult::Settled {
                    return out.signal(Signal::Swap);
                }
                // Refused: the agent that blocked the held turn takes it, so it is kept exactly
                // once, and a drain that joined the switch runs on it. Under a close it drops.
                let Some((id, update)) = self.take_in_progress() else { return };
                out.signal(Signal::ReplayHeldSpeech);
                if cx.closing {
                    return refuse(Some(id), update, HandoffRefused::Closing, out);
                }
                refuse(Some(id), update, HandoffRefused::DrainRefused, out);
                out.signal(Signal::DeliverHeldTurn);
                out.signal(Signal::AgentStays);
            }
            HandoffEvent::Next => self.start_next(&mut cx, out),
            HandoffEvent::Closing => {
                if let Some((id, update)) = self.take_in_progress() {
                    refuse(Some(id), update, HandoffRefused::Closing, out);
                }
                self.refuse_queued(out);
                self.held_user_turn = None;
                out.signal(Signal::ReplayHeldSpeech);
            }
        }
    }

    pub(crate) fn switching(&self) -> bool {
        self.in_progress.is_some()
    }

    /// The leaving agent drains: its scheduling is paused until the swap.
    pub(crate) fn draining(&self) -> bool {
        matches!(self.in_progress, Some(InProgress { phase: Phase::Draining, .. }))
    }

    /// The switch in progress.
    pub(crate) fn in_progress(&self) -> Option<&AgentUpdate> {
        self.in_progress.as_ref().map(|h| &*h.update)
    }

    /// Every switch it holds: the one in progress, then the queued ones.
    pub(crate) fn updates(&self) -> impl Iterator<Item = &AgentUpdate> {
        let queued = self.queue.iter().map(|(_, update)| &**update);
        self.in_progress().into_iter().chain(queued)
    }

    pub(crate) fn take_in_progress(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
        self.in_progress.take().map(|InProgress { id, update, phase: _ }| (id, update))
    }

    /// Every queued switch refused as closing.
    pub(crate) fn refuse_queued(&mut self, out: &mut Out) {
        while let Some((id, update)) = self.queue.pop_front() {
            refuse(Some(id), update, HandoffRefused::Closing, out);
        }
    }

    fn mint(&mut self) -> HandoffId {
        self.next += 1;
        HandoffId(self.next)
    }

    /// Refused when closing, queued behind a switch in progress, begun otherwise.
    fn asked(&mut self, update: Box<AgentUpdate>, cx: &mut Cx<'_>, out: &mut Out) {
        let id = self.mint();
        let queued = self.switching() || !self.queue.is_empty();
        if cx.closing {
            return refuse(Some(id), update, HandoffRefused::Closing, out);
        }
        // An open run owns every switch asked for while it is open.
        out.signal(Signal::RunOwn(Owned::Handoff(id)));
        if queued {
            self.queue.push_back((id, update));
            return;
        }
        self.start(id, update, cx, out);
    }

    /// The next queued switch, unless one is in progress.
    fn start_next(&mut self, cx: &mut Cx<'_>, out: &mut Out) {
        if self.switching() {
            return;
        }
        if let Some((id, update)) = self.queue.pop_front() {
            self.start(id, update, cx, out);
        }
    }

    /// The first phase: the leaving agent's `on_exit`, or straight to the drain when it already
    /// ran.
    fn start(&mut self, id: HandoffId, update: Box<AgentUpdate>, cx: &mut Cx<'_>, out: &mut Out) {
        let exit = cx.leave.claim();
        let phase = match exit.hook() {
            Some(hook) => Phase::Exiting { hook },
            None => Phase::Draining,
        };
        self.in_progress = Some(InProgress { id, update, phase });
        if exit == Exit::Done {
            out.signal(Signal::PauseScheduling(PauseCause::Handoff));
        }
    }
}

/// `id` is the handoff's when it was begun (`None` for a command refused before it was).
fn refuse(id: Option<HandoffId>, update: Box<AgentUpdate>, cause: HandoffRefused, out: &mut Out) {
    out.signal(Signal::RefuseHandoff { id, update, cause });
}

impl Driver {
    /// The caller answered and the run lets go.
    pub(crate) fn refuse_handoff(
        &mut self,
        id: Option<HandoffId>,
        mut update: Box<AgentUpdate>,
        cause: HandoffRefused,
        out: &mut Out,
    ) {
        if let Some(id) = id {
            self.run_release(Owned::Handoff(id), out);
        }
        out.answer(update.call.take(), Answer::Handoff(Err(cause)));
    }

    /// The swap: the closing re-check, the agent installed with its two records, the caller
    /// answered, `on_enter` issued, the held turn delivered, the next queued switch begun.
    pub(crate) fn perform_handoff(&mut self, now: Ts, out: &mut Out) {
        let Some((id, mut update)) = self.activity.handoff.take_in_progress() else {
            return;
        };
        if self.session.closing.is_some() {
            self.replay_held_speech(now, out);
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
            self.activity.handoff.refuse_queued(out);
            return;
        }
        let call = update.call.take();
        self.update_agent(now, *update, out);
        self.drain_step(DrainEvent::Leaving, out);
        self.activity.on_exit.swapped();
        out.answer(call, Answer::Handoff(Ok(())));
        // Before `on_enter`, whose speech comes after what was asked during the switch.
        self.replay_held_speech(now, out);
        self.issue_on_enter(now, out);
        // The run's `on_enter`, owned above, keeps the run open.
        self.run_release(Owned::Handoff(id), out);
        out.signal(Signal::DeliverHeldTurn);
        out.signal(Signal::NextHandoff);
    }

    /// The transition window: from a tool's handoff request (`_new_turns_blocked` is set in
    /// `update_agent`) through the swap.
    pub(crate) fn in_window(&self) -> bool {
        self.activity.handoff.switching()
    }

    /// The host's speech held over the drain, asked again of the agent now driving: scheduled, or
    /// refused as any speech is under a close.
    pub(crate) fn replay_held_speech(&mut self, now: Ts, out: &mut Out) {
        for cmd in std::mem::take(&mut self.activity.handoff.held_speech) {
            self.on_speech(now, cmd, out);
        }
    }

    pub(crate) fn deliver_held_turn(&mut self, now: Ts, out: &mut Out) {
        if self.in_window() {
            return;
        }
        if let Some(HeldUserTurn { info, edits }) = self.activity.handoff.held_user_turn.take() {
            match edits {
                None => self.on_end_of_turn(now, info, out),
                Some(edits) => self.finish_turn(now, info, edits, out),
            }
        }
    }

    /// The close: pending handoffs refused; a running callback is abandoned, not cancelled.
    pub(crate) fn abandon_handoffs(&mut self, out: &mut Out) {
        // A drain still waiting on its `on_exit` is refused as closing; one already pausing is
        // answered by the pause's refusal.
        self.drain_step(DrainEvent::Closing, out);
        // A DTMF collection started during the close ends with it: an owed answer is `Stopped`.
        self.dtmf_step(crate::agent_activity::dtmf::DtmfTaskEvent::Stop, out);
        self.handoff_step(HandoffEvent::Closing, out);
        self.activity.on_exit.abandon();
        self.activity.tools.abandon_replies();
        self.abandon_hooks(out);
    }

    /// Every callback still waited on is left to the host, its wait abandoned.
    pub(crate) fn abandon_hooks(&mut self, out: &mut Out) {
        // All abandoned before any is let go: the run reads what is still pending.
        let pending = self.activity.hooks.take_all();
        for &(id, kind) in &pending {
            self.disarm(Timer::Hook(id), out);
            let how = HookEnd::Abandoned;
            out.trace(HandoffStep::HookFinished { hook: id, kind, how });
            self.run_release(Owned::Hook(id), out);
        }
        // A step waiting on its `function_tools_executed` handlers goes on as issued: its results
        // are committed and its reply decided.
        for (id, kind) in pending {
            if kind == HookKind::FunctionToolsExecuted {
                self.on_tools_executed_handled(id, None, out);
            }
        }
    }

    /// A fresh park consumes the turn in the machine, so the next countdown cannot commit its words
    /// twice. A second turn in the window is folded in, so the arriving agent decides one turn.
    pub(crate) fn hold_user_turn(
        &mut self,
        now: Ts,
        mut info: EndOfTurnInfo,
        edits: Option<Vec<TurnEdit>>,
        out: &mut Out,
    ) -> bool {
        if !self.in_window() {
            return false;
        }
        if edits.is_none() {
            let taken = crate::audio_recognition::CommitDecision {
                verdict: crate::audio_recognition::CommitVerdict::Reply,
                committed: true,
                cancel_false_interruption: false,
            };
            let eff = self.activity.audio_recognition.on_commit_verdict(now, &info, &taken);
            out.machine().append(eff);
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

    /// Reconfigure the running session for a new agent. Nothing is swapped: the scheduling task,
    /// socket, pump and unchanged streams keep running; only what the next plan reads and the
    /// changed legs move.
    pub(crate) fn update_agent(&mut self, now: Ts, update: AgentUpdate, out: &mut Out) {
        let reopened = update.reopens();
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
            ..
        } = update;
        self.session.expressive.agent = expressive;
        // A reply in flight keeps the instructions and tools it snapshotted.
        let count = toolset.specs.len();
        let tool_names: std::collections::BTreeSet<String> =
            toolset.specs.iter().map(|t| t.name.clone()).collect();
        let previous_agent =
            self.activity.set_agent(agent, instructions, dynamic_instructions, toolset);
        self.session.apply_providers(llm, tts, out);

        // The new agent's view: the context it brought, or the conversation carried forward.
        let replaced_ctx = chat_ctx.is_some();
        let agent = self.activity.agent.clone();
        match chat_ctx {
            Some(items) => {
                self.session.chat_ctx.update_agent_chat_ctx(items, &agent);
            }
            None => {
                // The conversation carried forward; the host is told which items changed agent.
                let from = self.session.chat_ctx.agent().to_owned();
                let items = self.session.chat_ctx.hand_to(&agent, &tool_names);
                if from != agent {
                    out.machine().emit(crate::events::SessionEvent::ContextCarried {
                        from,
                        to: agent.clone(),
                        items,
                    });
                }
            }
        }
        self.publish_activity();

        // The handoff's two records, minted here only: the `AgentHandoff` item
        // (`record_handoff_item`) and the history-only `AgentConfigUpdate` (`_start_session`),
        // stamped on the session clock. Empty without an agent id (test harness): `None`.
        self.record_handoff((!previous_agent.is_empty()).then_some(previous_agent), out);

        let legs = self.reopen_legs(stt, vad, out);

        // The agent layer, replaced whole, over the legs this update left (an STT that refused to
        // open is still the old one). Only the STT and VAD facts are resolved against; the
        // session's providers are the one record of the LLM/TTS.
        self.session.config.legs.stt = legs.stt;
        self.session.config.legs.vad = legs.vad;
        let diff = self.session.config.set_agent(options);
        self.apply_config(now, crate::agent_activity::options::ConfigLayer::Agent, diff, out);

        // A realtime handoff is three commands on the same socket, not a reconnect. Ungated by the
        // `mutable_*` capabilities: a refusal on the wire is visible, a skipped push is not. The
        // chat context only when the agent brought one, through `for_realtime`.
        let ctx = replaced_ctx.then(|| self.session.chat_ctx.for_realtime());
        self.push_activity(ctx, out);

        // A handoff lifts the pause the switch caused; a preemptive generation about the old turn
        // goes too.
        self.resume_scheduling_task(now, out);
        let legs = crate::trace::ProviderPresence {
            stt: legs.stt,
            vad: legs.vad,
            llm: self.session.llm.is_some(),
            tts: self.session.tts.is_some(),
            realtime: self.activity.rt_session.is_some(),
        };
        out.trace(HandoffStep::AgentUpdated {
            tools: count,
            chat_ctx: replaced_ctx,
            reopened,
            legs,
        });
    }
}

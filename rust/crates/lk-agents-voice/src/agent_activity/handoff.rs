//! `update_agent()`: switching agents, from the leaving agent's `on_exit` through the drain to the
//! swap, in the order the switches were asked for, with the user turn held across it.

use crate::agent::TurnEdit;
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_session::effects::Out;
use crate::agent_session::{
    AgentUpdate, Answer, Driver, HandoffRefused, HookCall, HookId, HookKind, NewActivity,
    PreviousActivity,
};
use crate::audio_recognition::EndOfTurnInfo;
use crate::run_result::Owned;
use crate::trace::{HandoffStep, HookEnd};
use lk_agents_abi::Ts;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandoffId(pub u64);

/// The switches asked for and the one in progress, the user turn held while one runs, and the
/// `on_exit` each agent owes (`_on_exit_task`).
#[derive(Default)]
pub(crate) struct Handoff {
    next: u64,
    in_progress: Option<InProgress>,
    /// The order switches were asked for, which is the order they land in (each `update_agent`
    /// chains behind the one before).
    pub(crate) queue: VecDeque<Queued>,
    /// A user turn decided while a switch runs, delivered to the arriving agent once.
    pub(crate) held_user_turn: Option<HeldUserTurn>,
    /// The driving agent's `on_exit`, created once by whichever of `drain()`, a handoff or the
    /// close gets there first.
    pub(crate) exit: FrameExit,
    /// The exits of agents paused under an inline task, innermost last.
    paused_exits: Vec<FrameExit>,
}

struct InProgress {
    id: HandoffId,
    update: Box<AgentUpdate>,
    phase: Phase,
}

enum Phase {
    /// The leaving agent may still speak its goodbye (`None`: it owes no `on_exit`).
    Exiting { hook: Option<HookId> },
    /// `PauseScheduling { cause: Handoff }` in effect; `settled` answers when every speech not
    /// blocked on an inline task has finished, or closes when the pause was refused.
    Draining { settled: tokio::sync::oneshot::Receiver<()> },
}

pub(crate) enum Queued {
    /// Waiting for the switch in progress.
    Ready { id: HandoffId, update: Box<AgentUpdate> },
    /// A tool's handoff holds its place while the host describes the agent, so a host
    /// `update_agent` meanwhile waits behind it; given up when the describe fails or times out.
    Describing(HookId),
}

/// A turn kept for the arriving agent. `edits: None`: the arriving agent runs its own
/// `on_user_turn_completed`; `Some`: it already answered, so the turn is decided exactly once.
pub(crate) struct HeldUserTurn {
    pub(crate) info: EndOfTurnInfo,
    pub(crate) edits: Option<Vec<TurnEdit>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum FrameExit {
    #[default]
    Running,
    Exiting(HookId),
    Exited,
}

impl FrameExit {
    /// The `on_exit` running now.
    pub(crate) fn exiting(self) -> Option<HookId> {
        match self {
            FrameExit::Exiting(hook) => Some(hook),
            FrameExit::Running | FrameExit::Exited => None,
        }
    }
}

/// The `on_exit` an agent owes when it is left: issued now, already running, or none (it ran).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Issue(HookId),
    Wait(HookId),
    Done,
}

impl Exit {
    pub(crate) fn hook(self) -> Option<HookId> {
        match self {
            Exit::Issue(hook) | Exit::Wait(hook) => Some(hook),
            Exit::Done => None,
        }
    }
}

/// Where a switch's drain is.
pub(crate) enum Poll {
    Waiting,
    /// The swap is due now.
    Settled,
    /// The pause was refused: the switch, taken.
    Refused(HandoffId, Box<AgentUpdate>),
}

impl Handoff {
    pub(crate) fn mint(&mut self) -> HandoffId {
        self.next += 1;
        HandoffId(self.next)
    }

    pub(crate) fn switching(&self) -> bool {
        self.in_progress.is_some()
    }

    /// The agent a host (not an inline task's) switch in progress installs.
    pub(crate) fn host_agent(&self) -> Option<&str> {
        (self.in_progress.as_ref())
            .filter(|h| h.update.task.is_none())
            .map(|h| h.update.agent.as_str())
    }

    pub(crate) fn take_in_progress(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
        self.in_progress.take().map(|InProgress { id, update, phase: _ }| (id, update))
    }

    /// The switch begins, waiting on the leaving agent's `on_exit` `hook`.
    fn start(&mut self, id: HandoffId, update: Box<AgentUpdate>, hook: Option<HookId>) {
        self.in_progress = Some(InProgress { id, update, phase: Phase::Exiting { hook } });
    }

    /// The next switch to begin: none while one is in progress or a describe holds the head.
    fn next_ready(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
        if self.in_progress.is_some() {
            return None;
        }
        let Some(Queued::Ready { .. }) = self.queue.front() else { return None };
        match self.queue.pop_front() {
            Some(Queued::Ready { id, update }) => Some((id, update)),
            Some(Queued::Describing(_)) | None => None,
        }
    }

    fn place_of(&self, hook: HookId) -> Option<usize> {
        self.queue.iter().position(|q| matches!(q, Queued::Describing(h) if *h == hook))
    }

    /// Has the switch's drain settled? A refused pause takes the switch with it.
    fn poll(&mut self) -> Poll {
        let Some(InProgress { phase: Phase::Draining { settled }, .. }) = &mut self.in_progress
        else {
            return Poll::Waiting;
        };
        match settled.try_recv() {
            Ok(()) => Poll::Settled,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => Poll::Waiting,
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => match self.take_in_progress()
            {
                Some((id, update)) => Poll::Refused(id, update),
                None => Poll::Waiting,
            },
        }
    }

    /// The `on_exit` `hook` ended: every agent's exit it was is over.
    pub(crate) fn exit_ended(&mut self, hook: HookId) {
        for exit in std::iter::once(&mut self.exit).chain(self.paused_exits.iter_mut()) {
            if *exit == FrameExit::Exiting(hook) {
                *exit = FrameExit::Exited;
            }
        }
    }

    /// The swap: a paused agent keeps its exit beneath the arriving one and gets it back on
    /// resume.
    fn swapped(&mut self, previous: PreviousActivity, next: NewActivity) {
        match (previous, next) {
            (PreviousActivity::Pause, NewActivity::Start) => {
                let paused = std::mem::take(&mut self.exit);
                self.paused_exits.push(paused);
            }
            (PreviousActivity::Close, NewActivity::Start) => self.exit = FrameExit::Running,
            (_, NewActivity::Resume) => self.exit = self.paused_exits.pop().unwrap_or_default(),
        }
    }
}

impl Driver {
    /// The handoff begins when the host answers; its place in the order is now.
    pub(crate) fn request_describe(
        &mut self,
        now: Ts,
        speech: SpeechId,
        agent: lk_agents_abi::HostRef,
        out: &mut Out,
    ) {
        let id = self.activity.hooks.mint();
        let wait = self.hook_wait();
        if self.issue_hook(now, HookCall::DescribeAgent { id, agent, speech }, Some(wait), out) {
            self.activity.handoff.queue.push_back(Queued::Describing(id));
        }
    }

    /// Refused when closing (a resume is not), queued behind a switch in progress or a pending
    /// describe, begun otherwise.
    pub(crate) fn begin_handoff(&mut self, now: Ts, update: Box<AgentUpdate>, out: &mut Out) {
        let handoff = &mut self.activity.handoff;
        let id = handoff.mint();
        // A task's resume completes the current transition before a queued host switch: queued
        // behind it, the task's old parent would overwrite the host's agent.
        let resuming_task = update.task.is_some() && update.next == NewActivity::Resume;
        let queued = handoff.switching() || (!handoff.queue.is_empty() && !resuming_task);
        if self.session.closing.is_some() && update.next == NewActivity::Start {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
            return;
        }
        // An open run owns every switch asked for while it is open.
        self.run_own(Owned::Handoff(id), out);
        if queued {
            self.activity.handoff.queue.push_back(Queued::Ready { id, update });
            return;
        }
        self.start_handoff(now, id, update, out);
    }

    pub(super) fn describe_answered(
        &mut self,
        now: Ts,
        hook: HookId,
        update: Box<AgentUpdate>,
        out: &mut Out,
    ) {
        let Some(place) = self.activity.handoff.place_of(hook) else {
            // No place held (the close emptied the queue): asked for now, and refused as closing.
            self.begin_handoff(now, update, out);
            return;
        };
        let id = self.activity.handoff.mint();
        if self.session.closing.is_some() && update.next == NewActivity::Start {
            self.activity.handoff.queue.remove(place);
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
            return;
        }
        self.run_own(Owned::Handoff(id), out);
        self.activity.handoff.queue[place] = Queued::Ready { id, update };
        self.pump_queue(now, out);
    }

    /// A describe that will not be answered gives up its place: the asking agent takes the parked
    /// turn, and what waited behind goes ahead.
    pub(super) fn drop_describe(&mut self, now: Ts, hook: HookId, out: &mut Out) {
        if let Some(place) = self.activity.handoff.place_of(hook) {
            self.activity.handoff.queue.remove(place);
        }
        self.deliver_parked_turn(now, out);
        self.pump_queue(now, out);
    }

    fn pump_queue(&mut self, now: Ts, out: &mut Out) {
        if let Some((id, update)) = self.activity.handoff.next_ready() {
            self.start_handoff(now, id, update, out);
        }
    }

    /// `id` is the handoff's when it was begun (`None` for a command refused before it was).
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
        self.answer(update.call.take(), Answer::Handoff(Err(cause)));
        // An inline task's entry or exit: the task is told.
        if let Some(task) = update.task {
            let now = self.step.now;
            self.on_task_handoff_refused(now, task, update.next, out);
        }
    }

    /// Every queued switch refused as closing (a held describe just goes).
    pub(crate) fn refuse_queued_handoffs(&mut self, out: &mut Out) {
        while let Some(queued) = self.activity.handoff.queue.pop_front() {
            if let Queued::Ready { id, update } = queued {
                self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
            }
        }
    }

    /// The close: the switch in progress and every queued one refused, and a turn parked for the
    /// arriving agent dropped, since none arrives.
    pub(crate) fn refuse_handoffs(&mut self, out: &mut Out) {
        if let Some((id, update)) = self.activity.handoff.take_in_progress() {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
        }
        self.refuse_queued_handoffs(out);
        self.activity.handoff.held_user_turn = None;
    }

    /// The first phase: `on_exit` on the agent being left (`Close`), or straight to the drain
    /// (`Pause`: an awaited `AgentTask`'s parent runs no `on_exit`).
    fn start_handoff(&mut self, now: Ts, id: HandoffId, update: Box<AgentUpdate>, out: &mut Out) {
        // The parent is captured now that the entry starts (it may have changed while queued).
        if let Some(task) = update.task.clone()
            && update.next == NewActivity::Start
        {
            self.capture_task_parent(task, &update);
        }
        let exit = match update.previous {
            PreviousActivity::Pause => Exit::Done,
            PreviousActivity::Close => self.frame_exit(),
        };
        self.activity.handoff.start(id, update, exit.hook());
        if self.issue_exit(now, exit, out) {
            self.drain_handoff(now, out);
        }
    }

    /// The `on_exit` the driving agent owes. It runs once: minted to issue while the agent runs,
    /// waited for while it exits, none once it ran.
    pub(crate) fn frame_exit(&mut self) -> Exit {
        match self.activity.handoff.exit {
            FrameExit::Running => Exit::Issue(self.activity.hooks.mint()),
            FrameExit::Exiting(hook) if self.activity.hooks.is_pending(hook) => Exit::Wait(hook),
            FrameExit::Exiting(_) | FrameExit::Exited => Exit::Done,
        }
    }

    /// `true` when the exit is over: none owed, or issued with no host.
    pub(crate) fn issue_exit(&mut self, now: Ts, exit: Exit, out: &mut Out) -> bool {
        match exit {
            Exit::Done => true,
            Exit::Wait(_) => false,
            Exit::Issue(hook) => self.issue_on_exit(now, hook, out),
        }
    }

    /// `true` when there is no host and it is over.
    pub(crate) fn issue_on_exit(&mut self, now: Ts, hook: HookId, out: &mut Out) -> bool {
        let call = HookCall::OnExit { id: hook, agent: self.activity.agent.clone() };
        let wait = self.hook_wait();
        !self.issue_hook(now, call, Some(wait), out)
    }

    /// The `on_exit` `hook` ended: the switch waiting on it drains.
    pub(crate) fn handoff_exit_done(&mut self, now: Ts, hook: HookId, out: &mut Out) {
        if let Some(InProgress { phase: Phase::Exiting { hook: Some(h) }, .. }) =
            &self.activity.handoff.in_progress
            && *h == hook
        {
            self.drain_handoff(now, out);
        }
    }

    /// The second phase: the drain, through the scheduling task; the swap lands in the settling
    /// cycle.
    fn drain_handoff(&mut self, now: Ts, out: &mut Out) {
        let Some(in_progress) = &mut self.activity.handoff.in_progress else { return };
        let (done, settled) = tokio::sync::oneshot::channel();
        in_progress.phase = Phase::Draining { settled };
        self.pause_scheduling(now, PauseCause::Handoff, done, out);
    }

    /// `true` means the swap is due now. A refused pause ends the handoff as refused, and the
    /// agent that blocked the parked turn takes it in that cycle, so it is kept exactly once.
    /// Under a close it drops.
    pub(crate) fn handoff_ready(&mut self, out: &mut Out) -> bool {
        match self.activity.handoff.poll() {
            Poll::Settled => true,
            Poll::Waiting => false,
            Poll::Refused(id, update) => {
                let cause = if self.session.closing.is_some() {
                    HandoffRefused::Closing
                } else {
                    HandoffRefused::DrainRefused
                };
                self.refuse_handoff(Some(id), update, cause, out);
                if cause == HandoffRefused::DrainRefused {
                    let now = self.step.now;
                    self.deliver_parked_turn(now, out);
                    // The agent stays: a drain that joined the switch runs on it.
                    self.rebind_joined_drain(now, out);
                }
                false
            }
        }
    }

    /// The swap: the closing re-check, the agent installed with its two records, the caller
    /// answered, `on_enter` issued, the parked turn delivered, the next queued switch begun.
    pub(crate) fn perform_handoff(&mut self, now: Ts, out: &mut Out) {
        let Some((id, mut update)) = self.activity.handoff.take_in_progress() else {
            return;
        };
        if self.session.closing.is_some() && update.next == NewActivity::Start {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, out);
            self.refuse_queued_handoffs(out);
            return;
        }
        let call = update.call.take();
        let next = update.next;
        let previous = update.previous;
        let task = update.task.clone();
        // A drain bound to an agent an inline task pauses runs that agent's `on_exit` first; one
        // on an agent left for good is settled once it is no longer driving.
        if previous == PreviousActivity::Pause {
            self.settle_drain_at_swap(now, previous, out);
        }
        self.update_agent(now, *update, out);
        if previous == PreviousActivity::Close {
            self.settle_drain_at_swap(now, previous, out);
        }
        self.activity.handoff.swapped(previous, next);
        self.activity.hooks.stop_describing();
        self.answer(call, Answer::Handoff(Ok(())));
        // The task's step precedes its `on_enter`, as `AgentUpdated` does.
        if let Some(task) = task {
            match next {
                NewActivity::Start => self.on_task_frame_installed(now, task, out),
                NewActivity::Resume => self.on_task_frame_popped(now, task, out),
            }
        }
        if next == NewActivity::Start {
            self.issue_on_enter(now, out);
        }
        // The run's `on_enter`, owned above, keeps the run open.
        self.run_release(Owned::Handoff(id), out);
        self.deliver_parked_turn(now, out);
        // Unless this swap began a switch itself (a task whose handle settled before its agent
        // landed hands back at once): starting the queued one over it would lose the exit.
        self.pump_queue(now, out);
    }

    /// The transition window: from a tool's handoff request (`_new_turns_blocked` is set in
    /// `update_agent`) through the swap. A describe pending across a swap opens none for the
    /// arriving agent.
    pub(crate) fn in_window(&self) -> bool {
        self.activity.handoff.switching() || self.activity.hooks.describing()
    }

    fn deliver_parked_turn(&mut self, now: Ts, out: &mut Out) {
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
        if self.activity.drain.exiting() {
            self.finish_drain(Some(crate::agent_session::DrainRefused::Closing), out);
        }
        // A DTMF collection ends with the session: its answer, if still owed, is `Stopped`.
        self.dtmf_stop(out);
        self.refuse_handoffs(out);
        self.activity.tools.abandon_replies();
        // All abandoned before any is let go: the run reads what is still pending.
        let pending = self.activity.hooks.take_all();
        for &(id, kind) in &pending {
            let how = HookEnd::Abandoned;
            out.trace(HandoffStep::HookFinished { hook: id, kind, how });
            self.run_release(Owned::Hook(id), out);
        }
        // A step waiting on its `function_tools_executed` handlers goes on as issued: its results
        // are committed and its reply decided.
        for (id, kind) in pending {
            if kind == HookKind::ToolsExecuted {
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
        // `previous`, `next`, `call` and `task` are the caller's, read before the swap.
        let AgentUpdate {
            agent, instructions, toolset, chat_ctx, llm, tts, stt, vad, options, ..
        } = update;
        // A reply in flight keeps the instructions and tools it snapshotted.
        let count = toolset.specs.len();
        let previous_agent = self.activity.set_frame(agent, instructions, toolset);
        self.session.apply_providers(llm, tts);

        // The new agent's view: the context it brought, or the conversation carried forward.
        let replaced_ctx = chat_ctx.is_some();
        let agent = self.activity.agent.clone();
        match chat_ctx {
            Some(items) => {
                self.session.record.update_agent_chat_ctx(items, &agent);
            }
            None => {
                // The resume's carry: the task's calls leave the view first, then the host is told
                // which of the task's items are now the parent's.
                self.exclude_exiting_task_calls();
                let from = self.session.record.frame().to_owned();
                let items = self.session.record.set_frame(&agent, true);
                if from != agent {
                    out.machine().emit(crate::events::SessionEvent::ContextCarried {
                        from,
                        to: agent.clone(),
                        items,
                    });
                }
            }
        }
        self.publish_frame();

        // The handoff's two records, minted here only: the `AgentHandoff` item
        // (`record_handoff_item`) and the history-only `AgentConfigUpdate` (`_start_session`),
        // stamped on the session clock. Empty without an agent id (test harness): `None`.
        self.record_frame((!previous_agent.is_empty()).then_some(previous_agent), out);

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
        let ctx = replaced_ctx.then(|| self.session.record.for_realtime());
        self.push_frame(ctx, out);

        // A handoff lifts the pause the switch caused; a preemptive generation about the old turn
        // goes too.
        self.resume_scheduling_task(now, out);
        let legs = crate::trace::LegPresence {
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

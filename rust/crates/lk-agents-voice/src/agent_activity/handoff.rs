//! Frame changes: a tool's or the host's handoff from `on_exit` through the drain to the swap
//! (`update_agent`), the turn parked across it, and the host's `drain()`.

use crate::agent::TurnEdit;
use crate::agent_session::cycle::Cycle;
use crate::agent_activity::hooks::{DrainJoin, DrainPoll, HandoffId, HandoffPoll, SwapExit};
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_session::{
    AgentUpdate, DrainRefused, Driver, HandoffRefused, HookCall, HookId, HookKind, NewActivity,
    PreviousActivity,
};
use crate::run_result::Owned;
use crate::trace::{HandoffStep, HookEnd};
use lk_agents_abi::Ts;

impl Driver {
    /// The handoff begins when the host answers.
    pub(crate) fn request_describe(
        &mut self,
        now: Ts,
        speech: SpeechId,
        agent: lk_agents_abi::HostRef,
        cycle: &mut Cycle,
    ) {
        let id = self.activity.hooks_mut().mint();
        let wait = self.hook_wait();
        // Its place in the order is now, while the host answers ([`Queued::Describing`]).
        if self.issue_hook(now, HookCall::DescribeAgent { id, agent, speech }, Some(wait), cycle) {
            self.activity.hooks_mut().hold_place(id);
        }
    }

    /// The entry to a frame change. Refused when closing (a resume is not), queued behind a change
    /// in flight or a pending describe, begun otherwise.
    pub(crate) fn begin_handoff(&mut self, now: Ts, update: Box<AgentUpdate>, cycle: &mut Cycle) {
        let id = self.activity.hooks_mut().mint_handoff();
        // A task's resume completes the current transition before a queued host change: queued
        // behind it, the task's old parent would overwrite the host's agent.
        let resuming_task = update.task.is_some() && update.next == NewActivity::Resume;
        let queued = self.activity.hooks().must_wait(resuming_task);
        if self.session.is_closing() && update.next == NewActivity::Start {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
            return;
        }
        // An open run owns every frame change asked for while it is open.
        self.run_own(Owned::Handoff(id), cycle);
        if queued {
            self.activity.hooks_mut().queue_handoff(id, update);
            return;
        }
        self.start_handoff(now, id, update, cycle);
    }

    pub(super) fn describe_answered(
        &mut self,
        now: Ts,
        hook: HookId,
        update: Box<AgentUpdate>,
        cycle: &mut Cycle,
    ) {
        let Some(slot) = self.activity.hooks().place_of(hook) else {
            // No place held (the close emptied the queue): asked for now, and refused as closing.
            self.begin_handoff(now, update, cycle);
            return;
        };
        let id = self.activity.hooks_mut().mint_handoff();
        if self.session.is_closing() && update.next == NewActivity::Start {
            self.activity.hooks_mut().give_up_place(slot);
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
            return;
        }
        self.run_own(Owned::Handoff(id), cycle);
        self.activity.hooks_mut().fill_place(slot, id, update);
        self.pump_queue(now, cycle);
    }

    /// A describe that will not be answered gives up its place: the asking frame takes the parked
    /// turn, and what waited behind goes ahead.
    pub(super) fn drop_describe(&mut self, now: Ts, hook: HookId, cycle: &mut Cycle) {
        self.activity.hooks_mut().drop_place(hook);
        self.deliver_parked_turn(now, cycle);
        self.pump_queue(now, cycle);
    }

    /// A held describe blocks what is behind.
    fn pump_queue(&mut self, now: Ts, cycle: &mut Cycle) {
        if let Some((id, update)) = self.activity.hooks_mut().next_handoff() {
            self.start_handoff(now, id, update, cycle);
        }
    }

    /// `id` is the handoff's when it was begun (`None` for a command refused before it was).
    pub(crate) fn refuse_handoff(
        &mut self,
        id: Option<HandoffId>,
        mut update: Box<AgentUpdate>,
        cause: HandoffRefused,
        cycle: &mut Cycle,
    ) {
        if let Some(id) = id {
            self.run_release(Owned::Handoff(id), cycle);
        }
        if let Some(done) = update.done.take() {
            let _ = done.send(Err(cause));
        }
        // An inline task's entry or exit: the task is told.
        if let Some(task) = update.task {
            let now = self.shared.time.now();
            self.on_task_handoff_refused(now, task, update.next, cycle);
        }
    }

    /// Every queued frame change refused as closing (a held describe just goes).
    pub(crate) fn refuse_queued_handoffs(&mut self, cycle: &mut Cycle) {
        while let Some(queued) = self.activity.hooks_mut().pop_queued() {
            if let Some((id, update)) = queued {
                self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
            }
        }
    }

    /// The first phase: `on_exit` on the frame being left (`Close`), or straight to the drain
    /// (`Pause`: an awaited `AgentTask`'s parent runs no hook).
    fn start_handoff(
        &mut self,
        now: Ts,
        id: HandoffId,
        update: Box<AgentUpdate>,
        cycle: &mut Cycle,
    ) {
        // The parent is captured now that the entry starts (it may have changed while queued).
        if let Some(task) = update.task.clone()
            && update.next == NewActivity::Start
        {
            self.capture_task_parent(task, &update);
        }
        // Which `on_exit` the change waits on, and whether this change issues it. It runs once: one
        // already running is waited for, one that ran is not rerun; an awaited `AgentTask`'s parent
        // runs none.
        let (exit_hook, issue) = match update.previous {
            PreviousActivity::Pause => (None, false),
            PreviousActivity::Close => self.activity.hooks_mut().frame_exit(),
        };
        let hook = exit_hook.unwrap_or(HookId(0));
        self.activity.hooks_mut().start_handoff(id, update, hook);
        if self.issue_exit(now, exit_hook, issue, cycle) {
            self.drain_handoff(now, cycle);
        }
    }

    fn issue_exit(
        &mut self,
        now: Ts,
        hook: Option<HookId>,
        issue: bool,
        cycle: &mut Cycle,
    ) -> bool {
        match hook {
            None => true,
            Some(_) if !issue => false,
            Some(hook) => self.issue_on_exit(now, hook, cycle),
        }
    }

    /// `true` when there is no host and it is over.
    pub(crate) fn issue_on_exit(&mut self, now: Ts, hook: HookId, cycle: &mut Cycle) -> bool {
        let call = HookCall::OnExit { id: hook, agent: self.activity.agent().to_owned() };
        let wait = self.hook_wait();
        !self.issue_hook(now, call, Some(wait), cycle)
    }

    /// The second phase: the drain, through the scheduling task; the swap lands in the settling
    /// cycle.
    pub(super) fn drain_handoff(&mut self, now: Ts, cycle: &mut Cycle) {
        let (done, settled) = tokio::sync::oneshot::channel();
        if !self.activity.hooks_mut().handoff_draining(settled) {
            return;
        }
        self.pause_scheduling(now, PauseCause::Handoff, done, cycle);
    }

    /// `true` means the swap is due now. A refused pause ends the handoff as refused, and the frame
    /// that blocked the parked turn takes it in that cycle, so it is kept exactly once. Under a
    /// close it drops.
    pub(crate) fn handoff_ready(&mut self, cycle: &mut Cycle) -> bool {
        match self.activity.hooks_mut().poll_handoff() {
            HandoffPoll::Settled => true,
            HandoffPoll::Waiting => false,
            HandoffPoll::Refused(id, update) => {
                let cause = if self.session.is_closing() {
                    HandoffRefused::Closing
                } else {
                    HandoffRefused::DrainRefused
                };
                self.refuse_handoff(Some(id), update, cause, cycle);
                if cause == HandoffRefused::DrainRefused {
                    let now = self.shared.time.now();
                    self.deliver_parked_turn(now, cycle);
                    // The frame stays: a drain that joined the change runs on it.
                    self.rebind_joined_drain(now, cycle);
                }
                false
            }
        }
    }

    /// The swap: the closing re-check, the frame installed with its two records, `done` answered,
    /// `on_enter` issued, the parked turn delivered, the next queued handoff begun.
    pub(crate) fn perform_handoff(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some((id, mut update)) = self.activity.hooks_mut().take_handoff() else {
            return;
        };
        if self.session.is_closing() && update.next == NewActivity::Start {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
            self.refuse_queued_handoffs(cycle);
            return;
        }
        let done = update.done.take();
        let next = update.next;
        let previous = update.previous;
        let task = update.task.clone();
        // A drain bound to a frame an inline task pauses runs that frame's `on_exit` first.
        if previous == PreviousActivity::Pause {
            self.settle_drain_at_swap(now, previous, cycle);
        }
        self.update_agent(now, *update, cycle);
        // The frame a host `drain()` asked about is no longer driving (`settle_drain_at_swap`).
        if previous == PreviousActivity::Close {
            self.settle_drain_at_swap(now, previous, cycle);
        }
        // A paused frame keeps its exit beneath the arriving one and gets it back on resume. The
        // frame that asked a pending describe has left; the arriving one is not blocked by it.
        self.activity.hooks_mut().frame_swapped(previous, next);
        if let Some(done) = done {
            let _ = done.send(Ok(()));
        }
        // The task's step precedes its `on_enter`, as `AgentUpdated` does.
        if let Some(task) = task {
            match next {
                NewActivity::Start => self.on_task_frame_installed(now, task, cycle),
                NewActivity::Resume => self.on_task_frame_popped(now, task, cycle),
            }
        }
        if next == NewActivity::Start {
            self.issue_on_enter(now, cycle);
        }
        // The run's `on_enter`, owned above, keeps the run open.
        self.run_release(Owned::Handoff(id), cycle);
        self.deliver_parked_turn(now, cycle);
        // The next queued handoff -- unless this swap began one itself (a task whose handle settled
        // before its frame landed hands back at once): starting the queued one over it would lose
        // the exit. The reference's `_activity_lock` orders it the same way.
        self.pump_queue(now, cycle);
    }

    fn deliver_parked_turn(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.activity.hooks().in_window() {
            return;
        }
        if let Some((info, edits)) = self.activity.hooks_mut().take_parked_turn() {
            match edits {
                None => self.on_end_of_turn(now, info, cycle),
                Some(edits) => self.finish_turn(now, info, edits, cycle),
            }
        }
    }

    /// The close: pending handoffs refused; a running hook is abandoned, not cancelled.
    pub(crate) fn abandon_handoffs(&mut self, cycle: &mut Cycle) {
        // A drain still waiting on its `on_exit` is refused as closing; one already pausing is
        // answered by the pause's refusal.
        if self.activity.hooks().drain_exiting() {
            self.finish_drain(Some(DrainRefused::Closing), cycle);
        }
        // A DTMF collection ends with the session: its answer, if still owed, is `Stopped`.
        self.dtmf_stop(cycle);
        if let Some((id, update)) = self.activity.hooks_mut().take_handoff() {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
        }
        self.refuse_queued_handoffs(cycle);
        self.activity.hooks_mut().clear_parked_turn();
        self.activity.tools_mut().abandon_replies();
        // All abandoned before any is let go: the run reads what is still pending.
        let pending = self.activity.hooks_mut().take_pending();
        for &(id, kind) in &pending {
            let how = HookEnd::Abandoned;
            cycle.trace(HandoffStep::HookFinished { hook: id, kind, how });
            self.run_release(Owned::Hook(id), cycle);
        }
        // A step waiting on its `function_tools_executed` handlers goes on as issued: its results
        // are committed and its reply decided.
        for (id, kind) in pending {
            if kind == HookKind::ToolsExecuted {
                self.on_tools_executed_handled(id, None, cycle);
            }
        }
    }

    /// A fresh park consumes the turn in the machine, so the next countdown cannot commit its words
    /// twice. A second turn in the window is folded in, so the arriving frame decides one turn.
    pub(crate) fn park_turn(
        &mut self,
        now: Ts,
        mut info: crate::audio_recognition::EndOfTurnInfo,
        edits: Option<Vec<TurnEdit>>,
        cycle: &mut Cycle,
    ) -> bool {
        if !self.activity.hooks().in_window() {
            return false;
        }
        if edits.is_none() {
            let taken = crate::audio_recognition::CommitDecision {
                verdict: crate::audio_recognition::CommitVerdict::Reply,
                committed: true,
                cancel_false_interruption: false,
            };
            let eff = self.activity.audio_recognition_mut().on_commit_verdict(now, &info, &taken);
            cycle.machine().append(eff);
        }
        let edits = match self.activity.hooks_mut().take_parked_turn() {
            Some((earlier, earlier_edits)) => {
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
        self.activity.hooks_mut().park_turn(info, edits);
        true
    }

    /// The handoff: **reconfigure the running session for a new agent.** Nothing is swapped: the
    /// scheduling task, socket, pump and unchanged streams keep running; only what the next plan
    /// reads and the changed legs move.
    pub(crate) fn update_agent(&mut self, now: Ts, update: AgentUpdate, cycle: &mut Cycle) {
        let reopened = update.reopens();
        // `previous`, `next`, `done` and `task` are the caller's, read before the swap.
        let AgentUpdate {
            agent, instructions, toolset, chat_ctx, llm, tts, stt, vad, options, ..
        } = update;
        // Empty without an agent id (test harness); the record spells it `None`.
        // A reply in flight keeps the instructions and tools it snapshotted.
        let count = toolset.specs.len();
        let previous_agent = self.activity.set_frame(agent, instructions, toolset);
        self.session.apply_providers(llm, tts);

        // The new frame's view: the context it brought, or the conversation carried forward.
        let replaced_ctx = chat_ctx.is_some();
        let agent = self.activity.agent().to_owned();
        match chat_ctx {
            Some(items) => {
                self.session.update_agent_chat_ctx(items, &agent);
            }
            None => {
                // The resume's carry: the task's calls leave the view first, then the host is told
                // which of the task's items are now the parent's.
                self.exclude_exiting_task_calls();
                let from = self.session.chat_ctx().frame().to_owned();
                let items = self.session.set_chat_ctx_frame(&agent, true);
                if from != agent {
                    cycle.machine().emit(crate::events::SessionEvent::ContextCarried {
                        from,
                        to: agent.clone(),
                        items,
                    });
                }
            }
        }
        self.publish_frame();

        // The handoff's two records, minted here only: the `AgentHandoff` item
        // (`_update_activity`'s `record_handoff_item`) and the history-only `AgentConfigUpdate`
        // (`_start_session`), stamped on the session clock.
        self.record_frame((!previous_agent.is_empty()).then_some(previous_agent), cycle);

        let legs = self.reopen_legs(stt, vad, cycle);

        // The agent layer, replaced whole, over the legs this update left (an STT that refused to
        // open is still the old one). The owner is re-resolved by the stack. Only the STT and VAD
        // facts are resolved against; the session's providers are the one record of the LLM/TTS.
        let diff = self.session.set_agent_options(legs.stt, legs.vad, options);
        self.apply_config(now, crate::agent_activity::options::ConfigLayer::Agent, diff, cycle);

        // A realtime handoff is three commands on the same socket, not a reconnect. Ungated by the
        // `mutable_*` capabilities: a refusal on the wire is visible, a skipped push is not. The
        // chat context only when the agent brought one, through `for_realtime`.
        let ctx = replaced_ctx.then(|| self.session.chat_ctx().for_realtime());
        self.push_frame(ctx, cycle);

        // A handoff lifts the pause the frame change caused; a preemptive generation about the old
        // turn goes too.
        self.resume_scheduling_task(now, cycle);
        let legs = crate::trace::LegPresence {
            stt: legs.stt,
            vad: legs.vad,
            llm: self.session.llm().is_some(),
            tts: self.session.tts().is_some(),
            realtime: self.activity.rt_session().is_some(),
        };
        cycle.trace(HandoffStep::AgentUpdated {
            tools: count,
            chat_ctx: replaced_ctx,
            reopened,
            legs,
        });
    }

    /// `AgentActivity.drain()`, for `waiters` (one request, or the ones a frame change handed on).
    /// Never refused while running; it waits where the reference's `_lock` would: on the close's
    /// drain, a frame change's swap, or the frame's `on_exit` then the pause.
    pub(crate) fn drain(
        &mut self,
        now: Ts,
        waiters: Vec<tokio::sync::oneshot::Sender<Result<(), DrainRefused>>>,
        cycle: &mut Cycle,
    ) {
        if self.session.is_closing() {
            self.join_close_drain(waiters, cycle);
        } else {
            self.start_drain(now, waiters, cycle);
        }
    }

    fn start_drain(
        &mut self,
        now: Ts,
        waiters: Vec<tokio::sync::oneshot::Sender<Result<(), DrainRefused>>>,
        cycle: &mut Cycle,
    ) {
        let quiet = self.activity.scheduling_task().is_empty();
        match self.activity.hooks_mut().join_drain(waiters, quiet) {
            DrainJoin::Waiting => {}
            DrainJoin::AnswerNow(waiters) => {
                for done in waiters {
                    self.answer_drain(Some(done), None, cycle);
                }
            }
            DrainJoin::JoinedChange { exit_hook } => {
                cycle.trace(HandoffStep::DrainStarted { exit_hook });
            }
            DrainJoin::Exiting { exit_hook, issue } => {
                cycle.trace(HandoffStep::DrainStarted { exit_hook });
                if self.issue_exit(now, exit_hook, issue, cycle) {
                    self.drain_pause(now, cycle);
                }
            }
        }
    }

    /// Left for good (`Close`): answered `Ok` once nothing it said is open. Paused under an inline
    /// task: the paused frame's `on_exit` runs now, once, before the task's `on_enter`, as the
    /// reference does.
    pub(crate) fn settle_drain_at_swap(
        &mut self,
        now: Ts,
        previous: PreviousActivity,
        cycle: &mut Cycle,
    ) {
        let Some(waiters) = self.activity.hooks_mut().take_drain_at_swap() else { return };
        // The `on_exit` the callers still wait for, if any.
        let waiting_on = match self.activity.hooks().exit_at_swap(previous) {
            SwapExit::Nothing => None,
            SwapExit::Running(hook) => Some(hook),
            SwapExit::Issue => {
                let hook = self.activity.hooks_mut().mint();
                cycle.trace(HandoffStep::DrainStarted { exit_hook: Some(hook) });
                (!self.issue_on_exit(now, hook, cycle)).then_some(hook)
            }
        };
        self.activity.hooks_mut().leave_drains(waiting_on, waiters);
        self.left_drains_ready(cycle);
    }

    /// A joined frame change ended without a swap: the callers' drain runs on this frame now.
    pub(crate) fn rebind_joined_drain(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(waiters) = self.activity.hooks_mut().take_joined_drain() else { return };
        if !waiters.is_empty() {
            self.drain(now, waiters, cycle);
        }
    }

    /// Callers whose frame was left for good, answered `Ok` once no speech is open.
    fn left_drains_ready(&mut self, cycle: &mut Cycle) {
        if !self.activity.scheduling_task().is_empty() {
            return;
        }
        let ready = self.activity.hooks_mut().take_left_drains();
        if ready.is_empty() {
            return;
        }
        cycle.trace(HandoffStep::Drained { refused: None });
        for done in ready {
            let _ = done.send(Ok(()));
        }
    }

    /// The `on_exit` hook `id` ended: a drain waiting on it pauses scheduling.
    pub(crate) fn drain_exit_done(&mut self, now: Ts, id: HookId, cycle: &mut Cycle) {
        if self.activity.hooks_mut().exit_ended(id) {
            self.drain_pause(now, cycle);
        }
        // Callers bound to the paused frame wait only for nothing to be open.
        self.activity.hooks_mut().unbind_left_drains(id);
        self.left_drains_ready(cycle);
    }

    /// `_pause_scheduling_task()`, after `on_exit`, so a goodbye it scheduled is waited for.
    fn drain_pause(&mut self, now: Ts, cycle: &mut Cycle) {
        let (done, settled) = tokio::sync::oneshot::channel();
        if !self.activity.hooks_mut().drain_pausing(settled) {
            return;
        }
        self.pause_scheduling(now, PauseCause::Drain, done, cycle);
    }

    /// Has the drain's pause settled? Settled: waiters answered `Ok`; refused: with the reason.
    pub(crate) fn drain_ready(&mut self, cycle: &mut Cycle) {
        self.left_drains_ready(cycle);
        let quiet = self.activity.scheduling_task().is_empty();
        match self.activity.hooks_mut().poll_drain(quiet) {
            DrainPoll::Settled => self.finish_drain(None, cycle),
            DrainPoll::Waiting => {}
            DrainPoll::Refused => {
                let refused = if self.session.is_closing() {
                    DrainRefused::Closing
                } else {
                    DrainRefused::FrameChanging
                };
                self.finish_drain(Some(refused), cycle);
            }
        }
    }

    /// Answer every waiter. `None` leaves the frame `Drained`; a refusal forgets the drain.
    fn finish_drain(&mut self, refused: Option<DrainRefused>, cycle: &mut Cycle) {
        let Some(waiters) = self.activity.hooks_mut().finish_drain(refused) else { return };
        cycle.trace(HandoffStep::Drained { refused });
        let answer = refused.map_or(Ok(()), Err);
        for done in waiters {
            let _ = done.send(answer);
        }
    }

    /// One caller answered on its own: a refusal at the request, or a frame already drained.
    pub(crate) fn answer_drain(
        &mut self,
        done: Option<tokio::sync::oneshot::Sender<Result<(), DrainRefused>>>,
        refused: Option<DrainRefused>,
        cycle: &mut Cycle,
    ) {
        cycle.trace(HandoffStep::Drained { refused });
        if let Some(done) = done {
            let _ = done.send(refused.map_or(Ok(()), Err));
        }
    }
}

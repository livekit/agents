//! Inline `AgentTask`s: [`Tasks`], the entry and exit handoffs, and the awaiting speech's
//! block.

use crate::agent_session::cycle::Cycle;
use crate::agent_activity::pause::InterruptOutcome;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{
    AgentUpdate, Driver, NewActivity, PreviousActivity, TaskCommand, TaskEntry,
};
use crate::run_result::Owned;
use crate::speech_handle::SpeechHandle;
use crate::trace::{CoreStep, InterruptCause, SpeechStep};
use crate::workflows::{ContextMerge, TaskHandle, TaskId, TaskPhase, ToolError};
use lk_agents_abi::{HostRef, Maybe, Ts};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// `agent_activity.AgentActivity._inline_task_slot`.
pub const SPEECH_INTERRUPTED: &str = "the speech that awaited the inline task is interrupted";

/// `agent_activity.AgentActivity._inline_task_slot`.
pub const ACTIVITY_CLOSING: &str = "the activity that awaited the inline task is closing";

/// `agent._handle_task_done`, what the awaiter reads when its speech ended under the task.
pub const AWAITER_FINISHED: &str = "The asyncio.Task finished before the AgentTask was completed.";

/// `AgentTask.__await_impl`'s warning, as the awaiter's error when the entry is refused as orphaned.
pub const AGENT_CHANGED: &str = "the agent changed before the inline task could start, likely due to `AgentSession.update_agent` being invoked";

pub(crate) use tasks::{FrameInstalled, FramePopped, LiveTask, TaskExit, TaskSlot, Tasks};

/// The inline tasks' state, behind its methods.
mod tasks {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum LivePhase {
        /// The entry handoff (`Pause`/`Start`) is in flight; the parent still drives.
        Entering,
        Running,
        /// The exit handoff (`Close`/`Resume`) is in flight; the task's frame still drives.
        Exiting,
    }

    pub(crate) struct LiveTask {
        handle: TaskHandle<HostRef>,
        /// The awaiting speech, blocked and held for the task's lifetime; `None` from a hook.
        speech: Option<SpeechId>,
        /// The task's agent id: what its frame is stamped with and what the exit compares against.
        agent: String,
        merge: ContextMerge,
        phase: LivePhase,
        /// The parent frame as it stood when the entry handoff started: what the exit reinstalls.
        restore: Option<AgentUpdate>,
        /// The settle was journalled. Set at `TaskSettled`, and at once for a refusal.
        settled_seen: bool,
        /// What the open run let go of while this task runs, and owns again on the way out.
        suspended: Vec<Owned>,
    }

    /// One frame's slot: the task past it and the ones waiting for it (`_inline_task_slot`).
    #[derive(Default)]
    pub(crate) struct TaskSlot {
        /// The frame's agent id, once a task named it (the root's is filled at its first entry).
        agent: Option<String>,
        current: Option<LiveTask>,
        queue: VecDeque<TaskEntry>,
    }

    impl LiveTask {
        pub(crate) fn handle(&self) -> &TaskHandle<HostRef> {
            &self.handle
        }

        /// The awaiting speech.
        pub(crate) fn speech(&self) -> Option<SpeechId> {
            self.speech
        }

        pub(crate) fn agent(&self) -> &str {
            &self.agent
        }

        /// What the open run let go of while it ran.
        pub(crate) fn into_suspended(self) -> Vec<Owned> {
            self.suspended
        }
    }

    /// The task's frame landed.
    pub(crate) enum FrameInstalled {
        NoTask,
        /// The driving frame's slot holds another task.
        NotEntering,
        Running {
            speech: Option<SpeechId>,
            exit_now: bool,
        },
    }

    /// The parent's frame came back.
    pub(crate) enum FramePopped {
        NoFrame,
        /// The pushed slot held no task.
        NoTask,
        Popped {
            live: LiveTask,
            task_slot: TaskSlot,
        },
    }

    /// The driving task's exit, as it begins.
    pub(crate) struct TaskExit {
        pub(crate) task: TaskId,
        pub(crate) agent: String,
        pub(crate) awaiting: Option<SpeechId>,
        pub(crate) cancel: bool,
        pub(crate) merge: ContextMerge,
    }

    impl TaskSlot {
        /// Its task and the entries waiting for it.
        pub(crate) fn into_parts(self) -> (Option<LiveTask>, VecDeque<TaskEntry>) {
            (self.current, self.queue)
        }

        /// Whether `entry` belongs in this slot: a task here was awaited from the same speech, or
        /// the awaiter named this frame.
        fn owns(&self, entry: &TaskEntry) -> bool {
            let same_speech = |s: Option<SpeechId>| entry.speech.is_some() && s == entry.speech;
            self.current.as_ref().is_some_and(|t| same_speech(t.speech))
                || self.queue.iter().any(|q| same_speech(q.speech))
                || (entry.parent.is_some() && self.agent == entry.parent)
        }
    }

    #[derive(Default)]
    pub(crate) struct Tasks {
        slot: TaskSlot,
        /// Innermost last. `stack[i].current` is the task whose frame `stack[i+1]` (or `slot`)
        /// serves.
        stack: Vec<TaskSlot>,
        /// Counted: a nested task from the same speech registers again; the last release unblocks.
        blocked: BTreeMap<SpeechId, u32>,
    }

    impl Tasks {
        /// The speeches an inline task blocks, with their registration counts.
        pub(crate) fn blocked(&self) -> &BTreeMap<SpeechId, u32> {
            &self.blocked
        }

        pub(crate) fn any_live(&self) -> bool {
            self.slot.current.is_some() || self.stack.iter().any(|s| s.current.is_some())
        }

        /// A frame beneath the driving one is `agent`'s (a task serves it).
        pub(crate) fn serves_frame(&self, agent: &str) -> bool {
            self.stack.iter().any(|s| s.agent.as_deref() == Some(agent))
        }

        /// The innermost task: one entering on the driving frame, else the one whose frame drives.
        fn innermost(&mut self) -> Option<&mut LiveTask> {
            self.slot.current.as_mut().or_else(|| self.stack.last_mut()?.current.as_mut())
        }

        /// The task whose frame drives has its result: that frame is about to drain.
        pub(crate) fn driving_settled(&self) -> bool {
            self.stack.last().and_then(|s| s.current.as_ref()).is_some_and(|t| t.handle.done())
        }

        /// The task the loop's third wake polls: the innermost, while its settle is unseen.
        pub(crate) fn watched(&self) -> Option<&LiveTask> {
            let live =
                self.slot.current.as_ref().or_else(|| self.stack.last()?.current.as_ref())?;
            (!live.settled_seen).then_some(live)
        }

        /// `entry` joins the slot it belongs to (the driving frame's, named `agent` at the first
        /// entry); returned when that slot is free for it to take now.
        pub(crate) fn enqueue(&mut self, entry: TaskEntry, agent: &str) -> Option<TaskEntry> {
            if self.slot.agent.is_none() {
                self.slot.agent = Some(agent.to_owned());
            }
            // The slot is the awaiter's frame's (`_inline_task_slot` is per activity). Such a slot
            // always holds a task, so the entry queues there.
            if let Some(i) = self.stack.iter().position(|s| s.owns(&entry)) {
                self.stack[i].queue.push_back(entry);
                None
            } else if self.slot.current.is_some() {
                self.slot.queue.push_back(entry);
                None
            } else {
                Some(entry)
            }
        }

        /// The entry takes the driving frame's slot; its handoff is in flight.
        pub(crate) fn enter(
            &mut self,
            handle: TaskHandle<HostRef>,
            speech: Option<SpeechId>,
            agent: String,
            merge: ContextMerge,
        ) {
            self.slot.current = Some(LiveTask {
                handle,
                speech,
                agent,
                merge,
                phase: LivePhase::Entering,
                restore: None,
                settled_seen: false,
                suspended: Vec::new(),
            });
        }

        /// The parent frame the task's exit reinstalls.
        pub(crate) fn set_restore(&mut self, task: &TaskId, restore: AgentUpdate) {
            if let Some(live) = self.find(task) {
                live.restore = Some(restore);
            }
        }

        /// The task's frame landed: it runs, and the parent's slot is pushed beneath its own.
        pub(crate) fn frame_installed(&mut self, task: &TaskId) -> FrameInstalled {
            let Some(live) = self.slot.current.as_mut() else { return FrameInstalled::NoTask };
            if live.handle.id() != *task {
                return FrameInstalled::NotEntering;
            }
            live.phase = LivePhase::Running;
            live.handle.advance(TaskPhase::Running);
            let agent = live.agent.clone();
            let speech = live.speech;
            let exit_now = live.settled_seen;
            let parent_slot = std::mem::take(&mut self.slot);
            self.stack.push(parent_slot);
            self.slot.agent = Some(agent);
            FrameInstalled::Running { speech, exit_now }
        }

        /// What the open run let go of while the driving task runs.
        pub(crate) fn set_suspended(&mut self, suspended: Vec<Owned>) {
            if let Some(live) = self.stack.last_mut().and_then(|s| s.current.as_mut()) {
                live.suspended = suspended;
            }
        }

        /// The parent's frame is back: its slot restored, the task's own slot and the task taken.
        pub(crate) fn pop_frame(&mut self) -> FramePopped {
            let Some(mut parent_slot) = self.stack.pop() else { return FramePopped::NoFrame };
            let Some(live) = parent_slot.current.take() else {
                self.slot = parent_slot;
                return FramePopped::NoTask;
            };
            let task_slot = std::mem::replace(&mut self.slot, parent_slot);
            FramePopped::Popped { live, task_slot }
        }

        /// The entering task `task`, taken (its entry was refused).
        pub(crate) fn take_entering(&mut self, task: &TaskId) -> Option<LiveTask> {
            let live = self.slot.current.take()?;
            if live.handle.id() != *task {
                self.slot.current = Some(live);
                return None;
            }
            Some(live)
        }

        /// `task` is the one whose frame drives.
        pub(crate) fn drives(&self, task: &TaskId) -> bool {
            let top = self.stack.last().and_then(|s| s.current.as_ref());
            top.is_some_and(|t| t.handle.id() == *task)
        }

        /// The exiting task whose calls leave the view (`exclude_function_call`): its agent.
        pub(crate) fn exiting_excluded_agent(&mut self) -> Option<String> {
            let live = self.innermost()?;
            (matches!(live.phase, LivePhase::Exiting) && live.merge.exclude_function_call)
                .then(|| live.agent.clone())
        }

        /// `agent._handle_task_done`: the innermost task `task` settled (its awaiter gone, it is
        /// terminated). Whether its exit begins now (a task still entering begins it when its frame
        /// lands).
        pub(crate) fn settle(&mut self, task: &TaskId) -> bool {
            let Some(live) = self.innermost() else { return false };
            if live.handle.id() != *task || live.settled_seen {
                return false;
            }
            if !live.handle.done() {
                let _ = live.handle.fail(ToolError::plain(AWAITER_FINISHED));
            }
            live.settled_seen = true;
            live.phase == LivePhase::Running
        }

        /// The driving task's exit begins.
        pub(crate) fn begin_exit(&mut self) -> Option<TaskExit> {
            let live = self.stack.last_mut()?.current.as_mut()?;
            live.phase = LivePhase::Exiting;
            live.handle.advance(TaskPhase::Exiting);
            Some(TaskExit {
                task: live.handle.id(),
                agent: live.agent.clone(),
                awaiting: live.speech,
                cancel: live.handle.cancel_requested(),
                merge: live.merge,
            })
        }

        /// The driving task's captured parent frame, taken; `None` with no driving task.
        pub(crate) fn take_restore(&mut self) -> Option<Option<AgentUpdate>> {
            let live = self.stack.last_mut()?.current.as_mut()?;
            Some(live.restore.take())
        }

        /// The innermost pushed slot, popped: a parent frame that is not coming back.
        pub(crate) fn pop_parent(&mut self) -> Option<TaskSlot> {
            self.stack.pop()
        }

        /// The driving frame's slot, taken whole.
        pub(crate) fn take_driving_slot(&mut self) -> TaskSlot {
            std::mem::take(&mut self.slot)
        }

        /// The next entry queued on the driving frame's slot, once it is free.
        pub(crate) fn next_entry(&mut self) -> Option<TaskEntry> {
            if self.slot.current.is_some() {
                return None;
            }
            self.slot.queue.pop_front()
        }

        /// The entries queued on the driving frame's slot, taken.
        pub(crate) fn take_queue(&mut self) -> VecDeque<TaskEntry> {
            std::mem::take(&mut self.slot.queue)
        }

        /// The close cancels the innermost task: `None` when nothing is live, `Some(None)` when it
        /// is already handing back, else the task whose settle the loop now takes.
        pub(crate) fn cancel_innermost(&mut self) -> Option<Option<TaskId>> {
            let live = self.innermost()?;
            if live.settled_seen {
                return Some(None);
            }
            if !live.handle.done() {
                live.handle.cancel(&live.agent);
            } else {
                live.handle.request_cancel();
            }
            Some(Some(live.handle.id()))
        }

        /// `_add_drain_blocked_tasks`, counted: whether this is the speech's first block.
        pub(crate) fn block(&mut self, speech: SpeechId) -> bool {
            let count = self.blocked.entry(speech).or_insert(0);
            *count += 1;
            *count == 1
        }

        /// The matching release: whether it was the last.
        pub(crate) fn unblock(&mut self, speech: SpeechId) -> bool {
            let Some(count) = self.blocked.get_mut(&speech) else { return false };
            *count = count.saturating_sub(1);
            if *count != 0 {
                return false;
            }
            self.blocked.remove(&speech);
            true
        }

        fn find(&mut self, task: &TaskId) -> Option<&mut LiveTask> {
            if self.slot.current.as_ref().is_some_and(|t| &t.handle.id() == task) {
                return self.slot.current.as_mut();
            }
            self.stack
                .iter_mut()
                .find_map(|s| s.current.as_mut().filter(|t| &t.handle.id() == task))
        }
    }
}

/// The parent's leg, to change back to, when the task's frame changes it.
fn back_if<A, T: Clone>(entering: &Maybe<A>, current: &T) -> Maybe<T> {
    if entering.is_given() { Maybe::Given(current.clone()) } else { Maybe::NotGiven }
}

/// What the loop's third wake polls: the innermost unsettled task and its awaiting speech.
pub(crate) struct TaskWake {
    pub(crate) task: TaskId,
    pub(crate) handle: TaskHandle<HostRef>,
    /// The awaiting speech: `Some(None)` when the loop released it already (fires at once).
    pub(crate) speech: Option<Option<Arc<SpeechHandle>>>,
}

impl TaskWake {
    /// Resolves when the handle settles or the speech ends. Level-triggered, so an early settle
    /// is not lost.
    pub(crate) async fn wait(self) -> TaskId {
        let TaskWake { task, handle, speech } = self;
        let speech_ended = async {
            match speech {
                Some(Some(s)) => s.wait_for_done().await,
                Some(None) => {}
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            () = handle.wait_settled() => {}
            () = speech_ended => {}
        }
        task
    }
}

impl Driver {
    /// The loop's task wake, if one is live. A speech the loop no longer holds fires it at once.
    pub(crate) fn task_wake(&self) -> Option<TaskWake> {
        let live = self.activity.tasks().watched()?;
        let speech = live.speech().map(|s| self.activity.scheduling_task().handle(s).cloned());
        Some(TaskWake { task: live.handle().id(), handle: live.handle().clone(), speech })
    }

    pub(crate) fn on_task(&mut self, now: Ts, cmd: TaskCommand, cycle: &mut Cycle) {
        match cmd {
            TaskCommand::Enter(entry) => self.enter_task(now, *entry, cycle),
        }
    }

    /// `await task` up to the slot: the re-entrancy check, the speech's interruption check and
    /// hold, the block, then the slot -- taken now or queued for.
    fn enter_task(&mut self, now: Ts, entry: TaskEntry, cycle: &mut Cycle) {
        if entry.handle.begin().is_err() {
            return;
        }
        // Closing: refused before the speech is held or blocked, so the close's drain cannot cut
        // the speech before its outputs commit.
        if self.session.is_closing() {
            return self.refuse_closing(now, &entry.handle, None, cycle);
        }
        if let Some(speech) = entry.speech {
            // An interrupted speech cannot anchor a task; a live one is held so a barge-in cannot
            // end it under the tool (`_inline_task_slot`).
            let held = self
                .activity
                .scheduling_task()
                .handle(speech)
                .is_some_and(|h| !h.interrupted() && h.hold_interruptions().is_ok());
            if !held {
                let _ = entry.handle.fail(ToolError::plain(SPEECH_INTERRUPTED));
                return self.release_awaiter(now, &entry.handle, None, cycle);
            }
            // The host swapped the agent under the speech's tool: the reference would refuse the
            // hand-back ("Ignoring handoff to the previous agent"), so the entry is refused as an
            // orphan and the speech cut whole.
            if self.speech_frame_left(speech) {
                return self.refuse_orphan(now, &entry.handle, speech, cycle);
            }
            // Block before queueing: otherwise a pause waits for a speech waiting for the pause.
            self.block_speech(now, speech, cycle);
        }
        let agent = self.activity.agent().to_owned();
        if let Some(entry) = self.activity.tasks_mut().enqueue(entry, &agent) {
            self.take_slot(now, entry, cycle);
        }
    }

    /// Past the queue: `enter_task`'s closing and swapped-frame checks again now that the entry
    /// starts, then the entry handoff (parent paused, drained but for blocked speeches, the task's
    /// frame installed with its `on_enter`).
    fn take_slot(&mut self, now: Ts, entry: TaskEntry, cycle: &mut Cycle) {
        let TaskEntry { handle, mut frame, speech, preserve_function_call_history, parent: _ } =
            entry;
        if self.session.is_closing() {
            self.refuse_closing(now, &handle, speech, cycle);
            return self.next_queued(now, cycle);
        }
        if let Some(speech) = speech
            && self.speech_frame_left(speech)
        {
            self.refuse_orphan(now, &handle, speech, cycle);
            return self.next_queued(now, cycle);
        }
        handle.advance(TaskPhase::Entering);
        let agent = frame.agent.clone();
        frame.previous = PreviousActivity::Pause;
        frame.next = NewActivity::Start;
        frame.done = None;
        frame.task = Some(handle.id());
        let merge = ContextMerge::for_task(preserve_function_call_history);
        self.activity.tasks_mut().enter(handle, speech, agent, merge);
        self.begin_handoff(now, Box::new(frame), cycle);
    }

    /// The entry handoff is starting: capture the parent now, since the entry may have queued
    /// behind a host handoff.
    pub(crate) fn capture_task_parent(&mut self, task: TaskId, entering: &AgentUpdate) {
        let restore = AgentUpdate {
            agent: self.activity.agent().to_owned(),
            instructions: self.activity.instructions().clone(),
            toolset: (**self.activity.toolset()).clone(),
            // The record is carried, the task's turns folded into the parent's view (`set_frame`).
            chat_ctx: None,
            // Only a leg the task's frame changes is changed back: `Given(same)` reopens a leg.
            llm: back_if(&entering.llm, &self.session.llm().cloned()),
            tts: back_if(&entering.tts, &self.session.tts().cloned()),
            stt: back_if(&entering.stt, &self.session.stt().cloned()),
            vad: back_if(&entering.vad, &self.session.vad().cloned()),
            options: self.session.agent_options(),
            previous: PreviousActivity::Close,
            next: NewActivity::Resume,
            done: None,
            task: Some(task.clone()),
        };
        self.activity.tasks_mut().set_restore(&task, restore);
    }

    /// The task's frame landed: push the parent's slot. A settle seen while entering begins the
    /// exit at once.
    pub(crate) fn on_task_frame_installed(&mut self, now: Ts, task: TaskId, cycle: &mut Cycle) {
        let (speech, exit_now) = match self.activity.tasks_mut().frame_installed(&task) {
            FrameInstalled::NoTask => return,
            FrameInstalled::NotEntering => {
                cycle.trace(CoreStep::PortFailure(
                    "a task frame landed for a task that is not the frame's entering one",
                ));
                return;
            }
            FrameInstalled::Running { speech, exit_now } => (speech, exit_now),
        };
        // The watching run lets the awaiting speech and parent's `on_enter` go while the task runs.
        let suspended = self.run_suspend_for_task(speech, cycle);
        self.activity.tasks_mut().set_suspended(suspended);
        if exit_now {
            self.begin_task_exit(now, cycle);
        }
    }

    /// The parent's frame is back: the task's slot is discarded (its queue refused), the parent's
    /// restored, the awaiter released; then the next queued task takes the slot.
    pub(crate) fn on_task_frame_popped(&mut self, now: Ts, task: TaskId, cycle: &mut Cycle) {
        let (live, task_slot) = match self.activity.tasks_mut().pop_frame() {
            FramePopped::NoFrame => {
                cycle.trace(CoreStep::PortFailure(
                    "a task exit landed with no pushed frame beneath it",
                ));
                return;
            }
            FramePopped::NoTask => {
                cycle.trace(CoreStep::PortFailure(
                    "a task exit landed for a frame that holds no task",
                ));
                return;
            }
            FramePopped::Popped { live, task_slot } => (live, task_slot),
        };
        if live.handle().id() != task {
            cycle.trace(CoreStep::PortFailure(
                "a task exit landed for a task that is not the frame's",
            ));
        }
        self.refuse_slot(now, task_slot, cycle);
        self.finish_task(now, live, cycle);
        self.after_task_exit(now, cycle);
    }

    /// A task's handoff was refused. Entry: the task never ran (`ToolError("activity doesn't start
    /// ...")` unless it settled meanwhile). Exit: the parent is not reinstalled; the awaiter is
    /// released.
    pub(crate) fn on_task_handoff_refused(
        &mut self,
        now: Ts,
        task: TaskId,
        next: NewActivity,
        cycle: &mut Cycle,
    ) {
        match next {
            NewActivity::Start => {
                let Some(live) = self.activity.tasks_mut().take_entering(&task) else { return };
                if !live.handle().done() {
                    let _ = live.handle().fail(ToolError::plain(format!(
                        "activity doesn't start for {}, likely due to session closing",
                        live.agent()
                    )));
                }
                self.release_awaiter(now, live.handle(), live.speech(), cycle);
                self.next_queued(now, cycle);
            }
            NewActivity::Resume => {
                if !self.activity.tasks().drives(&task) {
                    return;
                }
                // The task's frame stays; a close abandons the rest, the parent's queue with it.
                let Some(live) = self.pop_parent_slot(now, cycle) else { return };
                self.finish_task(now, live, cycle);
            }
        }
        self.after_task_exit(now, cycle);
    }

    /// Before the exit's carry folds the task's turns into the parent: the task's calls and
    /// outputs leave the view when `exclude_function_call` says so, including those committed
    /// during the exit's drain.
    pub(crate) fn exclude_exiting_task_calls(&mut self) {
        let Some(agent) = self.activity.tasks_mut().exiting_excluded_agent() else { return };
        self.session.exclude_task_calls(&agent);
    }

    /// `Wake::Task`: the innermost task's handle settled, or its awaiting speech ended under it.
    /// A task still entering begins its exit when its frame lands (`on_task_frame_installed`).
    pub(crate) fn on_task_wake(&mut self, now: Ts, task: TaskId, cycle: &mut Cycle) {
        // `agent._handle_task_done`: the awaiter is gone; the task is terminated.
        if self.activity.tasks_mut().settle(&task) {
            self.begin_task_exit(now, cycle);
        }
    }

    /// The `finally` of `AgentTask.__await_impl`: interrupt the task's speeches, the agent-changed
    /// check, the merge, the exit handoff.
    fn begin_task_exit(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(TaskExit { task, agent, awaiting, cancel, merge }) =
            self.activity.tasks_mut().begin_exit()
        else {
            return;
        };
        let blocked: Vec<SpeechId> = self.activity.tasks().blocked().keys().copied().collect();
        if cancel {
            // `interrupt(force=True)`: every speech of the driving frame but parked parent speeches.
            let open: Vec<SpeechId> = self
                .activity
                .scheduling_task()
                .speech_ids()
                .filter(|s| !blocked.contains(s))
                .collect();
            for speech in open {
                match self.interrupt_speech(now, speech, InterruptCause::Host, true, cycle) {
                    Ok(InterruptOutcome::Now { .. } | InterruptOutcome::Already) => {}
                    Err(_) => cycle.trace(CoreStep::PortFailure(
                        "a cancelled task's forced interrupt was refused",
                    )),
                }
            }
        }
        if self.activity.agent() != agent || self.activity.hooks().host_handoff_pending() {
            // The host swapped the agent (`current_agent` moves at the `update_agent` call, so a
            // change still draining or queued counts): no parent to hand back to ("Ignoring handoff
            // to the previous agent"). The parent's slot goes with it, its queued tasks refused;
            // the awaiting speech is cut whole, its outputs still commit.
            if let Some(speech) = awaiting {
                self.cut_orphaned(now, speech, cycle);
            }
            let Some(live) = self.pop_parent_slot(now, cycle) else { return };
            self.finish_task(now, live, cycle);
            self.after_task_exit(now, cycle);
            return;
        }
        // `old_agent.chat_ctx.merge(self.chat_ctx, exclude_function_call=..)`.
        if merge.exclude_function_call {
            self.session.exclude_task_calls(&agent);
            self.publish_frame();
        }
        let Some(restore) = self.activity.tasks_mut().take_restore() else { return };
        let Some(restore) = restore else {
            cycle.trace(CoreStep::PortFailure("a task exit with no captured parent frame"));
            let Some(live) = self.pop_parent_slot(now, cycle) else { return };
            self.finish_task(now, live, cycle);
            self.after_task_exit(now, cycle);
            return;
        };
        debug_assert_eq!(restore.task, Some(task));
        self.begin_handoff(now, Box::new(restore), cycle);
    }

    /// An orphaned task's awaiting speech, cut whole (its outputs still commit).
    fn cut_orphaned(&mut self, now: Ts, speech: SpeechId, cycle: &mut Cycle) {
        if self.activity.scheduling_task().contains(speech)
            && self.interrupt_speech(now, speech, InterruptCause::Handoff, true, cycle).is_err()
        {
            cycle.trace(CoreStep::PortFailure(
                "an orphaned task's awaiting speech refused its forced cut",
            ));
        }
    }

    /// The entry refused as orphaned: the awaiter reads `AGENT_CHANGED`, the speech is cut whole
    /// (outputs still commit), and whatever the entry held is released.
    fn refuse_orphan(
        &mut self,
        now: Ts,
        handle: &TaskHandle<HostRef>,
        speech: SpeechId,
        cycle: &mut Cycle,
    ) {
        let _ = handle.fail(ToolError::plain(AGENT_CHANGED));
        self.cut_orphaned(now, speech, cycle);
        self.release_awaiter(now, handle, Some(speech), cycle);
    }

    /// A parent frame that is not coming back: its slot leaves the chain, its running task
    /// returned, every queued task refused as closing.
    fn pop_parent_slot(&mut self, now: Ts, cycle: &mut Cycle) -> Option<LiveTask> {
        let (current, queue) = self.activity.tasks_mut().pop_parent()?.into_parts();
        self.refuse_queue(now, queue, cycle);
        current
    }

    /// The task is over for its awaiter: speech unblocked and released, handle `Exited`; what the
    /// task suspended is the current run's again (`agent.AgentTask.__await_impl`).
    fn finish_task(&mut self, now: Ts, live: LiveTask, cycle: &mut Cycle) {
        self.release_awaiter(now, live.handle(), live.speech(), cycle);
        self.run_resume_after_task(live.into_suspended(), cycle);
    }

    /// Unblock and release the speech, then wake the awaiter (`Exited` last, so it never sees a
    /// resumed parent still un-interruptible).
    fn release_awaiter(
        &mut self,
        now: Ts,
        handle: &TaskHandle<HostRef>,
        speech: Option<SpeechId>,
        cycle: &mut Cycle,
    ) {
        if let Some(speech) = speech {
            self.unblock_speech(now, speech, cycle);
            if let Some(h) = self.activity.scheduling_task().handle(speech) {
                h.release_interruptions();
            }
        }
        if handle.phase() < TaskPhase::Exiting {
            handle.advance(TaskPhase::Exiting);
        }
        handle.advance(TaskPhase::Exited);
    }

    /// After a task left the chain: the next queued task takes the slot; an unwinding close moves on.
    fn after_task_exit(&mut self, now: Ts, cycle: &mut Cycle) {
        self.next_queued(now, cycle);
        self.close_unwind_step(now, cycle);
    }

    fn next_queued(&mut self, now: Ts, cycle: &mut Cycle) {
        if let Some(entry) = self.activity.tasks_mut().next_entry() {
            self.take_slot(now, entry, cycle);
        }
    }

    /// A slot whose frame is gone: every task queued on it is refused as closing.
    fn refuse_slot(&mut self, now: Ts, slot: TaskSlot, cycle: &mut Cycle) {
        let (current, queue) = slot.into_parts();
        if let Some(live) = current {
            self.refuse_closing(now, live.handle(), live.speech(), cycle);
        }
        self.refuse_queue(now, queue, cycle);
    }

    fn refuse_queue(&mut self, now: Ts, queue: VecDeque<TaskEntry>, cycle: &mut Cycle) {
        for entry in queue {
            self.refuse_closing(now, &entry.handle, entry.speech, cycle);
        }
    }

    /// The task is refused as closing: its awaiter reads `ACTIVITY_CLOSING` (unless it settled
    /// already), then whatever the entry held is released.
    fn refuse_closing(
        &mut self,
        now: Ts,
        handle: &TaskHandle<HostRef>,
        speech: Option<SpeechId>,
        cycle: &mut Cycle,
    ) {
        let _ = handle.fail(ToolError::plain(ACTIVITY_CLOSING));
        self.release_awaiter(now, handle, speech, cycle);
    }

    /// The close with a task live: cancel the innermost (`ToolError("AgentTask {id} is
    /// cancelled")`); `on_task_frame_popped` cancels the next. `false` when nothing is live.
    pub(crate) fn cancel_innermost_task(&mut self, now: Ts, cycle: &mut Cycle) -> bool {
        // Queued tasks never run: refused now, so their awaiters are released within the close.
        let queued = self.activity.tasks_mut().take_queue();
        self.refuse_queue(now, queued, cycle);
        match self.activity.tasks_mut().cancel_innermost() {
            None => false,
            // Already handing back (or entering with its settle seen): the unwind waits for it.
            Some(None) => true,
            // Entering: the close's `plan_exit` refuses the entry handoff; the wake handles the
            // settle.
            Some(Some(task)) => {
                self.on_task_wake(now, task, cycle);
                true
            }
        }
    }

    /// The unwind's bound passed: every pushed frame popped without a handoff, awaiters released
    /// with the closing error. The driving frame stays for the close's `on_exit`.
    pub(crate) fn abandon_tasks(&mut self, now: Ts, cycle: &mut Cycle) {
        let slot = self.activity.tasks_mut().take_driving_slot();
        self.refuse_slot(now, slot, cycle);
        while let Some(slot) = self.activity.tasks_mut().pop_parent() {
            let (current, queue) = slot.into_parts();
            if let Some(live) = current {
                let _ = live.handle().fail(ToolError::plain(ACTIVITY_CLOSING));
                self.finish_task(now, live, cycle);
            }
            self.refuse_queue(now, queue, cycle);
        }
    }

    /// The loop is ending: no awaiter may be left parked on a handle that will never move.
    pub(crate) fn release_tasks_at_end(&mut self) {
        let mut cycle = self.ports();
        let now = self.shared.time.now();
        self.abandon_tasks(now, &mut cycle);
    }

    /// A `TaskCommand` the loop will not perform: the awaiter is released with the closing error.
    pub(crate) fn refuse_task_command(&mut self, now: Ts, cmd: TaskCommand, cycle: &mut Cycle) {
        match cmd {
            TaskCommand::Enter(entry) => {
                let _ = entry.handle.begin();
                // Never held or blocked: the command was refused before either.
                self.refuse_closing(now, &entry.handle, None, cycle);
            }
        }
    }

    /// `_add_drain_blocked_tasks`, counted per speech; the scheduling task hears the first only.
    fn block_speech(&mut self, _now: Ts, speech: SpeechId, cycle: &mut Cycle) {
        if self.activity.tasks_mut().block(speech) {
            cycle.trace(SpeechStep::Blocked { speech, blocked: true });
            self.set_drain_blocked(speech, true, cycle);
        }
    }

    /// The matching release: the last one puts the parked speech back as the current speech.
    fn unblock_speech(&mut self, _now: Ts, speech: SpeechId, cycle: &mut Cycle) {
        if self.activity.tasks_mut().unblock(speech) {
            cycle.trace(SpeechStep::Blocked { speech, blocked: false });
            self.set_drain_blocked(speech, false, cycle);
        }
    }

    /// Whether `speech`'s frame is one `session.current_agent` no longer names: a host change in
    /// flight or landed. A same-agent update is not a change of agent.
    pub(crate) fn speech_frame_left(&self, speech: SpeechId) -> bool {
        let Some(frame) = self.activity.speech_frame(speech) else { return false };
        if let Some(agent) = self.activity.hooks().host_handoff_agent() {
            return agent != frame;
        }
        frame != self.activity.agent() && !self.activity.tasks().serves_frame(frame)
    }
}

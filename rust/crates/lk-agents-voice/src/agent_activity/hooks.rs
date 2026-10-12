//! The host's hooks (`on_enter`, `on_exit`, `on_user_turn_completed`, ...) and [`Hooks`], which
//! also holds the frame-change queue, the host's `drain()` and the DTMF collection.

use crate::agent::TurnEdit;
use crate::agent_session::cycle::Cycle;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::cycle::TurnTimer;
use crate::agent_session::{
    AgentUpdate, DrainRefused, Driver, HookCall, HookId, HookKind, HookOutcome, HostEvent,
    NewActivity, PreviousActivity,
};
use crate::audio_recognition::PendingTurn;
use crate::run_result::Owned;
use crate::trace::{CoreStep, HandoffStep, HookEnd, InterruptCause};
use lk_agents_abi::Ts;
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

/// How long the driver waits on a hook whose answer it needs when the host named no bound.
pub const HOOK_BOUND: Duration = crate::speech_handle::INTERRUPTION_TIMEOUT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandoffId(pub u64);

pub(crate) use hooks::{DrainJoin, DrainPoll, HandoffPoll, Hooks, SwapExit};

/// The hooks and frame changes' state, behind its methods: the driver reaches none of its fields.
mod hooks {
    use super::*;

    struct Handoff {
        id: HandoffId,
        update: Box<AgentUpdate>,
        phase: HandoffPhase,
    }

    enum HandoffPhase {
        /// The outgoing agent may still speak.
        Exiting { hook: HookId },
        /// `PauseScheduling{cause: Handoff}` in effect; `settled` answers when every speech not
        /// blocked on an inline task has finished, or closes when the pause was refused.
        Draining { settled: tokio::sync::oneshot::Receiver<()> },
    }

    /// The reference's `_on_exit_task`, created once by whichever of `drain()`, a handoff or the
    /// close gets there first. A frame paused under an inline task keeps its own on
    /// `Hooks::exit_stack`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    enum FrameExit {
        #[default]
        Running,
        Exiting(HookId),
        Exited,
    }

    impl FrameExit {
        pub(crate) fn running(self) -> Option<HookId> {
            match self {
                FrameExit::Exiting(hook) => Some(hook),
                FrameExit::Running | FrameExit::Exited => None,
            }
        }
    }

    /// A host `drain()` of the driving frame (`FrameCommand::Drain`): its `on_exit`, then its
    /// pause.
    struct FrameDrain {
        phase: DrainPhase,
        /// Every caller of `drain()` still waiting; answered together.
        waiters: Vec<tokio::sync::oneshot::Sender<Result<(), DrainRefused>>>,
    }

    enum DrainPhase {
        /// The frame may still speak its goodbye.
        Exiting { hook: HookId },
        /// A frame change in flight is leaving the frame: callers wait for its swap, as
        /// `activity.drain()` waits on the `_lock` the change holds.
        Joined,
        /// `settled` answers when nothing is outstanding, or closes when refused.
        Pausing { settled: tokio::sync::oneshot::Receiver<()> },
        /// The pause settled with a speech blocked on an inline task still open: wait for its end.
        /// (The reference's drain deadlocks here on the task's `pause()`.)
        Settling,
        /// Exited and paused: the frame stays, idle, until a close or a new frame.
        Drained,
    }

    /// `GetDtmfTask`'s event half while it runs.
    struct DtmfRun {
        collector: crate::workflows::builtins::dtmf::DtmfCollector,
        done: Option<tokio::sync::oneshot::Sender<crate::workflows::builtins::dtmf::DtmfOutcome>>,
        /// Whether `TurnTimer::DtmfDebounce` is armed, so a cancel is never a cancel of nothing.
        armed: bool,
        /// The confirm arm's read-back: the collector's guard is held until this speech is done.
        confirm: Option<SpeechId>,
    }

    /// One place in the order frame changes were asked for, which is the order they land in (the
    /// reference chains each `update_agent` behind the one before). A tool's handoff holds its
    /// place while the host describes the agent, so a host `Update` meanwhile waits behind it.
    enum Queued {
        /// Waiting for the change in flight (`_update_activity_task(old_task)` awaits it).
        Ready { id: HandoffId, update: Box<AgentUpdate> },
        /// A tool's handoff not yet described; given up when the describe fails or times out.
        Describing(HookId),
    }

    pub(crate) struct PendingHook {
        kind: HookKind,
        /// When the driver stops waiting; `None` when nothing waits.
        deadline: Option<Ts>,
        /// A `DescribeAgent` asked by the driving frame. A tool's handoff blocks the frame whose
        /// tool returned the agent, not one a swap installs meanwhile (cleared at the swap).
        describing: bool,
    }

    type DrainWaiter = tokio::sync::oneshot::Sender<Result<(), DrainRefused>>;

    #[derive(Default)]
    pub(crate) struct Hooks {
        pending: BTreeMap<HookId, PendingHook>,
        next: u64,
        next_handoff: u64,
        handoff: Option<Handoff>,
        queue: VecDeque<Queued>,
        /// A user turn decided inside the window, delivered to the arriving frame once.
        parked_turn: Option<ParkedTurn>,
        /// The host's `drain()` of the driving frame, from the request until a new frame replaces
        /// it.
        drain: Option<FrameDrain>,
        /// `drain()` callers whose frame was left: answered `Ok` once no speech is open -- and, for
        /// a frame an inline task paused (`Some(hook)`), once its `on_exit` has ended.
        left_drains: Vec<(Option<HookId>, tokio::sync::oneshot::Sender<Result<(), DrainRefused>>)>,
        dtmf: Option<DtmfRun>,
        exit: FrameExit,
        exit_stack: Vec<FrameExit>,
    }

    /// A turn kept for the arriving frame. `edits: None` -- the arriving frame runs its own hook;
    /// `edits: Some` -- the hook already answered, so the turn is decided exactly once.
    struct ParkedTurn {
        info: crate::audio_recognition::EndOfTurnInfo,
        edits: Option<Vec<TurnEdit>>,
    }

    impl Hooks {
        pub(crate) fn mint(&mut self) -> HookId {
            self.next += 1;
            HookId(self.next)
        }

        pub(crate) fn mint_handoff(&mut self) -> HandoffId {
            self.next_handoff += 1;
            HandoffId(self.next_handoff)
        }

        /// The hook is no longer waited on (answered elsewhere, timed out, or its asker gone).
        pub(crate) fn forget(&mut self, hook: HookId) {
            self.pending.remove(&hook);
        }

        /// The pending hooks of one kind, in id order.
        pub(crate) fn pending_of(&self, kind: HookKind) -> Vec<HookId> {
            self.pending.iter().filter(|(_, h)| h.kind == kind).map(|(id, _)| *id).collect()
        }

        pub(crate) fn is_pending(&self, hook: HookId) -> bool {
            self.pending.contains_key(&hook)
        }

        /// A turn parked for the arriving frame is dropped: no frame arrives for it.
        pub(crate) fn clear_parked_turn(&mut self) {
            self.parked_turn = None;
        }

        pub(crate) fn has_parked_turn(&self) -> bool {
            self.parked_turn.is_some()
        }

        /// The agent a host (non-task) frame change in flight installs.
        pub(crate) fn host_handoff_agent(&self) -> Option<&str> {
            self.handoff
                .as_ref()
                .filter(|h| h.update.task.is_none())
                .map(|h| h.update.agent.as_str())
        }

        /// The frame change in flight, taken: its id and its update.
        pub(crate) fn take_handoff(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
            self.handoff.take().map(|Handoff { id, update, phase: _ }| (id, update))
        }

        /// Every host `drain()` caller still waiting, taken: a left frame's first, then the driving
        /// frame's (whose drain is forgotten).
        pub(crate) fn take_drain_waiters(&mut self) -> Vec<DrainWaiter> {
            let mut waiters: Vec<_> =
                std::mem::take(&mut self.left_drains).into_iter().map(|(_, w)| w).collect();
            if let Some(drain) = self.drain.take() {
                waiters.extend(drain.waiters);
            }
            waiters
        }

        /// The DTMF collection's answer still owed, taken with the run (no guard release: the loop
        /// is over).
        pub(crate) fn take_dtmf_done(
            &mut self,
        ) -> Option<tokio::sync::oneshot::Sender<crate::workflows::builtins::dtmf::DtmfOutcome>>
        {
            self.dtmf.take().and_then(|run| run.done)
        }

        /// The driving frame's `on_exit` while it runs.
        pub(crate) fn exit_hook(&self) -> Option<HookId> {
            self.exit.running()
        }

        pub(crate) fn next_deadline(&self) -> Option<Ts> {
            self.pending.values().filter_map(|h| h.deadline).min()
        }

        /// `(hook, issue)`: minted and to be issued while the frame runs, waited for while it
        /// exits, none once it ran.
        pub(crate) fn frame_exit(&mut self) -> (Option<HookId>, bool) {
            match self.exit {
                FrameExit::Running => (Some(self.mint()), true),
                FrameExit::Exiting(hook) if self.pending.contains_key(&hook) => (Some(hook), false),
                FrameExit::Exiting(_) | FrameExit::Exited => (None, false),
            }
        }

        /// A non-task frame change in flight (not merely queued: a task exit begun now goes ahead
        /// of a queued one, and the host's change lands on top).
        pub(crate) fn host_handoff_pending(&self) -> bool {
            self.handoff.as_ref().is_some_and(|h| h.update.task.is_none())
        }

        /// The drain answers: its callers. `None` leaves the frame `Drained`; a refusal forgets it.
        pub(crate) fn finish_drain(
            &mut self,
            refused: Option<DrainRefused>,
        ) -> Option<Vec<DrainWaiter>> {
            let drain = self.drain.as_mut()?;
            let waiters = std::mem::take(&mut drain.waiters);
            match refused {
                None => drain.phase = DrainPhase::Drained,
                Some(_) => self.drain = None,
            }
            Some(waiters)
        }

        /// The DTMF collection stops: its guard released; whether its debounce was armed, and the
        /// answer still owed.
        pub(crate) fn stop_dtmf(
            &mut self,
        ) -> Option<(
            bool,
            Option<tokio::sync::oneshot::Sender<crate::workflows::builtins::dtmf::DtmfOutcome>>,
        )> {
            let mut run = self.dtmf.take()?;
            // The host reading `dtmf_reply_running()` must not see a guard nobody will release.
            run.collector.finish_reply();
            Some((run.armed, run.done))
        }

        /// Callers of a left frame's drain waiting on nothing but quiet (none while any waits on an
        /// `on_exit`), taken.
        pub(crate) fn take_left_drains(&mut self) -> Vec<DrainWaiter> {
            if !self.left_drains.iter().any(|(h, _)| h.is_none()) {
                return Vec::new();
            }
            let (ready, waiting) =
                std::mem::take(&mut self.left_drains).into_iter().partition(|(h, _)| h.is_none());
            self.left_drains = waiting;
            ready.into_iter().map(|(_, done)| done).collect()
        }

        /// The transition window is open: from a tool's handoff request (`_new_turns_blocked` is
        /// set synchronously in `update_agent`) through the swap. A describe pending across a swap
        /// opens no window for the arriving frame.
        pub(crate) fn in_window(&self) -> bool {
            self.handoff.is_some() || self.pending.values().any(|h| h.describing)
        }
        // --- the hooks the host is asked ---------------------------------------------------------

        /// The driving frame's `on_exit` was issued: running while a host answers it, else over.
        pub(crate) fn exit_issued(&mut self, hook: HookId, hosted: bool) {
            self.exit = if hosted { FrameExit::Exiting(hook) } else { FrameExit::Exited };
        }

        /// The host was asked `hook`: waited on until `deadline` (`None`: nothing waits).
        pub(crate) fn wait_for(&mut self, hook: HookId, kind: HookKind, deadline: Option<Ts>) {
            let describing = kind == HookKind::DescribeAgent;
            self.pending.insert(hook, PendingHook { kind, deadline, describing });
        }

        /// The hook is answered (or given up on): its kind, if it was waited on.
        pub(crate) fn finish(&mut self, hook: HookId) -> Option<HookKind> {
            self.pending.remove(&hook).map(|h| h.kind)
        }

        /// The hooks whose bound passed by `now`, in id order.
        pub(crate) fn due(&self, now: Ts) -> Vec<HookId> {
            (self.pending.iter())
                .filter(|(_, h)| h.deadline.is_some_and(|d| d <= now))
                .map(|(id, _)| *id)
                .collect()
        }

        /// Every hook still waited on, taken, in id order.
        pub(crate) fn take_pending(&mut self) -> Vec<(HookId, HookKind)> {
            std::mem::take(&mut self.pending).into_iter().map(|(id, h)| (id, h.kind)).collect()
        }

        // --- frame changes, in the order they were asked for -------------------------------------

        /// A tool's handoff holds its place in the order while the host describes the agent.
        pub(crate) fn hold_place(&mut self, hook: HookId) {
            self.queue.push_back(Queued::Describing(hook));
        }

        /// Where the describe `hook` holds its place.
        pub(crate) fn place_of(&self, hook: HookId) -> Option<usize> {
            self.queue.iter().position(|q| matches!(q, Queued::Describing(h) if *h == hook))
        }

        /// The described change takes the place its describe held.
        pub(crate) fn fill_place(&mut self, place: usize, id: HandoffId, update: Box<AgentUpdate>) {
            self.queue[place] = Queued::Ready { id, update };
        }

        pub(crate) fn give_up_place(&mut self, place: usize) {
            self.queue.remove(place);
        }

        /// The describe `hook` will not be answered: its place goes.
        pub(crate) fn drop_place(&mut self, hook: HookId) {
            self.queue.retain(|q| !matches!(q, Queued::Describing(h) if *h == hook));
        }

        /// A change asked now waits behind the one in flight or a place held before it; a task's
        /// resume goes ahead of what is queued.
        pub(crate) fn must_wait(&self, resuming_task: bool) -> bool {
            self.handoff.is_some() || (!self.queue.is_empty() && !resuming_task)
        }

        pub(crate) fn queue_handoff(&mut self, id: HandoffId, update: Box<AgentUpdate>) {
            self.queue.push_back(Queued::Ready { id, update });
        }

        /// The next change to begin: none while one is in flight or a describe holds the head.
        pub(crate) fn next_handoff(&mut self) -> Option<(HandoffId, Box<AgentUpdate>)> {
            if self.handoff.is_some() || !matches!(self.queue.front(), Some(Queued::Ready { .. })) {
                return None;
            }
            match self.queue.pop_front() {
                Some(Queued::Ready { id, update }) => Some((id, update)),
                Some(Queued::Describing(_)) | None => None,
            }
        }

        /// The head of the order, taken: a change, or a describe's held place (`Some(None)`).
        pub(crate) fn pop_queued(&mut self) -> Option<Option<(HandoffId, Box<AgentUpdate>)>> {
            self.queue.pop_front().map(|q| match q {
                Queued::Ready { id, update } => Some((id, update)),
                Queued::Describing(_) => None,
            })
        }

        /// The change begins, waiting on the leaving frame's `on_exit` `hook` (`HookId(0)`: none).
        pub(crate) fn start_handoff(
            &mut self,
            id: HandoffId,
            update: Box<AgentUpdate>,
            hook: HookId,
        ) {
            self.handoff = Some(Handoff { id, update, phase: HandoffPhase::Exiting { hook } });
        }

        /// The change in flight waits on the `on_exit` `hook`.
        pub(crate) fn handoff_waits_on(&self, hook: HookId) -> bool {
            matches!(
                self.handoff,
                Some(Handoff { phase: HandoffPhase::Exiting { hook: h }, .. }) if h == hook
            )
        }

        /// The change drains; `settled` answers when the pause does. `false` with none in flight.
        pub(crate) fn handoff_draining(
            &mut self,
            settled: tokio::sync::oneshot::Receiver<()>,
        ) -> bool {
            let Some(handoff) = &mut self.handoff else { return false };
            handoff.phase = HandoffPhase::Draining { settled };
            true
        }

        /// Has the change's drain settled? A refused pause takes the change with it.
        pub(crate) fn poll_handoff(&mut self) -> HandoffPoll {
            let Some(handoff) = &mut self.handoff else { return HandoffPoll::Waiting };
            let HandoffPhase::Draining { settled } = &mut handoff.phase else {
                return HandoffPoll::Waiting;
            };
            match settled.try_recv() {
                Ok(()) => HandoffPoll::Settled,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => HandoffPoll::Waiting,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => match self.handoff.take()
                {
                    Some(Handoff { id, update, phase: _ }) => HandoffPoll::Refused(id, update),
                    None => HandoffPoll::Waiting,
                },
            }
        }

        /// The swap: a paused frame keeps its exit beneath the arriving one and gets it back on
        /// resume; the arriving frame is not blocked by a describe the left one asked.
        pub(crate) fn frame_swapped(&mut self, previous: PreviousActivity, next: NewActivity) {
            match (previous, next) {
                (PreviousActivity::Pause, NewActivity::Start) => {
                    let paused = std::mem::take(&mut self.exit);
                    self.exit_stack.push(paused);
                }
                (PreviousActivity::Close, NewActivity::Start) => self.exit = FrameExit::Running,
                (_, NewActivity::Resume) => self.exit = self.exit_stack.pop().unwrap_or_default(),
            }
            self.pending.values_mut().for_each(|h| h.describing = false);
        }

        // --- the user turn parked across a transition --------------------------------------------

        /// The parked turn, taken, with the hook's edits if they were already answered.
        pub(crate) fn take_parked_turn(
            &mut self,
        ) -> Option<(crate::audio_recognition::EndOfTurnInfo, Option<Vec<TurnEdit>>)> {
            self.parked_turn.take().map(|ParkedTurn { info, edits }| (info, edits))
        }

        pub(crate) fn park_turn(
            &mut self,
            info: crate::audio_recognition::EndOfTurnInfo,
            edits: Option<Vec<TurnEdit>>,
        ) {
            self.parked_turn = Some(ParkedTurn { info, edits });
        }

        // --- the host's `drain()` of the driving frame -------------------------------------------

        /// `drain()` callers join the frame's drain; what the driver does next.
        pub(crate) fn join_drain(&mut self, waiters: Vec<DrainWaiter>, quiet: bool) -> DrainJoin {
            let changing = self.handoff.is_some();
            match &mut self.drain {
                // Already exited and paused: answered now, or once refused speeches are released;
                // under a frame change, at the swap.
                Some(drain @ FrameDrain { phase: DrainPhase::Drained, .. }) if changing => {
                    drain.waiters.extend(waiters);
                    DrainJoin::Waiting
                }
                Some(FrameDrain { phase: DrainPhase::Drained, .. }) if quiet => {
                    DrainJoin::AnswerNow(waiters)
                }
                Some(drain @ FrameDrain { phase: DrainPhase::Drained, .. }) => {
                    drain.phase = DrainPhase::Settling;
                    drain.waiters.extend(waiters);
                    DrainJoin::Waiting
                }
                Some(drain) => {
                    drain.waiters.extend(waiters);
                    DrainJoin::Waiting
                }
                // A frame change in flight: this caller waits for its `on_exit` and pause.
                None if changing => {
                    let exit_hook = self.exit.running();
                    self.drain = Some(FrameDrain { phase: DrainPhase::Joined, waiters });
                    DrainJoin::JoinedChange { exit_hook }
                }
                // Resumed after its `on_exit` ran (`Exited`): only the pause is left.
                None => {
                    let (exit_hook, issue) = self.frame_exit();
                    let hook = exit_hook.unwrap_or(HookId(0));
                    self.drain = Some(FrameDrain { phase: DrainPhase::Exiting { hook }, waiters });
                    DrainJoin::Exiting { exit_hook, issue }
                }
            }
        }

        /// The drain waits on its frame's `on_exit` still.
        pub(crate) fn drain_exiting(&self) -> bool {
            self.drain.as_ref().is_some_and(|d| matches!(d.phase, DrainPhase::Exiting { .. }))
        }

        /// The drain pauses scheduling; `settled` answers when the pause does. `false` with no
        /// drain.
        pub(crate) fn drain_pausing(
            &mut self,
            settled: tokio::sync::oneshot::Receiver<()>,
        ) -> bool {
            let Some(drain) = &mut self.drain else { return false };
            drain.phase = DrainPhase::Pausing { settled };
            true
        }

        /// Has the drain's pause settled, with nothing open (`quiet`)?
        pub(crate) fn poll_drain(&mut self, quiet: bool) -> DrainPoll {
            let Some(drain) = &mut self.drain else { return DrainPoll::Waiting };
            let settled = match &mut drain.phase {
                DrainPhase::Settling if quiet => return DrainPoll::Settled,
                DrainPhase::Pausing { settled } => settled,
                DrainPhase::Exiting { .. }
                | DrainPhase::Joined
                | DrainPhase::Settling
                | DrainPhase::Drained => return DrainPoll::Waiting,
            };
            match settled.try_recv() {
                Ok(()) if quiet => DrainPoll::Settled,
                Ok(()) => {
                    drain.phase = DrainPhase::Settling;
                    DrainPoll::Waiting
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => DrainPoll::Waiting,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => DrainPoll::Refused,
            }
        }

        /// The `on_exit` `hook` ended: every frame's exit it was is over. Whether the drain waited
        /// on it.
        pub(crate) fn exit_ended(&mut self, hook: HookId) -> bool {
            for exit in std::iter::once(&mut self.exit).chain(self.exit_stack.iter_mut()) {
                if *exit == FrameExit::Exiting(hook) {
                    *exit = FrameExit::Exited;
                }
            }
            matches!(
                self.drain,
                Some(FrameDrain { phase: DrainPhase::Exiting { hook: h }, .. }) if h == hook
            )
        }

        /// Callers bound to the paused frame whose `on_exit` `hook` ended wait only for quiet now.
        pub(crate) fn unbind_left_drains(&mut self, hook: HookId) {
            for (waits_on, _) in &mut self.left_drains {
                if *waits_on == Some(hook) {
                    *waits_on = None;
                }
            }
        }

        /// The driving frame's drain, taken at the swap: its callers, if any wait.
        pub(crate) fn take_drain_at_swap(&mut self) -> Option<Vec<DrainWaiter>> {
            let drain = self.drain.take()?;
            (!drain.waiters.is_empty()).then_some(drain.waiters)
        }

        /// What the left frame's drain callers still wait for at the swap.
        pub(crate) fn exit_at_swap(&self, previous: PreviousActivity) -> SwapExit {
            match (previous, self.exit) {
                (PreviousActivity::Close, _) | (PreviousActivity::Pause, FrameExit::Exited) => {
                    SwapExit::Nothing
                }
                (PreviousActivity::Pause, FrameExit::Exiting(hook)) => SwapExit::Running(hook),
                (PreviousActivity::Pause, FrameExit::Running) => SwapExit::Issue,
            }
        }

        /// The left frame's callers, waiting on `waiting_on` (or only for quiet).
        pub(crate) fn leave_drains(
            &mut self,
            waiting_on: Option<HookId>,
            waiters: Vec<DrainWaiter>,
        ) {
            self.left_drains.extend(waiters.into_iter().map(|w| (waiting_on, w)));
        }

        /// A joined change ended without a swap: the callers of the drain that joined it, taken.
        pub(crate) fn take_joined_drain(&mut self) -> Option<Vec<DrainWaiter>> {
            if self.handoff.is_some()
                || !matches!(self.drain, Some(FrameDrain { phase: DrainPhase::Joined, .. }))
            {
                return None;
            }
            self.drain.take().map(|d| d.waiters)
        }

        // --- `GetDtmfTask` -----------------------------------------------------------------------

        pub(crate) fn start_dtmf(
            &mut self,
            collector: crate::workflows::builtins::dtmf::DtmfCollector,
            done: tokio::sync::oneshot::Sender<crate::workflows::builtins::dtmf::DtmfOutcome>,
        ) {
            self.dtmf = Some(DtmfRun { collector, done: Some(done), armed: false, confirm: None });
        }

        pub(crate) fn dtmf_running(&self) -> bool {
            self.dtmf.is_some()
        }

        pub(crate) fn dtmf_digit(
            &mut self,
            digit: &str,
            now: Ts,
        ) -> Option<crate::workflows::builtins::dtmf::DtmfAction> {
            self.dtmf.as_mut().map(|run| run.collector.on_sip_dtmf(digit, now))
        }

        /// A state edge the collection reads; the read-back's end releases its guard.
        pub(crate) fn dtmf_event(
            &mut self,
            ev: &crate::events::SessionEvent,
            now: Ts,
        ) -> Option<crate::workflows::builtins::dtmf::DtmfAction> {
            use crate::events::SessionEvent;
            let run = self.dtmf.as_mut()?;
            match ev {
                SessionEvent::AgentStateChanged { new, .. } => {
                    Some(run.collector.on_agent_state_changed(new.as_str(), now))
                }
                SessionEvent::UserStateChanged { new, .. } => {
                    Some(run.collector.on_user_state_changed(new.as_str(), now))
                }
                SessionEvent::SpeechDone { speech, .. } if run.confirm == Some(*speech) => {
                    run.confirm = None;
                    run.collector.finish_reply();
                    None
                }
                _ => None,
            }
        }

        /// The debounce is armed. `false` with no collection.
        pub(crate) fn dtmf_schedule(&mut self) -> bool {
            let Some(run) = &mut self.dtmf else { return false };
            run.armed = true;
            true
        }

        /// The debounce is disarmed: whether it was armed; `None` with no collection.
        pub(crate) fn dtmf_disarm(&mut self) -> Option<bool> {
            self.dtmf.as_mut().map(|run| std::mem::take(&mut run.armed))
        }

        /// The debounce fired. `false` with no collection.
        pub(crate) fn dtmf_fired(&mut self) -> bool {
            let Some(run) = &mut self.dtmf else { return false };
            run.armed = false;
            true
        }

        pub(crate) fn dtmf_generate_reply(
            &mut self,
        ) -> Option<crate::workflows::builtins::dtmf::DtmfReply> {
            self.dtmf.as_mut().map(|run| run.collector.generate_reply())
        }

        /// The collection's answer: the guard released, the task told once.
        pub(crate) fn dtmf_finish(
            &mut self,
            outcome: crate::workflows::builtins::dtmf::DtmfOutcome,
        ) {
            let Some(run) = &mut self.dtmf else { return };
            run.collector.finish_reply();
            // `if not self.done()`: a task already completed takes no second result.
            if let Some(done) = run.done.take() {
                let _ = done.send(outcome);
            }
        }

        /// The confirm arm's read-back holds the guard until `speech` is done; a read-back refused
        /// at creation (`None`) releases it now.
        pub(crate) fn dtmf_confirming(&mut self, speech: Option<SpeechId>) {
            let Some(run) = &mut self.dtmf else { return };
            match speech {
                None => run.collector.finish_reply(),
                Some(speech) => run.confirm = Some(speech),
            }
        }
    }

    /// Where a frame change's drain is.
    pub(crate) enum HandoffPoll {
        Waiting,
        /// The swap is due now.
        Settled,
        /// The pause was refused: the change, taken.
        Refused(HandoffId, Box<AgentUpdate>),
    }

    /// What `drain()` callers joining the frame's drain need next.
    pub(crate) enum DrainJoin {
        Waiting,
        /// Drained and nothing open: answered now.
        AnswerNow(Vec<DrainWaiter>),
        /// A frame change in flight: they wait for its swap.
        JoinedChange {
            exit_hook: Option<HookId>,
        },
        /// A new drain: the frame's `on_exit` (issued by this drain when `issue`), then the pause.
        Exiting {
            exit_hook: Option<HookId>,
            issue: bool,
        },
    }

    /// Where the drain's pause is.
    pub(crate) enum DrainPoll {
        Waiting,
        /// Settled with nothing open: answered `Ok`.
        Settled,
        /// Refused: answered with the reason.
        Refused,
    }

    /// What the left frame's drain callers wait for at the swap.
    pub(crate) enum SwapExit {
        Nothing,
        Running(HookId),
        /// The paused frame's `on_exit` has not run: issued now.
        Issue,
    }
}

impl Driver {
    /// Journalled where decided, sent to the host when one runs hooks, else finished in place
    /// (`false`). `deadline` bounds the wait when an answer is needed.
    pub(crate) fn issue_hook(
        &mut self,
        now: Ts,
        call: HookCall,
        wait: Option<Duration>,
        cycle: &mut Cycle,
    ) -> bool {
        let id = call.id();
        let kind = call.kind();
        let (agent, speech, turn, at_tail) = match &call {
            HookCall::OnEnter { agent, .. } | HookCall::OnExit { agent, .. } => {
                (Some(agent.clone()), None, None, None)
            }
            HookCall::OnUserTurnCompleted { turn, .. } => (None, None, Some(*turn), None),
            HookCall::DescribeAgent { speech, .. } => (None, Some(*speech), None, None),
            HookCall::RenderToolReply { at_tail, .. } => (None, None, None, Some(*at_tail)),
            HookCall::FillerSource { .. } => (None, None, None, None),
            HookCall::ToolsExecuted { speech, .. } => (None, Some(*speech), None, None),
        };
        cycle.trace(HandoffStep::HookIssued { hook: id, kind, agent, speech, turn, at_tail });
        let hosted =
            self.activity.hook_bound().is_some() && self.events.send(HostEvent::Hook(call)).is_ok();
        // Every `OnExit` is the driving frame's: its exit is now running, or -- no host -- over.
        if kind == HookKind::OnExit {
            self.activity.hooks_mut().exit_issued(id, hosted);
        }
        if !hosted {
            cycle.trace(HandoffStep::HookFinished { hook: id, kind, how: HookEnd::NoHost });
            return false;
        }
        let deadline = wait.map(|d| now.saturating_add(d));
        self.activity.hooks_mut().wait_for(id, kind, deadline);
        // An open run owns the frame's `on_enter` and the describe a tool's handoff waits on.
        match kind {
            HookKind::OnEnter | HookKind::DescribeAgent => self.run_own(Owned::Hook(id), cycle),
            HookKind::OnExit
            | HookKind::OnUserTurnCompleted
            | HookKind::RenderToolReply
            | HookKind::FillerSource
            | HookKind::ToolsExecuted => {}
        }
        true
    }

    pub(crate) fn hook_wait(&self) -> Duration {
        self.activity.hook_bound().unwrap_or(HOOK_BOUND)
    }

    /// Nothing waits on it.
    pub(crate) fn issue_on_enter(&mut self, now: Ts, cycle: &mut Cycle) {
        let id = self.activity.hooks_mut().mint();
        let agent = self.activity.agent().to_owned();
        self.issue_hook(now, HookCall::OnEnter { id, agent }, None, cycle);
    }

    /// The turn parks on it until answer or bound.
    pub(crate) fn issue_turn_hook(
        &mut self,
        now: Ts,
        info: crate::audio_recognition::EndOfTurnInfo,
        bound: Duration,
        cycle: &mut Cycle,
    ) {
        let id = self.activity.hooks_mut().mint();
        let call = HookCall::OnUserTurnCompleted {
            id,
            turn: info.turn_id,
            new_transcript: info.new_transcript.clone(),
            transcript_confidence: info.transcript_confidence,
        };
        if self.issue_hook(now, call, None, cycle) {
            self.activity.park_turn(PendingTurn {
                info,
                issued_at: now,
                deadline: now.saturating_add(bound),
                hook: Some(id),
            });
        } else {
            self.finish_turn(now, info, Vec::new(), cycle);
        }
    }

    pub(crate) fn on_hook_done(
        &mut self,
        now: Ts,
        id: HookId,
        outcome: HookOutcome,
        cycle: &mut Cycle,
    ) {
        let how = match &outcome {
            HookOutcome::Failed => HookEnd::Failed,
            HookOutcome::Turn(edits) => HookEnd::Returned { edits: edits.len() },
            HookOutcome::Returned
            | HookOutcome::Described(_)
            | HookOutcome::Rendered(_)
            | HookOutcome::Filler { .. }
            | HookOutcome::ToolsExecuted { .. } => HookEnd::Returned { edits: 0 },
        };
        // An id nobody waits on (a late answer) is dropped.
        let Some(kind) = self.finish_hook(id, how, cycle) else { return };
        self.hook_ended(now, kind, id, outcome, cycle);
    }

    /// The hook left `pending`, journalled and let go by the open run; `None` if not pending.
    fn finish_hook(&mut self, id: HookId, how: HookEnd, cycle: &mut Cycle) -> Option<HookKind> {
        let kind = self.activity.hooks_mut().finish(id)?;
        cycle.trace(HandoffStep::HookFinished { hook: id, kind, how });
        self.run_release(Owned::Hook(id), cycle);
        Some(kind)
    }

    /// What waited on the hook goes on. `Failed` is also a hook that will not be answered (its
    /// bound passed): an exit proceeds, a describe is dropped, a render or a filler source falls
    /// back to its default. A turn hook's bound is `pending_turn`'s. A mismatched answer counts
    /// as unanswered.
    fn hook_ended(
        &mut self,
        now: Ts,
        kind: HookKind,
        id: HookId,
        outcome: HookOutcome,
        cycle: &mut Cycle,
    ) {
        use HookOutcome::{Described, Failed, Filler, Rendered, Returned, ToolsExecuted, Turn};
        let fits = matches!(
            (kind, &outcome),
            (HookKind::OnExit | HookKind::OnEnter, Returned | Failed)
                | (HookKind::DescribeAgent, Described(_) | Failed)
                | (HookKind::OnUserTurnCompleted, Turn(_) | Returned | Failed)
                | (HookKind::RenderToolReply, Rendered(_) | Returned | Failed)
                | (HookKind::FillerSource, Filler { .. } | Returned | Failed)
                | (HookKind::ToolsExecuted, ToolsExecuted { .. } | Returned | Failed)
        );
        // The sequence must not hang on a malformed answer: it goes on as if unanswered.
        if !fits {
            cycle.trace(CoreStep::PortFailure("hook outcome does not match the call it answers"));
        }
        match kind {
            // The handoff waiting on it drains, a host drain pauses, the close moves on.
            HookKind::OnExit => {
                if self.activity.hooks().handoff_waits_on(id) {
                    self.drain_handoff(now, cycle);
                }
                self.drain_exit_done(now, id, cycle);
                self.close_exit_done(now, id, cycle);
            }
            HookKind::OnEnter => {}
            HookKind::DescribeAgent => match outcome {
                Described(update) => self.describe_answered(now, id, update, cycle),
                Failed | Returned | Turn(_) | Rendered(_) | Filler { .. } | ToolsExecuted { .. } => {
                    self.drop_describe(now, id, cycle)
                }
            },
            // The reference: "error occurred during on_user_turn_completed" -> return, no reply.
            HookKind::OnUserTurnCompleted => {
                let edits = match outcome {
                    Turn(edits) => edits,
                    Failed if fits => vec![TurnEdit::StopResponse],
                    Failed | Returned | Described(_) | Rendered(_) | Filler { .. }
                    | ToolsExecuted { .. } => Vec::new(),
                };
                // Matched to its parked turn; a late answer is dropped.
                if let Some(pending) = self.activity.take_pending_turn_of(id) {
                    self.finish_turn(now, pending.finished(now), edits, cycle)
                }
            }
            // The deferred tool reply's instructions, or the defaults when rendering failed.
            HookKind::RenderToolReply => {
                let text = match outcome {
                    Rendered(text) => Some(text),
                    Failed | Returned | Turn(_) | Described(_) | Filler { .. }
                    | ToolsExecuted { .. } => None,
                };
                self.on_tool_reply_rendered(now, id, text, cycle)
            }
            // The filler's callable: text, a speech it made itself, or nothing; raising skips.
            HookKind::FillerSource => {
                let (text, spoke) = match outcome {
                    Filler { text, spoke } => (text, spoke),
                    Failed | Returned | Turn(_) | Described(_) | Rendered(_)
                    | ToolsExecuted { .. } => (None, false),
                };
                self.on_filler_source(now, id, text, spoke, cycle)
            }
            // What the handlers left; unanswered, the event stands as issued.
            HookKind::ToolsExecuted => {
                let left = match outcome {
                    ToolsExecuted { reply_required, handoff_required } => {
                        Some((reply_required, handoff_required))
                    }
                    Failed | Returned | Turn(_) | Described(_) | Rendered(_) | Filler { .. } => None,
                };
                self.on_tools_executed_handled(id, left, cycle)
            }
        }
    }

    /// Expired hook bounds: an unreturned `on_exit` lets the drain go on; an unanswered describe
    /// drops the handoff.
    pub(crate) fn expire_hooks(&mut self, now: Ts, cycle: &mut Cycle) {
        for id in self.activity.hooks().due(now) {
            if let Some(kind) = self.finish_hook(id, HookEnd::TimedOut, cycle) {
                self.hook_ended(now, kind, id, HookOutcome::Failed, cycle);
            }
        }
    }

    pub(crate) fn on_dtmf(
        &mut self,
        now: Ts,
        cmd: crate::agent_session::DtmfCommand,
        cycle: &mut Cycle,
    ) {
        use crate::agent_session::DtmfCommand;
        match cmd {
            DtmfCommand::Start { collector, done } => {
                self.dtmf_stop(cycle);
                self.activity.hooks_mut().start_dtmf(collector, done);
            }
            DtmfCommand::Digit { digit } => {
                let Some(action) = self.activity.hooks_mut().dtmf_digit(&digit, now) else {
                    return;
                };
                self.dtmf_perform(now, action, cycle);
            }
            DtmfCommand::Stop => self.dtmf_stop(cycle),
        }
    }

    /// Debounce cancelled, an owed answer is `Stopped`.
    pub(crate) fn dtmf_stop(&mut self, cycle: &mut Cycle) {
        let Some((armed, done)) = self.activity.hooks_mut().stop_dtmf() else { return };
        if armed {
            cycle.machine().cancel(TurnTimer::DtmfDebounce);
        }
        if let Some(done) = done {
            let _ = done.send(crate::workflows::builtins::dtmf::DtmfOutcome::Stopped);
        }
    }

    /// A state edge on its way to the host: a speaker cancels the debounce, both quiet restarts
    /// it, and the read-back's end releases the guard.
    pub(crate) fn dtmf_on_event(
        &mut self,
        now: Ts,
        ev: &crate::events::SessionEvent,
        cycle: &mut Cycle,
    ) {
        let Some(action) = self.activity.hooks_mut().dtmf_event(ev, now) else { return };
        self.dtmf_perform(now, action, cycle);
    }

    fn dtmf_perform(
        &mut self,
        now: Ts,
        action: crate::workflows::builtins::dtmf::DtmfAction,
        cycle: &mut Cycle,
    ) {
        use crate::workflows::builtins::dtmf::DtmfAction;
        match action {
            DtmfAction::Idle | DtmfAction::Ignored => {}
            // `schedule()` replaces a pending timer: one arm, which the timer set replaces.
            DtmfAction::Schedule { at } => {
                if self.activity.hooks_mut().dtmf_schedule() {
                    cycle.machine().arm(TurnTimer::DtmfDebounce, at);
                }
            }
            // `Debounced.run()`: the pending timer cancelled, the body now.
            DtmfAction::Cancel | DtmfAction::RunNow => {
                let Some(armed) = self.activity.hooks_mut().dtmf_disarm() else { return };
                if armed {
                    cycle.machine().cancel(TurnTimer::DtmfDebounce);
                }
                if matches!(action, DtmfAction::RunNow) {
                    self.dtmf_reply(now, cycle);
                }
            }
        }
    }

    pub(crate) fn on_dtmf_debounce(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.activity.hooks_mut().dtmf_fired() {
            self.dtmf_reply(now, cycle);
        }
    }

    /// `_generate_dtmf_reply`: interrupt, then the collector's decision; the confirm arm holds the
    /// guard until the read-back is done.
    fn dtmf_reply(&mut self, now: Ts, cycle: &mut Cycle) {
        use crate::workflows::builtins::dtmf::{DtmfOutcome, DtmfReply};
        if !self.activity.hooks().dtmf_running() {
            return;
        }
        // `self.session.interrupt()`: the host's non-forced interrupt, from the task's code.
        cycle.trace(crate::trace::InterruptStep::HostInterrupt { speech: None, force: false });
        if let Ok(speeches) = self.interrupt_all(now, false, InterruptCause::Host, cycle) {
            let waiter = crate::agent_activity::pause::InterruptWaiter { speeches, done: None };
            self.activity.add_interrupt_waiter(waiter);
        }
        let Some(reply) = self.activity.hooks_mut().dtmf_generate_reply() else { return };
        let outcome = match reply {
            DtmfReply::Incomplete(error) => DtmfOutcome::Incomplete { message: error.message },
            DtmfReply::Complete(result) => DtmfOutcome::Complete(result),
            DtmfReply::Confirm { user_input } => return self.dtmf_confirm(now, user_input, cycle),
        };
        self.activity.hooks_mut().dtmf_finish(outcome);
    }

    /// The confirm arm's read-back, which holds the collector's guard until it is done.
    fn dtmf_confirm(&mut self, now: Ts, user_input: String, cycle: &mut Cycle) {
        let h = self.generate_reply_speech(
            now,
            Some(user_input),
            None,
            None,
            None,
            None,
            None,
            false,
            cycle,
        );
        // Refused at creation (a draining or closing queue): already over.
        let queued = self.activity.scheduling_task().contains(h.id()).then_some(h.id());
        self.activity.hooks_mut().dtmf_confirming(queued);
    }
}

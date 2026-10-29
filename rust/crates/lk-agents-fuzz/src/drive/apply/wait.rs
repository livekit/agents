//! Awaiting answers and advancing the session scheduler to completion.

use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

use super::super::executor::Polled;
use super::*;
use crate::record::{Task, TaskEnd};

// ==============================================================================================
// Waiting
// ==============================================================================================

/// How long one awaited thing gets, in virtual milliseconds. Not a production SLA: every fake
/// answers the moment it is released, so anything still pending after this is stuck.
pub const BUDGET_MS: u64 = 30_000;

/// When a close that has not finished never will: a virtual day, where the longest close the
/// phases allow is under a minute. The paused clock jumps to the next armed timer whenever nothing
/// can run, so a session left idle with nothing armed to wake it gets here at once, for free.
const NEVER_MS: u64 = 86_400_000;

/// How many times closing will wait for a fresh batch of tasks before giving up on the driver.
const MAX_JOIN_ROUNDS: u32 = 100;

impl Run {
    pub(super) async fn await_answer(&mut self, nth: u32) -> bool {
        let Some(mut waiting) = self.pending.remove(&nth) else { return false };
        match self.drive(Some(self.budget()), waiting.recv()).await {
            Some(settled) => {
                self.record_answer(nth, settled);
                true
            }
            None => {
                self.pending.insert(nth, waiting);
                false
            }
        }
    }

    pub(super) async fn await_speech(&mut self, nth: u32) -> bool {
        let Some(resource) = self.record.resource(Kind::Speech, nth) else { return false };
        let Id::Speech(speech) = resource.id() else { return false };
        let Some(handle) = self.handles.get(&speech.0).cloned() else { return false };
        if self.api.api().is_err() {
            return false;
        }
        // `await handle.wait_for_generation()`, `await handle`, then what a program reads off it: a
        // speech the run saw created, so its waits are the speech's own.
        let waited = handle.clone();
        let done = self.drive(Some(self.budget()), async move {
            let _ = waited.wait_for_latest_generation().await;
            waited.wait_for_playout().await
        });
        if done.await.is_none() {
            return false;
        }
        let _ =
            (HostSpeech::id(&*handle), handle.num_steps(), handle.chat_items(), handle.scheduled());
        let _ = (HostSpeech::interrupted(&*handle), HostSpeech::allow_interruptions(&*handle));
        let _ = HostSpeech::exception(&*handle);

        true
    }

    /// Before a close nothing obliges the session to exit, so that wait is an ordinary one. Once a
    /// close is begun it is waited out however long it takes: its phases are each bounded, and
    /// virtual time is free.
    ///
    /// Once the loop has exited, the host's `aclose()` and `close_io()` are waited out too, as a
    /// host awaits them; each of their waits is bounded on the session clock.
    pub(super) async fn await_exit(&mut self) -> bool {
        let wait = if self.record.close_begun().is_some() { NEVER_MS } else { BUDGET_MS };
        let until = self.time.now().saturating_add(Duration::from_millis(wait));
        self.drive(Some(until), std::future::pending::<()>()).await;
        let exited = self.session.is_none();
        // The player goes with the session (`BackgroundAudioPlayer.aclose()`).
        if exited && let Some(player) = self.background.take() {
            player.0.close().await;
        }
        if exited && let Some(worker) = self.avatar.take() {
            worker.close().await;
            for (at, step) in worker.drain() {
                self.record.record_avatar(at, step);
            }
        }
        if exited {
            if let Some(closing) = self.closing.take() {
                let until = self.time.now().saturating_add(Duration::from_millis(NEVER_MS));
                tokio::select! {
                    () = closing => {}
                    () = self.time.wait_until(until) => {}
                }
            }
            self.record.enter_phase(Phase::Closed);
        }
        exited
    }

    /// How long one awaited thing may take from now.
    pub(in crate::drive) fn budget(&self) -> Ts {
        self.time.now().saturating_add(Duration::from_millis(BUDGET_MS))
    }

    /// Step the session until `done` resolves, virtual time reaches `until`, or the session is
    /// gone. What its sources sent is pending; the loop is polled whenever it was woken, so a due
    /// wake runs before any same-instant input; then the policy chooses between the next pending
    /// event, handed to the loop and polled at once, and a scheduling round for the session's
    /// ready tasks. Once the decision core stopped, what is pending goes to the loop's close, and
    /// the session is dropped once that returned, so what waited on it is released. `None` unless
    /// `done` resolved.
    pub(in crate::drive) async fn drive<T>(
        &mut self,
        until: Option<Ts>,
        done: impl std::future::Future<Output = T>,
    ) -> Option<T> {
        let mut done = std::pin::pin!(done);
        let gate = self.tasks.gate.clone();
        let _shut = gate.as_ref().map(|gate| gate.shut());
        // Iterations at one virtual instant: a poll of the loop, a step or a task's turn each.
        let mut instant = (self.time.now(), 0u32);
        loop {
            tokio::select! {
                biased;
                value = &mut done => return Some(value),
                () = std::future::ready(()) => {}
            }
            let now = self.time.now();
            instant = if now == instant.0 { (now, instant.1 + 1) } else { (now, 0) };
            self.busiest_instant = self.busiest_instant.max(instant.1);
            let counted = (instant.1 >= NO_PROGRESS_STEPS).then(|| {
                format!(
                    "{NO_PROGRESS_STEPS} steps of the session at {} ms without virtual time moving",
                    self.stamps.now().at_ms
                )
            });
            let observed = self.livelock.lock().unwrap().take();
            if let Some(detail) = observed.or(counted) {
                self.record.set_livelock(detail);
                self.session = None;
            }
            let Some(session) = &mut self.session else {
                // The host's loop outlives the session's: what is ready still runs, so a task the
                // session cancelled on its way out is gone before the case sends anything else.
                // Bounded, as the session's own instants are: a task left spinning never stops.
                if let Some(gate) = &gate {
                    gate.open();
                }
                for _ in 0..NO_PROGRESS_STEPS {
                    if !tasks_runnable() {
                        break;
                    }
                    one_task_turn().await;
                }
                return None;
            };
            if session.serving() {
                while let Ok(event) = session.intake.try_recv() {
                    self.queue.push(event);
                }
                while let Ok(frame) = session.mic.try_recv() {
                    self.queue.push_frame(frame);
                }
            }
            if self.poll_session() {
                continue;
            }
            let Some(session) = &mut self.session else { continue };
            let ready = gate.as_ref().map(|gate| gate.ready()).unwrap_or_default();
            match self.queue.next(tasks_runnable(), &ready) {
                Some(Next::Event(event)) => {
                    session.deliver(event);
                    self.poll_session();
                    continue;
                }
                Some(Next::Frames(frames)) => {
                    session.deliver_frames(frames);
                    self.poll_session();
                    continue;
                }
                Some(Next::Tasks) => {
                    one_task_turn().await;
                    continue;
                }
                Some(Next::Task(id)) => {
                    let gate = gate.as_ref().expect("a bridged task's turn has a gate");
                    self.grant(gate, id).await;
                    continue;
                }
                None => {}
            }
            if until.is_some_and(|until| now >= until) {
                return None;
            }
            let woken = async {
                match &gate {
                    Some(gate) => gate.woken().await,
                    None => std::future::pending().await,
                }
            };
            let (serving, loop_woken) = (session.serving(), session.woken());
            tokio::select! {
                biased;
                value = &mut done => return Some(value),
                () = woken => {}
                () = loop_woken => {}
                () = self.time.wait_until_opt(until) => {}
                Some(frame) = session.mic.recv(), if serving => self.queue.push_frame(frame),
                Some(event) = session.intake.recv(), if serving => self.queue.push(event),
            }
        }
    }

    /// A bridged task's turn: its tokio task is woken; the pool's spurious polls may come first.
    async fn grant(&self, gate: &super::super::queue::Gate, id: u64) {
        gate.grant(id);
        for _ in 0..GRANT_TURNS {
            one_task_turn().await;
            if !gate.granting() {
                break;
            }
        }
    }

    /// One poll of the session's loop, when something woke it. A panic is a finding about the
    /// session, recorded, and the end of it: it is dropped. The decision core's stop hands what is
    /// still pending to the loop's close; the close's return records the tasks it had to cancel
    /// and ends the session. `true` when the session moved to its close or ended.
    fn poll_session(&mut self) -> bool {
        let Some(session) = &mut self.session else { return false };
        let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.poll()));
        // The observer unwound out of a livelocked step: the loop reports it, not as a panic.
        if self.livelock.lock().unwrap().is_some() {
            self.session = None;
            return true;
        }
        match polled {
            Err(panic) => {
                self.record.set_driver_panic(panic_message(&*panic));
                self.session = None;
                true
            }
            Ok(Polled::Running) => false,
            Ok(Polled::Stopped(runtime)) => {
                let (_unread, held) = self.queue.drain();
                session.close(runtime, held);
                true
            }
            Ok(Polled::Closed(late)) => {
                self.record.record_late_tasks(self.stamps.now(), late);
                self.session = None;
                true
            }
        }
    }

    /// Await every retained task, and record how each ended.
    ///
    /// A task may spawn another before finishing, so this loops -- and because the work is the
    /// product's, the loop can fail to terminate. It is bounded for the same reason the release
    /// loop is: an unbounded loop here is a hang, not a slow run, and a driver that keeps
    /// spawning while closing is a result worth reporting rather than waiting on forever.
    ///
    /// Each batch is raced against a deadline a budget away, the session stepped meanwhile as its
    /// own loop would be: a session whose close never finished, frozen, would hold every task it
    /// ends. The clock is paused and moves only
    /// when nothing can run, so the deadline passes exactly when the tasks still pending can never
    /// finish: those are recorded as stuck and aborted, and the run completes.
    pub async fn join_tasks(&mut self) {
        // A livelocked session may have left a task spinning, which no deadline would outrun: the
        // livelock is the finding, and the tasks are aborted with the run.
        if self.record.livelock().is_some() {
            return;
        }
        for _ in 0..MAX_JOIN_ROUNDS {
            let batch = std::mem::take(&mut *self.tasks.spawned.lock().unwrap());
            if batch.is_empty() {
                return;
            }
            let deadline = self.budget();
            for Spawned { at, site, mut task } in batch {
                let joined = match self.drive(Some(deadline), &mut task).await {
                    Some(joined) => Some(joined),
                    // The session has exited: only its tasks are left to run.
                    None => tokio::select! {
                        joined = &mut task => Some(joined),
                        () = self.time.wait_until(deadline) => None,
                    },
                };
                let end = match joined {
                    Some(Err(error)) if error.is_panic() => {
                        TaskEnd::Panicked(panic_message(&*error.into_panic()))
                    }
                    Some(_) => TaskEnd::Finished,
                    None => {
                        // Nothing left that could finish it: no step pending, no wake, no task.
                        let enabled =
                            !self.queue.idle() || self.session.is_some() || tasks_runnable();
                        if !enabled && self.deadlock.is_none() {
                            self.deadlock = Some(format!(
                                "the task spawned at {} ({site}) never finished, with nothing \
                                 pending, no wake armed and no task runnable at {} ms",
                                at.at_ms,
                                self.stamps.now().at_ms
                            ));
                        }
                        task.abort();
                        TaskEnd::Stuck
                    }
                };
                self.record.record_task(Task { spawned: at, site, end });
            }
        }
        // Still spawning after every round: what is left is not going to finish either.
        for Spawned { at, site, task } in std::mem::take(&mut *self.tasks.spawned.lock().unwrap()) {
            let end = if task.is_finished() { TaskEnd::Finished } else { TaskEnd::Stuck };
            task.abort();
            self.record.record_task(Task { spawned: at, site, end });
        }
    }
}

/// What a panic said, when it said it with a string.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

/// Turns a granted task's poll may wait behind: the pool's own, each a poll that returns at once.
const GRANT_TURNS: usize = 64;

/// Let the runtime poll one task, then come back: the run's future wakes itself and, with an event
/// interval of one, is polled again after the next task. `yield_now` defers its wake until every
/// ready task has run, which would never let the driver step between two of them.
async fn one_task_turn() {
    let mut turned = false;
    std::future::poll_fn(|cx| {
        if std::mem::replace(&mut turned, true) {
            return std::task::Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    })
    .await
}

/// Whether a task the driver spawned is ready to run: a turn would move it.
fn tasks_runnable() -> bool {
    let metrics = tokio::runtime::Handle::current().metrics();
    metrics.global_queue_depth() + metrics.worker_local_queue_depth(0) > 0
}

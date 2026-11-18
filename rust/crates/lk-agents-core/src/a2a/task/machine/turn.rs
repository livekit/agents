use super::PARTITION_POLL_PAUSE;
use std::time::Duration;

use lk_agents_abi::storage::thread::{ItemRow, TaskRow};
use lk_agents_abi::storage::{Ask, StorageError, TaskState};

use super::{
    Completion, Finish, Handle, HandlerEvent, LeaseAfterCommit, Operation, Path, Phase, RunEnd,
    RunStep, Stage, TASKS_IN_CONTEXT, TaskMachine,
};
use crate::a2a::{
    HandlerOutput, Message, Policy, Settle, Stop, TurnIds, answer_pending, batch_to_chat_message,
    settle,
};

// Every storage read and the handler's stream are separate steps, so other requests and clients
// interleave with each of them.

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Step {
    /// Reading the pending question; `named` while it is the one the message continues.
    Pending {
        named: bool,
    },
    Prepare,
    Handle,
    Next,
    Checkpoint,
    /// Reading the answer row after the turn stopped.
    AnswerRow,
    Settle,
}

/// One turn's inputs and what it learned so far.
#[derive(Debug, Clone)]
pub(super) struct Turning {
    batch: Vec<Message>,
    /// One received task per message, in order.
    task_ids: Vec<String>,
    /// A batch inbox's drain, under the runner's kept lease.
    pub(super) drain: bool,
    /// Unix seconds: after everything the run loaded.
    at: f64,
    pub(super) answer_task: String,
    /// The question of the task the message continues.
    continued: Option<Ask>,
    /// The pending question's settlement, staged before the handler runs.
    staged: Stage,
    pub(super) user_message: Option<ItemRow>,
    pub(super) stop: Option<Stop>,
    pub(super) output: Option<HandlerOutput>,
}

impl TaskMachine {
    /// Starts the turn over `batch`. A `concurrent` run answers no pending question: it holds no
    /// lease to claim one and may only append to the history. Under `live` only an explicit
    /// `taskId` continues a task: an unaddressed message is new work with its own task, and a
    /// waiting question stays pending.
    pub(super) fn start_turn(
        &mut self,
        batch: Vec<Message>,
        task_ids: Vec<String>,
        path: Path,
        now: Duration,
    ) -> Operation {
        let drain = path == Path::Drain;
        let wall = self.epoch + now.as_secs_f64();
        // two steps past the loaded history, so an item placed one step before `at` follows it
        let after_history = self.history_end + 2.0 * crate::a2a::outcome::ITEM_ORDER_STEP;
        let at = if self.history_end >= wall { after_history } else { wall };
        let named = batch.last().and_then(|m| m.task_id.clone());
        let answer_task = task_ids.last().cloned().unwrap_or_default();
        self.turn = Some(Turning {
            batch,
            task_ids,
            drain,
            at,
            answer_task,
            continued: None,
            staged: Stage::default(),
            user_message: None,
            stop: None,
            output: None,
        });
        if self.policy == Policy::Concurrent || (self.policy == Policy::Live && named.is_none()) {
            return self.prepare(None);
        }
        self.phase = Phase::Turn(Step::Pending { named: named.is_some() });
        Operation::ReadPending(named)
    }

    fn turning(&mut self) -> &mut Turning {
        self.turn.as_mut().expect("a turn is running")
    }

    pub(super) fn turn_step(
        &mut self,
        step: Step,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let operation = match (step, completion) {
            (Step::Next, Completion::Event(HandlerEvent::Managed(report)))
                if self.policy == Policy::Live =>
            {
                return self.managed_turn(*report, now);
            }
            // under live a message continues only the task it names, never another question
            (Step::Pending { named: true }, Completion::Task(None))
                if self.policy == Policy::Live =>
            {
                self.prepare(None)
            }
            // the named question was answered meanwhile: the message answers the one waiting now
            (Step::Pending { named: true }, Completion::Task(None)) => {
                self.phase = Phase::Turn(Step::Pending { named: false });
                Operation::ReadPending(None)
            }
            (Step::Pending { .. }, Completion::Task(pending)) => self.prepare(pending.as_ref()),
            (Step::Prepare, Completion::Done) => self.open_handler(),
            (Step::Handle, Completion::Done) => {
                self.phase = Phase::Turn(Step::Next);
                if self.policy == Policy::Live {
                    Operation::NextAll { should_yield: None, pause: PARTITION_POLL_PAUSE }
                } else {
                    Operation::Next(None)
                }
            }
            (Step::Next, Completion::Inbox(rows)) if self.policy == Policy::Live => {
                return self.poll_live_tasks(rows, true);
            }
            (Step::Next, Completion::Event(HandlerEvent::LiveCheckpoint { request_id })) => {
                return self.live_checkpoint(request_id, false);
            }
            (Step::Next, Completion::Event(HandlerEvent::LiveOutput { request_id, output })) => {
                return self.live_output(request_id, output, now);
            }
            (Step::Next, Completion::Event(HandlerEvent::Checkpoint)) => {
                if self.turning().stop.is_some() {
                    Operation::Next(Some(true))
                } else {
                    let turn = self.turning();
                    let answer = turn.answer_task.clone();
                    let receipt = turn.task_ids.last().filter(|t| **t != answer).cloned();
                    let tasks = std::iter::once(answer).chain(receipt).collect();
                    self.phase = Phase::Turn(Step::Checkpoint);
                    Operation::Checkpoint(tasks)
                }
            }
            (Step::Checkpoint, Completion::Checkpoint { statuses }) => {
                let canceled = statuses.contains(&Some(TaskState::Canceled));
                let turn = self.turning();
                turn.stop = canceled.then_some(Stop::ClientCanceled);
                let should_yield = turn.stop.is_some();
                self.phase = Phase::Turn(Step::Next);
                if self.policy == Policy::Live {
                    Operation::NextAll {
                        should_yield: Some(should_yield),
                        pause: PARTITION_POLL_PAUSE,
                    }
                } else {
                    Operation::Next(Some(should_yield))
                }
            }
            (Step::Next, Completion::Event(HandlerEvent::Output(mut output))) => {
                if self.policy == Policy::Live && self.live.managed {
                    // Once attached, only the serialized native Session can replace this
                    // state. An opaque handler still owns its explicit scope/app writes.
                    output.state = None;
                }
                if self.policy == Policy::Live
                    && let Some(id) = self
                        .turn
                        .as_ref()
                        .and_then(|turn| turn.user_message.as_ref())
                        .map(|item| item.id.clone())
                {
                    self.live.turns.remove(&id);
                }
                let turn = self.turning();
                let stopped = turn.stop.is_some() || output.interrupted;
                turn.output = Some(*output);
                if stopped {
                    self.phase = Phase::Turn(Step::AnswerRow);
                    Operation::ReadTask(self.turning().answer_task.clone())
                } else {
                    return self.settle_turn(false, now);
                }
            }
            (Step::AnswerRow, Completion::Task(row)) => {
                let canceled = row.is_some_and(|r| r.status == TaskState::Canceled);
                return self.settle_turn(canceled, now);
            }
            (Step::Settle, Completion::Done) => {
                if self.policy == Policy::Live {
                    return Ok(Some(self.commit_live_turn()));
                }
                let turn = self.turn.take().expect("a turn is running");
                if turn.drain {
                    self.drain_commit()
                } else {
                    self.phase = Phase::Run { path: Path::Main, step: RunStep::Commit };
                    Operation::Commit {
                        answer_task: Some(turn.answer_task),
                        lease: LeaseAfterCommit::Release,
                        drained_id: None,
                    }
                }
            }
            (step, completion) => {
                return Err(Finish::Error(StorageError::Invalid(format!(
                    "unexpected A2A turn outcome {completion:?} in {step:?}"
                ))));
            }
        };
        Ok(Some(operation))
    }

    /// Stage transport task correlation. Human response admission belongs to the owner.
    fn prepare(&mut self, pending: Option<&TaskRow>) -> Operation {
        let turn = self.turning();
        let last = turn.batch.last().expect("a turn has a message");
        let received = turn.task_ids.last().expect("a turn has a task").clone();
        let answer = answer_pending(pending, last, &received);
        turn.answer_task = answer.answer_task.clone();
        turn.continued = answer.continued;
        turn.staged = Stage { transitions: answer.transitions, ..Stage::default() };
        if turn.staged == Stage::default() {
            return self.open_handler();
        }
        let staged = std::mem::take(&mut turn.staged);
        self.phase = Phase::Turn(Step::Prepare);
        Operation::Stage(Box::new(staged))
    }

    fn open_handler(&mut self) -> Operation {
        let id = self.mint("item");
        let partition_key = self.partition_key.clone();
        let read_only = self.policy == Policy::Concurrent;
        let endpoint = if self.turn.as_ref().is_some_and(|turn| turn.drain) {
            self.drain_endpoint.clone()
        } else {
            self.endpoint.clone()
        };
        let live = self.policy == Policy::Live;
        let turn = self.turning();
        let message = batch_to_chat_message(&turn.batch, turn.at, id);
        turn.user_message = Some(message.clone());
        let saved = live.then(|| (message.id.clone(), turn.clone()));
        let handle = Handle {
            endpoint,
            context_scope_id: format!("root:{partition_key}"),
            partition_key,
            read_only,
            recent_tasks: TASKS_IN_CONTEXT,
            message,
            batch: turn.batch.clone(),
            task_id: turn.answer_task.clone(),
            policy: self.policy,
        };
        if let Some((id, turn)) = saved {
            self.live.turns.insert(id, turn);
        }
        self.phase = Phase::Turn(Step::Handle);
        Operation::Handle(Box::new(handle))
    }

    /// What the handler's output makes of the rows and the history, staged with the run's end.
    pub(super) fn settle_turn(
        &mut self,
        answer_canceled: bool,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let ids = TurnIds {
            assistant_item: self.mint("item"),
            status_message: self.mint("msg"),
            artifact: self.mint("art"),
        };
        let partition_question = self.policy != Policy::Concurrent;
        let live = self.policy == Policy::Live;
        let own_task = self.task_id.clone();
        let turn = self.turning();
        let out = turn.output.take().unwrap_or_default();
        let settled = settle(Settle {
            batch: &turn.batch,
            task_ids: &turn.task_ids,
            answer_task: &turn.answer_task,
            user_message: if live {
                turn.user_message.clone().expect("the live handler was opened")
            } else {
                turn.user_message.take().expect("the handler was opened")
            },
            out: &out,
            stop: turn.stop,
            answer_canceled,
            continued: turn.continued.as_ref(),
            partition_question,
            now: turn.at,
            ids,
        });
        let settled = match settled {
            Ok(settled) => settled,
            Err(error) => return self.fail(error, now),
        };
        // A live continuation answers with the task it continues, even when this request
        // drained its own message: the receipt only points there.
        let own_continuation =
            live && turn.continued.is_some() && turn.task_ids.contains(&own_task);
        if !turn.drain || own_continuation {
            self.answer = Some((turn.answer_task.clone(), settled.output));
        }
        let HandlerOutput { state, mut scope_writes, mut app_writes, usage, error, .. } = out;
        if error.is_some() {
            // a raised handler's saved context and managed transactions roll back with its turn
            scope_writes.clear();
            app_writes.clear();
        }
        let stage = Box::new(Stage {
            edits: settled.edits,
            transitions: settled.transitions,
            events: settled.events,
            end: Some(RunEnd {
                outcome: settled.outcome,
                reason: settled.reason,
                state,
                scope_writes,
                app_writes,
                usage,
            }),
        });
        if live && self.live.managed && self.live.report.is_none() {
            return self.prepare_opaque_dialogue(stage);
        }
        self.phase = Phase::Turn(Step::Settle);
        Ok(Some(Operation::Stage(stage)))
    }
}

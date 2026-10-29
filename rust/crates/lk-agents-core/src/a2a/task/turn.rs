//! The turn inside the open run, one operation at a time: the pending question the message
//! answers, the response streams the answer task takes over, the handler's checkpoints and
//! output, the answer row's status when the turn stopped, and the settlement staged for the
//! commit. Every storage read and the handler's stream are separate steps, so other requests and
//! clients interleave with each of them.

use std::time::Duration;

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::{ItemRow, TaskPatch, TaskRow};

use super::{
    Completion, Finish, Handle, HandlerEvent, Operation, Path, Phase, RunEnd, RunStep, Stage,
    TASKS_IN_CONTEXT, TaskMachine,
};
use crate::a2a::{
    HandlerOutput, Message, Policy, Settle, Stop, TaskTransition, TurnIds, Verdict, answer_pending,
    batch_to_chat_message, checkpoint_stop, settle,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Step {
    /// Reading the pending question; `named` while it is the one the message continues.
    Pending {
        named: bool,
    },
    Streams,
    Prepare,
    Handle,
    Next,
    Checkpoint,
    /// Reading the answer row after the turn stopped.
    AnswerRow,
    Settle,
}

/// One turn's inputs and what it learned so far.
#[derive(Debug)]
pub(super) struct Turning {
    batch: Vec<Message>,
    /// One received task per message, in order.
    task_ids: Vec<String>,
    /// A batch inbox's drain, under the runner's kept lease.
    pub(super) drain: bool,
    interrupt: bool,
    /// Unix seconds: after everything the run loaded.
    at: f64,
    answer_task: String,
    answer_tool: Option<String>,
    /// The pending question's settlement, staged before the handler runs.
    staged: Stage,
    user_message: Option<ItemRow>,
    stop: Option<Stop>,
    output: Option<HandlerOutput>,
}

impl TaskMachine {
    /// Starts the turn over `batch`. A `concurrent` run answers no pending question: it holds no
    /// lease to claim one and may only append to the history.
    pub(super) fn start_turn(
        &mut self,
        batch: Vec<Message>,
        task_ids: Vec<String>,
        drain: bool,
        now: Duration,
    ) -> Operation {
        let wall = self.epoch + now.as_secs_f64();
        let at = if self.history_end >= wall { self.history_end + 2e-6 } else { wall };
        let named = batch.last().and_then(|m| m.task_id.clone());
        let answer_task = task_ids.last().cloned().unwrap_or_default();
        self.turn = Some(Turning {
            batch,
            task_ids,
            drain,
            interrupt: !drain && self.policy == Policy::Interrupt,
            at,
            answer_task,
            answer_tool: None,
            staged: Stage::default(),
            user_message: None,
            stop: None,
            output: None,
        });
        if self.policy == Policy::Concurrent {
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
            // the named question was answered meanwhile: the message answers the one waiting now
            (Step::Pending { named: true }, Completion::Task(None)) => {
                self.phase = Phase::Turn(Step::Pending { named: false });
                Operation::ReadPending(None)
            }
            (Step::Pending { .. }, Completion::Task(pending)) => self.prepare(pending.as_ref()),
            (Step::Streams, Completion::Tasks(rows)) => {
                let answer = self.turning().answer_task.clone();
                let streams: Vec<_> = rows
                    .into_iter()
                    .flat_map(|row| row.internal.map(|internal| internal.streams))
                    .flatten()
                    .collect();
                self.stage_streams(answer, streams)
            }
            (Step::Prepare, Completion::Done) => self.open_handler(),
            (Step::Handle, Completion::Done) => {
                self.phase = Phase::Turn(Step::Next);
                Operation::Next(None)
            }
            (Step::Next, Completion::Event(HandlerEvent::Checkpoint)) => {
                if self.turning().stop.is_some() {
                    Operation::Next(Some(Verdict::Yield))
                } else {
                    let turn = self.turning();
                    let answer = turn.answer_task.clone();
                    let receipt = turn.task_ids.last().filter(|t| **t != answer).cloned();
                    let tasks = std::iter::once(answer).chain(receipt).collect();
                    self.phase = Phase::Turn(Step::Checkpoint);
                    Operation::Checkpoint(tasks)
                }
            }
            (Step::Checkpoint, Completion::Checkpoint { holder, rows }) => {
                // another holder marked itself waiting, or a client canceled one of the tasks
                let other_waiting = rows.iter().any(|(waiter, _)| {
                    waiter.as_ref().is_some_and(|w| !w.is_empty() && *w != holder)
                });
                let canceled = rows.iter().any(|(_, status)| *status == Some(TaskState::Canceled));
                let turn = self.turning();
                turn.stop = checkpoint_stop(turn.interrupt, other_waiting, canceled);
                let verdict = if turn.stop.is_some() { Verdict::Yield } else { Verdict::Continue };
                self.phase = Phase::Turn(Step::Next);
                Operation::Next(Some(verdict))
            }
            (Step::Next, Completion::Event(HandlerEvent::Output(output))) => {
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
                let turn = self.turn.take().expect("a turn is running");
                if turn.drain {
                    self.drain_commit()
                } else {
                    self.phase = Phase::Run { path: Path::Main, step: RunStep::Commit };
                    Operation::Commit {
                        answer_task: Some(turn.answer_task),
                        keep_lease: false,
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

    /// The pending question's settlement; a drained batch's answer also takes over every
    /// receipt's response stream, a continued task this message's.
    fn prepare(&mut self, pending: Option<&TaskRow>) -> Operation {
        let item_id = self.mint("item");
        let stream = self.stream.clone();
        let turn = self.turning();
        let last = turn.batch.last().expect("a turn has a message");
        let received = turn.task_ids.last().expect("a turn has a task").clone();
        let answer = answer_pending(pending, last, &received, turn.at, item_id);
        turn.answer_task = answer.answer_task.clone();
        turn.answer_tool = answer.answer_tool;
        turn.staged = Stage {
            edits: answer.edit.into_iter().collect(),
            transitions: answer.transitions,
            ..Stage::default()
        };
        if turn.drain {
            let received = turn.task_ids.clone();
            self.phase = Phase::Turn(Step::Streams);
            return Operation::ReadTasks(received);
        }
        let streams = stream.filter(|_| answer.answer_task != received).into_iter().collect();
        self.stage_streams(answer.answer_task, streams)
    }

    fn stage_streams(
        &mut self,
        answer: String,
        streams: Vec<lk_agents_abi::storage::StreamInfo>,
    ) -> Operation {
        let turn = self.turning();
        if !streams.is_empty() {
            turn.staged.transitions.push(TaskTransition {
                task_id: answer,
                status: TaskState::Working,
                patch: TaskPatch { streams: Some(streams), ..TaskPatch::default() },
            });
        }
        if turn.staged == Stage::default() {
            return self.open_handler();
        }
        let staged = std::mem::take(&mut turn.staged);
        self.phase = Phase::Turn(Step::Prepare);
        Operation::Stage(Box::new(staged))
    }

    fn open_handler(&mut self) -> Operation {
        let id = self.mint("item");
        let lane = self.lane.clone();
        let read_only = self.policy == Policy::Concurrent;
        let turn = self.turning();
        let message = batch_to_chat_message(&turn.batch, turn.at, id);
        turn.user_message = Some(message.clone());
        let handle = Handle {
            context_scope_id: format!("root:{lane}"),
            lane,
            read_only,
            recent_tasks: TASKS_IN_CONTEXT,
            message,
            batch: turn.batch.clone(),
            task_id: turn.answer_task.clone(),
            answer_tool: turn.answer_tool.clone(),
        };
        self.phase = Phase::Turn(Step::Handle);
        Operation::Handle(Box::new(handle))
    }

    /// What the handler's output makes of the rows and the history, staged with the run's end.
    fn settle_turn(
        &mut self,
        answer_canceled: bool,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let ids = TurnIds {
            assistant_item: self.mint("item"),
            status_message: self.mint("msg"),
            artifact: self.mint("art"),
        };
        let lane_question = self.policy != Policy::Concurrent;
        let turn = self.turning();
        let out = turn.output.take().unwrap_or_default();
        let settled = settle(Settle {
            batch: &turn.batch,
            task_ids: &turn.task_ids,
            answer_task: &turn.answer_task,
            user_message: turn.user_message.take().expect("the handler was opened"),
            out: &out,
            stop: turn.stop,
            answer_canceled,
            lane_question,
            now: turn.at,
            ids,
        });
        let settled = match settled {
            Ok(settled) => settled,
            Err(error) => return self.fail(error, now),
        };
        if !turn.drain {
            self.answer = Some((turn.answer_task.clone(), settled.output));
        }
        let HandlerOutput { state, scope_writes, mut app_writes, usage, error, .. } = out;
        if error.is_some() {
            // a raised handler's managed transactions roll back with its turn
            app_writes.clear();
        }
        self.phase = Phase::Turn(Step::Settle);
        Ok(Some(Operation::Stage(Box::new(Stage {
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
        }))))
    }
}

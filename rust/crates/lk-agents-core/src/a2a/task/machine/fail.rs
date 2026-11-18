use std::time::Duration;

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::thread::{RunOutcome, TaskReason};

use super::{
    AfterAbandon, Answer, Completion, Finish, Operation, Path, Phase, RUNS_FOR_REJECT, RunStep,
    TaskMachine, answered,
};
use crate::a2a::{Policy, failed_message, python_reason};

/// Tasks being failed: a named one, which a pickup first takes to `working` unless it already
/// ended, or a dead holder's `working` rows but `skip`. `then` takes the `Done` that ends it.
#[derive(Debug)]
pub(super) struct Failing {
    reason: TaskReason,
    then: Phase,
    target: FailTarget,
}

#[derive(Debug)]
enum FailTarget {
    Named(String),
    Orphans { skip: Vec<String> },
}

impl TaskMachine {
    /// Fails `task_id` `reason` unless it already ended; `then` follows.
    pub(super) fn fail_task(
        &mut self,
        task_id: String,
        reason: TaskReason,
        then: Phase,
    ) -> Operation {
        self.phase = Phase::Failing;
        let target = FailTarget::Named(task_id.clone());
        self.failing = Some(Failing { reason, then, target });
        Operation::Pickup(task_id)
    }

    /// A run whose save could not be written: the inbox rows its drain picked up fail, then the
    /// message's own task, `storage-unavailable`; nobody else would fail them but `worker-lost`.
    /// They fail while the run still holds the partition: a holder taking it would find them `working`.
    pub(super) fn fail_run(&mut self) -> Operation {
        self.answer = None;
        let own = self.task_id.clone();
        let picked: Vec<String> =
            std::mem::take(&mut self.task_ids).into_iter().filter(|t| *t != own).collect();
        if picked.is_empty() {
            return self.fail_task(own, TaskReason::StorageUnavailable, Phase::RunFailed);
        }
        self.phase = Phase::Failing;
        self.failing = Some(Failing {
            reason: TaskReason::StorageUnavailable,
            then: Phase::FailRun,
            target: FailTarget::Orphans { skip: Vec::new() },
        });
        self.failed(TaskReason::StorageUnavailable, picked)
    }

    /// Fails a dead holder's `working` rows `worker-lost`, but `skip`; `then` follows.
    pub(super) fn fail_orphans(&mut self, mut skip: Vec<String>, then: Phase) -> Operation {
        if self.policy == Policy::Live {
            skip.extend(self.live.turns.values().map(|turn| turn.answer_task.clone()));
        }
        self.phase = Phase::Failing;
        let target = FailTarget::Orphans { skip };
        self.failing = Some(Failing { reason: TaskReason::WorkerLost, then, target });
        Operation::ReadWorking
    }

    /// Fails each of `task_ids`, each with its status message.
    pub(super) fn failed(&mut self, reason: TaskReason, task_ids: Vec<String>) -> Operation {
        let failed = task_ids
            .into_iter()
            .map(|task_id| {
                let message = failed_message(&task_id, reason, self.mint("msg"));
                (task_id, message)
            })
            .collect();
        Operation::FailOrphans { reason, failed }
    }

    pub(super) fn failing_step(
        &mut self,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let failing = self.failing.as_ref().expect("tasks are failing");
        let task_ids: Vec<String> = match (&failing.target, completion) {
            (FailTarget::Named(task_id), Completion::Picked(was)) => {
                was.map(|_| task_id.clone()).into_iter().collect()
            }
            (FailTarget::Orphans { skip }, Completion::Tasks(rows)) => rows
                .into_iter()
                .filter(|r| !answered(r))
                .map(|r| r.task_id)
                .filter(|t| !skip.contains(t))
                .collect(),
            (_, Completion::Done) => Vec::new(),
            (_, completion) => {
                return Err(Finish::Error(StorageError::Invalid(format!(
                    "unexpected A2A operation outcome {completion:?} while failing tasks"
                ))));
            }
        };
        if task_ids.is_empty() {
            self.phase = self.failing.take().expect("tasks are failing").then;
            return self.complete(Completion::Done, now);
        }
        let reason = failing.reason;
        Ok(Some(self.failed(reason, task_ids)))
    }

    pub(super) fn fail(
        &mut self,
        error: StorageError,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        if self.phase == Phase::Failing {
            let then = self.failing.take().expect("tasks are failing").then;
            if let Phase::Answer { .. } = then {
                // the failure was not stored: the client hears the database's error, never a
                // failure its task does not show
                return Err(Finish::Error(error));
            }
            // any other failure while failing tasks is the failure of the phase that started it
            self.phase = then;
        }
        let operation = match (self.phase, &error) {
            (
                Phase::Run { path: Path::Main, step: RunStep::Open },
                StorageError::Rejected { .. },
            ) if self.policy == Policy::Reject => {
                self.phase = Phase::Refused;
                Operation::ReadRuns(RUNS_FOR_REJECT)
            }
            (
                Phase::Run { path: Path::Main, step: RunStep::Open },
                StorageError::Rejected { .. },
            ) => {
                // the partition was not released within the wait
                let then = Phase::Answer { reason: Some(TaskReason::LeaseTimeout) };
                self.fail_task(self.task_id.clone(), TaskReason::LeaseTimeout, then)
            }
            (Phase::Release, StorageError::LeaseLost(_)) => {
                // the kept lease lapsed and another holder took the partition: it drains what is
                // pending, and this message's task is watched
                self.leased = false;
                self.observe()
            }
            (Phase::Observe | Phase::Inbox, StorageError::Unavailable(_)) => {
                // a durable task is watched through a transient failure, not given up on
                return Ok(self.poll(now));
            }
            (Phase::Run { step: RunStep::Open, .. }, StorageError::Rejected { .. }) => {
                return Ok(self.poll(now));
            }
            (Phase::Turn(_), _) => {
                // the turn's save cannot be written: the database is lost, or the handler's own
                // save was refused (a stale save, a field that is not JSON, a state too large).
                // The host passes only storage errors and cancellation through (its handler's
                // other exceptions are the output's `error`): a cancellation ends the run
                // `canceled`, and a thread that is gone has no task to fail.
                let fail_task =
                    !matches!(error, StorageError::Cancelled(_) | StorageError::NotFound(_));
                let outcome = match error {
                    StorageError::Cancelled(_) => RunOutcome::Canceled,
                    _ => RunOutcome::Failed,
                };
                let reason = python_reason(&error);
                self.error = Some(error);
                self.turn = None;
                if fail_task {
                    self.ended = Some((outcome, reason));
                    self.fail_run()
                } else {
                    self.phase = Phase::Abandon(AfterAbandon::RaiseError);
                    Operation::Abandon { outcome, reason }
                }
            }
            (Phase::Answer { reason }, _) => {
                // the answer's row is unreadable: the task as this worker left it
                let answer = Answer::row(self.answering(), None, reason);
                return Err(Finish::Answer(Box::new(Answer {
                    run_id: self.run_id.take(),
                    ..answer
                })));
            }
            // a commit: the turn's, a drain's, or the close of a run with nothing of its own
            (Phase::Run { step: RunStep::Commit, .. } | Phase::Closing(_), _) => {
                self.answer = None;
                match error {
                    // an outcome unknown at the deadline is not a failure: the message's task
                    // stays as it is; applied, the commit names its answer, and a holder that
                    // takes the partition fails it `worker-lost` if it never applied
                    StorageError::Uncertain(_) => self.drop_uncertain(),
                    // the save was refused, whatever the reason: the message's own task fails;
                    // its staged changes are absent, so a question it answered still waits
                    _ => {
                        self.error = Some(error);
                        self.fail_run()
                    }
                }
            }
            _ => return Err(Finish::Error(error)),
        };
        Ok(Some(operation))
    }
}

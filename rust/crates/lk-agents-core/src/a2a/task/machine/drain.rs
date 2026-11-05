//! The `batch` policy's inbox drain: one run under the kept lease picks up every pending
//! receipt, fails a dead holder's rows and answers the rest in one turn.

use std::time::Duration;

use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::REASON_WORKER_LOST;

use super::{Access, Completion, Finish, Operation, Path, Phase, RunStep, TaskMachine};
use crate::a2a::Message;

impl TaskMachine {
    pub(super) fn read_inbox(&mut self) -> Operation {
        self.phase = Phase::Inbox;
        Operation::ReadInbox
    }

    /// The batch inbox: release the kept lease when it is empty, else drain it in one run.
    pub(super) fn on_inbox(
        &mut self,
        completion: Completion,
        now: Duration,
    ) -> Result<Option<Operation>, Finish> {
        let rows = match completion {
            Completion::Inbox(rows) => rows,
            other => return Err(self.unexpected(other)),
        };
        if rows.is_empty() && !self.leased {
            return Ok(self.poll(now));
        }
        if rows.is_empty() {
            self.phase = Phase::Release;
            return Ok(Some(Operation::ReleaseDrained));
        }
        let run_id = self.mint("drain");
        let last = rows.last().expect("nonempty inbox");
        let task_id = last.task_id.clone();
        let channel = Some(crate::a2a::channel_of(&last.message));
        self.rows = rows;
        self.index = 0;
        self.batch.clear();
        self.task_ids.clear();
        self.phase = Phase::Run { path: Path::Drain, step: RunStep::Open };
        let access = if self.leased {
            Access::Kept
        } else {
            Access::Exclusive { wait: false, interrupt: false }
        };
        Ok(Some(Operation::Open { run_id, task_id, channel, access, pickup: false }))
    }

    /// The next inbox row to pick up; past the last, the batch's turn or the commit of nothing.
    pub(super) fn drain_pickup(&mut self) -> Operation {
        if self.index < self.rows.len() {
            self.phase = Phase::Run { path: Path::Drain, step: RunStep::Pickup };
            Operation::Pickup(self.rows[self.index].task_id.clone())
        } else if self.batch.is_empty() {
            self.drain_commit()
        } else {
            let skip = self.rows.iter().map(|r| r.task_id.clone()).collect();
            self.fail_orphans(skip, Phase::Run { path: Path::Drain, step: RunStep::Recover })
        }
    }

    /// One inbox row picked up: a dead holder's fails, a live one joins the batch.
    pub(super) fn on_drain_picked(
        &mut self,
        was: Option<TaskState>,
    ) -> Result<Option<Operation>, Finish> {
        let row = &self.rows[self.index];
        match was {
            Some(TaskState::Working) => {
                // a dead holder's: the pickup took it, it fails worker-lost
                let task_id = row.task_id.clone();
                self.phase = Phase::Run { path: Path::Drain, step: RunStep::Orphan };
                return Ok(Some(self.failed(REASON_WORKER_LOST, vec![task_id])));
            }
            Some(_) => {
                // the receipt's id stays apart from the client's continuation target
                let message = Message::from_a2a(&row.message, row.received_at as f64, None)
                    .map_err(Finish::Error)?;
                self.batch.push(message);
                self.task_ids.push(row.task_id.clone());
            }
            None => {}
        }
        self.index += 1;
        Ok(Some(self.drain_pickup()))
    }

    /// Commits the drain run, keeping the lease for what arrives meanwhile.
    pub(super) fn drain_commit(&mut self) -> Operation {
        self.phase = Phase::Run { path: Path::Drain, step: RunStep::Commit };
        Operation::Commit {
            answer_task: None,
            keep_lease: true,
            drained_id: self.rows.last().map(|r| r.id),
        }
    }

    /// The kept lease: released, the task is watched; else more arrived and the inbox drains.
    pub(super) fn on_released(
        &mut self,
        completion: Completion,
    ) -> Result<Option<Operation>, Finish> {
        let released = match completion {
            Completion::Released(released) => released,
            other => return Err(self.unexpected(other)),
        };
        self.leased = !released;
        Ok(Some(if released { self.observe() } else { self.read_inbox() }))
    }
}

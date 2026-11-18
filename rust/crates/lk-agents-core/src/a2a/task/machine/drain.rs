use std::time::Duration;

use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::TaskReason;

use super::{
    Access, Completion, Finish, LeaseAfterCommit, Operation, Path, Phase, Policy, RunStep,
    TaskMachine, TaskPickup, WhenHeld,
};
use crate::a2a::Message;

impl TaskMachine {
    /// The rows of the next turn, from the head of the inbox, in arrival order: the drain cursor
    /// only moves forward. One endpoint's rows share a turn. A live owner serves a live-channel
    /// row alone, its own task and result source, and a burst of other channels' rows as one
    /// merged turn, as `batch` would; any other run merges every row.
    pub(super) fn next_drain(
        &mut self,
        routed: Vec<lk_agents_abi::storage::thread::RoutedInbox>,
    ) -> Vec<lk_agents_abi::storage::thread::InboxRow> {
        self.drain_endpoint = routed.first().and_then(|entry| entry.endpoint.clone());
        let live = |row: &lk_agents_abi::storage::thread::InboxRow| {
            crate::a2a::resolve_policy(&self.policies, &crate::a2a::channel_of(&row.message))
                == Policy::Live
        };
        let head_live = self.policy == Policy::Live && routed.first().is_some_and(|e| live(&e.row));
        let mut rows = Vec::new();
        for entry in routed {
            if entry.endpoint != self.drain_endpoint
                || (head_live && !rows.is_empty())
                || (self.policy == Policy::Live && !head_live && live(&entry.row))
            {
                break;
            }
            rows.push(entry.row);
        }
        rows
    }

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
        let routed = match completion {
            Completion::Inbox(rows) => rows,
            other => return Err(self.unexpected(other)),
        };
        let rows = self.next_drain(routed);
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
        let access = if self.leased { Access::Kept } else { Access::Exclusive(WhenHeld::Reject) };
        Ok(Some(Operation::Open { run_id, task_id, channel, access, pickup: TaskPickup::Leave }))
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
                return Ok(Some(self.failed(TaskReason::WorkerLost, vec![task_id])));
            }
            Some(_) => {
                // the receipt's id stays apart from the client's continuation target
                let message =
                    Message::from_a2a(&row.message, row.received_at_ms as f64 / 1000.0, None)
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
            lease: LeaseAfterCommit::Keep,
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

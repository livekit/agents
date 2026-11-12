use crate::sql::{SqlValue, Table};
use crate::thread::db::{Acquire, Acquired, EventWrite, Receive};
use crate::thread::run::{
    LEASE_WAIT_TIMEOUT, LOOKUP_LIMIT, RunOutcome, RunSummary, SUBMIT_CHANNEL, TaskState,
};
use crate::thread::schema::*;
use crate::{PartitionAccess, StorageError, TaskPickup, TaskWrite, WhenHeld};
use lk_agents_abi::storage::journal::{JournalEvent, RunStarted};
use lk_agents_abi::storage::thread::Backoff;
use lk_agents_abi::usage::Usage;
use serde_json::{Map, Value};

use crate::thread::lease::{Holder, Lease, PendingLease};
use lk_agents_net::clock::unix_now_ms;

use super::{EventLog, Run, RunOptions, Tasks, Thread, db};
use crate::telemetry::{self, StorageWrite};
use lk_agents_abi::telemetry::health::RENEW_SLACK;
use std::ops::ControlFlow;

/// One registered agent's part of a thread: every statement of the partition.
#[derive(Clone)]
pub struct Partition {
    pub(super) thread: Thread,
    pub(super) name: String,
}

impl Partition {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The inbox receive, one atomic batch: the inbox row, the `submitted` task row and the
    /// lease attempt under `holder`. A lost acknowledgment is retried with the same token.
    pub async fn receive(
        &self,
        holder: &Holder,
        input: db::ReceiveInput<'_>,
    ) -> Result<(Receive, Option<Lease>), StorageError> {
        self.receive_at(holder, input, None).await
    }

    pub async fn receive_at(
        &self,
        holder: &Holder,
        input: db::ReceiveInput<'_>,
        endpoint: Option<&str>,
    ) -> Result<(Receive, Option<Lease>), StorageError> {
        self.thread.ensure_schema().await?;
        let lease_ms = self.thread.timing.millis();
        let conn = &*self.thread.conn;
        let (thread, wait, write) =
            (self.thread.clone(), LEASE_WAIT_TIMEOUT, StorageWrite::Receive);
        let mut pending = PendingLease::new(thread, self.name(), holder, None, wait, write);
        let (received, lease) = loop {
            let (token, lost) = (pending.token().to_owned(), pending.lost());
            let request = async {
                let out = db::receive(
                    conn,
                    &self.name,
                    holder.as_str(),
                    &token,
                    lease_ms,
                    endpoint,
                    input,
                );
                match out.await? {
                    // a duplicate of this very task after a lost acknowledgment is its own insert
                    Receive::Duplicate(Some(task)) if lost && task == input.task_id => {
                        let fence =
                            self.thread.read(SelectFence { partition_key: &self.name }, |row| {
                                Ok((row[0].as_i64(), row[1].as_str().map(str::to_owned)))
                            });
                        let held = fence.await?.into_iter().next().and_then(|(version, held)| {
                            version.filter(|_| held.as_deref() == Some(&*token))
                        });
                        Ok(Receive::Received(held))
                    }
                    received => Ok(received),
                }
            };
            let version = |received: &Receive| match received {
                Receive::Received(version) => *version,
                _ => None,
            };
            if let ControlFlow::Break(out) = pending.attempt(request, version).await {
                break out?;
            }
        };
        if let Some(lease) = &lease {
            lease.check()?;
            lease.keep_alive();
        }
        Ok((received, lease))
    }

    /// A turn's `submitted` row, before the lease is tried; `false` when its task or message id
    /// already had one. Keyed by the task id: after a lost acknowledgment, the row found under
    /// it is this insert.
    pub async fn submit(&self, task: &TaskWrite) -> Result<bool, StorageError> {
        self.submit_at(task, None).await
    }

    pub async fn submit_at(
        &self,
        task: &TaskWrite,
        endpoint: Option<&str>,
    ) -> Result<bool, StorageError> {
        self.thread.ensure_schema().await?;
        let insert = db::submitted_task;
        self.thread
            .resend(StorageWrite::Submit)
            .probed(
                || async move { Ok(self.thread.exec(insert(&self.name, task, endpoint)).await? > 0) },
                || async move { Ok(self.task(&task.task_id).await?.map(|_| true)) },
            )
            .await
    }

    /// A cancellation checkpoint: the task's current status.
    pub async fn checkpoint(&self, task_id: &str) -> Result<Option<TaskState>, StorageError> {
        db::checkpoint(&*self.thread.conn, &self.name, task_id).await
    }

    /// Takes a `submitted` or `working` row to `working`. The status it had (`working` means a
    /// dead holder had it), `None` when the row is terminal or missing.
    pub async fn pickup(&self, task_id: &str) -> Result<Option<TaskState>, StorageError> {
        db::pickup(&*self.thread.conn, &self.name, task_id).await
    }

    /// A `working` row whose holder died ends `failed` with the given status message. Keyed by
    /// the message: after a lost acknowledgment a row failed with it is this write's.
    pub async fn fail_orphan(
        &self,
        task_id: &str,
        status_message: &a2a::Message,
    ) -> Result<bool, StorageError> {
        let fail = || async move {
            let message = SqlValue::json(status_message);
            let fail = FailOrphan { partition_key: &self.name, task_id, status_message: message };
            Ok(self.thread.exec(fail).await? > 0)
        };
        let failed_by_it = || async move {
            let row = self.task(task_id).await?;
            let failed = row.is_some_and(|row| {
                row.status == TaskState::Failed
                    && row.data.status_message.as_ref().map(|m| &m.message_id)
                        == Some(&status_message.message_id)
            });
            Ok(failed.then_some(true))
        };
        self.thread.resend(StorageWrite::FailOrphan).probed(fail, failed_by_it).await
    }

    /// The whole history; how another partition reads it.
    pub async fn history(&self) -> Result<Vec<ItemRow>, StorageError> {
        self.thread.read(SelectItems { partition_key: &self.name }, ItemRow::from_row).await
    }

    pub async fn info(&self) -> Result<Option<PartitionRow>, StorageError> {
        let partition_key = SelectPartition { partition_key: &self.name };
        Ok(self.thread.read(partition_key, PartitionRow::from_row).await?.into_iter().next())
    }

    /// The newest task rows, of one status when given.
    pub async fn tasks(
        &self,
        status: Option<TaskState>,
        limit: i64,
    ) -> Result<Vec<TaskRow>, StorageError> {
        let partition_key = &self.name;
        match status {
            Some(status) => {
                let by_status =
                    SelectTasksByStatus { partition_key, status: status.as_str(), limit };
                self.thread.read(by_status, TaskRow::from_row).await
            }
            None => self.thread.read(SelectTasks { partition_key, limit }, TaskRow::from_row).await,
        }
    }

    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        let task = SelectTask { partition_key: &self.name, task_id };
        Ok(self.thread.read(task, TaskRow::from_row).await?.into_iter().next())
    }

    /// The task an A2A `messageId` created.
    pub async fn task_for_message(
        &self,
        message_id: &str,
    ) -> Result<Option<TaskRow>, StorageError> {
        let task = SelectTaskForMessage { partition_key: &self.name, message_id };
        Ok(self.thread.read(task, TaskRow::from_row).await?.into_iter().next())
    }

    /// The inbox rows in `id` order: every one, or only those above `drained_id`.
    pub async fn inbox(&self, pending: bool) -> Result<Vec<InboxRow>, StorageError> {
        let partition_key = &self.name;
        match pending {
            true => {
                self.thread.read(SelectInboxPending { partition_key }, InboxRow::from_row).await
            }
            false => self.thread.read(SelectInbox { partition_key }, InboxRow::from_row).await,
        }
    }

    pub async fn routed_inbox(
        &self,
    ) -> Result<Vec<lk_agents_abi::storage::thread::RoutedInbox>, StorageError> {
        self.thread
            .read(SelectRoutedInbox { partition_key: &self.name }, |cells| {
                use crate::sql::FromCell as _;
                Ok(lk_agents_abi::storage::thread::RoutedInbox {
                    row: InboxRow::from_row(&cells[..6])?,
                    endpoint: Option::<String>::from_cell(&cells[6])?,
                })
            })
            .await
    }

    /// The partition's journal after the event `after_id`.
    pub async fn journal(&self, after_id: i64) -> Result<Vec<EventRow>, StorageError> {
        let events = SelectPartitionEvents { partition_key: &self.name, after_id };
        self.thread.read(events, EventRow::from_row).await
    }

    /// The last `limit` runs, newest first; the `submitted` insert before a lease is
    /// bookkeeping, not a turn.
    pub async fn runs(&self, limit: usize) -> Result<Vec<RunSummary>, StorageError> {
        let rows = self.journal(0).await?;
        let mut runs: Vec<RunSummary> = RunSummary::from_events(&rows)?
            .into_iter()
            .filter(|r| r.channel.as_deref() != Some(SUBMIT_CHANNEL))
            .collect();
        let keep = runs.len().saturating_sub(limit);
        runs.drain(..keep);
        runs.reverse();
        Ok(runs)
    }

    /// `tasks/cancel`: a row not yet terminal goes `canceled` with `metadata` merged in; a
    /// running turn yields at its next checkpoint. The row after, unchanged when it was already
    /// terminal; `None` when the task is not the partition's.
    pub async fn cancel_task(
        &self,
        task_id: &str,
        metadata: Map<String, Value>,
    ) -> Result<Option<TaskRow>, StorageError> {
        self.thread.ensure_schema().await?;
        // the row as it stands after: the same answer however often it applied
        let conn = &*self.thread.conn;
        self.thread
            .resend(StorageWrite::Cancel)
            .send(|| db::cancel(conn, self.name(), task_id, &metadata))
            .await
    }

    /// Transfer a batch drain's existing lease into its next run deliberately.
    pub async fn resume(&self, opts: RunOptions, lease: Lease) -> Result<Run, StorageError> {
        lease.check()?;
        let result = self.open(opts, lease.owner()).await;
        lease.stop();
        result
    }

    /// Takes the partition for `owner` and loads it. The run commits with [`Run::commit`] and ends
    /// by error with [`Run::abandon`]; a run dropped without either ends `canceled` as it
    /// releases its lease. Only a dead worker leaves a `run_started` without a `run_ended`.
    pub async fn open(&self, opts: RunOptions, owner: &Holder) -> Result<Run, StorageError> {
        let (lease, closed, state, items, tasks, picked_up, context_scopes) = match opts.access {
            PartitionAccess::Exclusive(when_held) => {
                let (a, lease) = self.acquire(&opts, when_held, owner).await?;
                if !a.already_committed {
                    lease.check()?;
                    lease.keep_alive();
                }
                let tasks = Tasks::loaded(self.clone(), a.tasks);
                (
                    Some(lease),
                    a.already_committed,
                    a.partition_key.state,
                    a.items,
                    tasks,
                    a.picked_up,
                    a.context_scopes,
                )
            }
            PartitionAccess::Concurrent => {
                // without a lease nothing is claimed: the run goes only while its row is still
                // `submitted` (a client did not cancel it), and the row stays so, never an
                // orphan for a lease holder to fail
                let picked_up = opts.pickup == TaskPickup::Claim
                    && self.task(&opts.task_id).await?.map(|r| r.status)
                        == Some(TaskState::Submitted);
                let items = self.history().await?;
                (None, false, None, items, Tasks::new(self.clone()), picked_up, Vec::new())
            }
        };
        let (state, session_items) =
            crate::session_snapshot::hydrate(&*self.thread.conn, &self.name, state).await?;
        Ok(Run {
            session_items,
            partition_key: self.clone(),
            lease,
            access: opts.access,
            holder: owner.as_str().to_owned(),
            run_id: opts.run_id,
            task_id: opts.task_id,
            channel: opts.channel,
            closed,
            loaded: items.iter().map(|i| (i.id.clone(), i.payload.clone())).collect(),
            history: items,
            loaded_state: state.clone(),
            state,
            context_scopes,
            scope_writes: Vec::new(),
            app_writes: Vec::new(),
            events: EventLog::default(),
            tasks,
            picked_up,
            usage: Usage::default(),
            outcome: RunOutcome::Completed,
            reason: None,
            finished: false,
            unresolved: None,
        })
    }

    /// The acquire, polled with backoff while another holder has the partition, as `when_held` says.
    /// One token for every attempt: a lost acknowledgment is retried as the holder re-entering
    /// what it may have taken.
    async fn acquire(
        &self,
        opts: &RunOptions,
        when_held: WhenHeld,
        owner: &Holder,
    ) -> Result<(Acquired, Lease), StorageError> {
        let holder = owner.as_str();
        let epoch = tokio::time::Instant::now();
        let mut poll = Backoff::new(epoch.elapsed(), opts.wait_timeout);
        let (thread, wait) = (self.thread.clone(), opts.wait_timeout);
        let run_id = Some(opts.run_id.as_str());
        let mut pending =
            PendingLease::new(thread, self.name(), owner, run_id, wait, StorageWrite::Acquire);
        let token = pending.token().to_owned();
        let timing = self.thread.timing;
        let stale = timing.renew_interval + RENEW_SLACK;
        // when this acquire first found the partition held by a holder that stopped renewing
        let mut ownerless_since = None;
        let waited = |ownerless_since: Option<std::time::Duration>| {
            let waited = epoch.elapsed();
            let ownerless = ownerless_since.map(|since| waited.saturating_sub(since));
            telemetry::acquire_waited(&self.thread.cx, opts.kind, self.name(), waited, ownerless);
        };
        loop {
            let req = Acquire {
                partition_key: self.name(),
                holder,
                token: &token,
                lease_ms: self.thread.timing.millis(),
                wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
                run_id: &opts.run_id,
                task_id: &opts.task_id,
                run_started: EventWrite::new(
                    unix_now_ms(),
                    JournalEvent::RunStarted(RunStarted {
                        task_id: Some(opts.task_id.clone()),
                        stream_id: Some(opts.task_id.clone()),
                        channel: opts.channel.clone(),
                        exclusive: None,
                    }),
                ),
                pickup: opts.pickup,
                tasks_limit: LOOKUP_LIMIT,
            };
            // the DDL rides along on a fresh database; a batch that loses the race to another
            // creator's DDL is retried without it
            let bootstrap = self.thread.bootstrap();
            let initializing = !bootstrap.is_empty();
            let conn = &*self.thread.conn;
            let request = async {
                match db::acquire(conn, &req, bootstrap).await {
                    Err(error) if initializing => {
                        // an uncertain batch is resolved like any other
                        self.thread.reconcile_schema(error).await?;
                        self.thread.schema_applied();
                        db::acquire(conn, &req, Vec::new()).await
                    }
                    out => out,
                }
            };
            let version = |(acquired, _): &(Option<Acquired>, db::LeaseBefore)| {
                acquired.as_ref().map(|a| a.version)
            };
            let ((acquired, before), lease) = match pending.attempt(request, version).await {
                ControlFlow::Break(out) => out?,
                // the same token again: the holder re-enters what it may have taken
                ControlFlow::Continue(()) => continue,
            };
            self.thread.schema_applied();
            if let (Some(acquired), Some(lease)) = (acquired, lease) {
                if before.lapsed(holder) {
                    let lapsed = before.holder.as_deref().unwrap_or_default();
                    telemetry::lease_lapsed(&self.thread.cx, opts.kind, self.name(), lapsed);
                }
                waited(ownerless_since);
                return Ok((acquired, lease));
            }
            if ownerless_since.is_none() && before.ownerless(holder, timing.millis(), stale) {
                ownerless_since = Some(epoch.elapsed());
            }
            if when_held == WhenHeld::Reject {
                waited(ownerless_since);
                return Err(StorageError::Rejected { running_task_id: None });
            }
            let Some(at) = poll.next(epoch.elapsed()) else {
                log::debug!(target: lk_agents_abi::LOG_TARGET, partition_key = self.name(); "the partition_key was not released within {}s", opts.wait_timeout.as_secs_f64());
                waited(ownerless_since);
                return Err(StorageError::Rejected { running_task_id: None });
            };
            tokio::time::sleep_until(epoch + at).await;
        }
    }
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod tests;

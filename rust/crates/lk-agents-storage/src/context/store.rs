use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lk_agents_abi::storage::context::{
    AppStatement, ContextActivation as HostContextActivation, ContextField, ContextScope,
    ContextScopeWrite, ContextValues,
};
use parking_lot::Mutex;
use serde_json::Value;

use super::ContextActivation as Activation;
use super::transaction::Transaction;
use crate::backends::with_thread;
use crate::sql::{Bind as _, Row, Statement};
use crate::thread::run::RunOutcome;
use crate::thread::schema::{RecordAppliedWrite, SelectAppliedWrite};
use crate::thread::{RESOLVE_WAIT, Resend};
use crate::{CommitOptions, Holder, RunOptions, Storage, StorageError, Thread, shortuuid};
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::storage::thread::AbandonReason;
use tokio::sync::watch;

#[derive(Clone)]
pub struct ContextStore {
    activation: Arc<Mutex<Activation>>,
    database: Option<Database>,
    operation: Arc<tokio::sync::Mutex<()>>,
    managed: Arc<AtomicBool>,
    /// Bumped as each managed turn begins and ends: a transaction joins only the turn it began
    /// in.
    turn: Arc<AtomicU64>,
    /// What managed transactions staged for the turn's commit.
    statements: Arc<Mutex<Vec<AppStatement>>>,
    /// A direct save whose acknowledgment was lost, resolving before the next load or save.
    lost_save: Arc<Mutex<Option<watch::Receiver<Option<SaveOutcome>>>>>,
}

/// The lane's thread database: direct saves outside a managed turn, and application SQL.
#[derive(Clone)]
pub(crate) struct Database {
    storage: Arc<dyn Storage>,
    thread_id: String,
    lane: String,
    /// The store's direct saves, one owner of the lane.
    holder: Holder,
}

impl Database {
    fn new(storage: Arc<dyn Storage>, thread_id: String, lane: String) -> Self {
        Self { storage, thread_id, lane, holder: Holder::new("context") }
    }

    /// `f` over the thread opened for it, its schema applied, then closed.
    async fn with_thread<T>(
        &self,
        f: impl AsyncFnOnce(&Thread) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        with_thread(&*self.storage, &self.thread_id, async |thread| {
            thread.ensure_schema().await?;
            f(thread).await
        })
        .await
    }

    /// One read over committed records.
    pub(crate) async fn query(&self, read: Statement) -> Result<Vec<Row>, StorageError> {
        self.with_thread(async |thread| thread.conn().query(read).await).await
    }

    /// An application transaction's statements in one atomic batch, applied once: the batch
    /// records its request id first, so a resend after a lost acknowledgment that finds the
    /// record reports the commit instead of applying it again.
    pub(crate) async fn batch(&self, statements: Vec<Statement>) -> Result<(), StorageError> {
        let id = shortuuid("write_");
        let (id, statements) = (id.as_str(), &statements);
        self.with_thread(async |thread| {
            let conn = &**thread.conn();
            let write = || async move {
                let mut batch = vec![RecordAppliedWrite { id }.bind()];
                batch.extend(statements.iter().cloned());
                conn.batch(batch).await.map(drop)
            };
            let applied = || async move {
                let rows = conn.query(SelectAppliedWrite { id }.bind()).await?;
                Ok((!rows.is_empty()).then_some(()))
            };
            Resend::new(RESOLVE_WAIT).probed(write, applied).await
        })
        .await
    }

    async fn snapshot(&self) -> Result<(Vec<ContextScope>, Option<Value>), StorageError> {
        with_thread(&*self.storage, &self.thread_id, async |thread| {
            let scopes = thread.context_scopes(&self.lane).await?;
            let routing = thread.lane(&self.lane).info().await?.and_then(|lane| lane.state);
            Ok((scopes, routing))
        })
        .await
    }

    /// A direct save, one fenced run. One whose acknowledgment stayed lost keeps resolving in
    /// the background, on the holder's cleanup set, and is answered with its outcome's watch.
    async fn commit(&self, writes: Vec<ContextScopeWrite>) -> Result<(), Unsaved> {
        let thread = self.storage.open(&self.thread_id).await?;
        let mut run = match self.stage(&thread, writes).await {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error.into());
            }
        };
        match run.commit(CommitOptions::default()).await {
            Err(StorageError::Uncertain(why)) => {
                let (resolved, resolving) = watch::channel(None);
                let lost = why.clone();
                self.holder.spawn_cleanup("context save resolution", async move {
                    let outcome = match run.resolve_later().await {
                        Some(Ok(_)) => SaveOutcome::Committed,
                        Some(Err(StorageError::Uncertain(why))) => SaveOutcome::Unknown(why),
                        // refused or rejected: the save cannot apply any more
                        Some(Err(_)) => SaveOutcome::NeverApplied,
                        None => SaveOutcome::Unknown(lost),
                    };
                    thread.close().await;
                    resolved.send_replace(Some(outcome));
                });
                Err(Unsaved::Lost(why, resolving))
            }
            out => {
                thread.close().await;
                Ok(out.map(drop)?)
            }
        }
    }

    /// The direct save's run, its writes staged once the lease confirmed their revisions.
    async fn stage(
        &self,
        thread: &Thread,
        writes: Vec<ContextScopeWrite>,
    ) -> Result<crate::Run, StorageError> {
        let mut options = RunOptions::new(shortuuid("context_"), "");
        options.channel = Some("context".into());
        let mut run = thread.lane(&self.lane).open(options, &self.holder).await?;
        // The lease serializes direct and conversational saves for this scope's lane.
        // Check before any writes so a stale direct save has a typed refusal.
        for write in &writes {
            let revision = run
                .context_scopes()
                .iter()
                .find(|scope| scope.scope_id == write.scope.scope_id)
                .map_or(0, |scope| scope.revision);
            if revision != write.expected_revision {
                run.abandon(RunOutcome::Failed, AbandonReason::ContextVersionConflict.as_str())
                    .await?;
                return Err(StorageError::VersionConflict {
                    lane: self.lane.clone(),
                    expected: write.expected_revision,
                });
            }
        }
        run.stage_scopes(writes);
        Ok(run)
    }
}

/// A direct save that did not commit: refused, or its acknowledgment lost with its resolution
/// in flight.
enum Unsaved {
    Refused(StorageError),
    Lost(Unacknowledged, watch::Receiver<Option<SaveOutcome>>),
}

impl From<StorageError> for Unsaved {
    fn from(error: StorageError) -> Self {
        Self::Refused(error)
    }
}

/// How a lost direct save resolved.
#[derive(Clone)]
enum SaveOutcome {
    Committed,
    /// Refused or rejected: it cannot apply any more.
    NeverApplied,
    /// Nobody could tell.
    Unknown(Unacknowledged),
}

impl ContextStore {
    fn with(activation: Activation, database: Option<Database>, managed: bool) -> Self {
        Self {
            activation: Arc::new(Mutex::new(activation)),
            database,
            operation: Arc::new(tokio::sync::Mutex::new(())),
            managed: Arc::new(AtomicBool::new(managed)),
            turn: Arc::new(AtomicU64::new(0)),
            statements: Arc::default(),
            lost_save: Arc::default(),
        }
    }

    /// A managed turn's store over the scopes and routing state its run loaded: saves and
    /// joining transactions stage for that turn's commit.
    pub async fn turn(
        storage: Arc<dyn Storage>,
        thread_id: String,
        lane: String,
        scopes: Vec<ContextScope>,
        routing: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        let activation = Activation::new(scopes, routing, read_only)?;
        Ok(Self::with(activation, Some(Database::new(storage, thread_id, lane)), true))
    }

    /// Stages a joining transaction's writes, unless its managed turn has ended.
    pub(crate) async fn join_turn(
        &self,
        turn: u64,
        writes: Vec<AppStatement>,
    ) -> Result<(), StorageError> {
        let _operation = self.operation.lock().await;
        if !self.managed.load(Ordering::Acquire) || self.turn.load(Ordering::Acquire) != turn {
            return Err(StorageError::Invalid(
                "the managed turn this transaction joined has ended; nothing was written".into(),
            ));
        }
        self.statements.lock().extend(writes);
        Ok(())
    }

    fn clear_statements(&self) {
        self.statements.lock().clear();
    }

    /// Settles a lost direct save by its outcome before anything reads or stages over it:
    /// committed, its writes are acknowledged; never applied, they are dropped. `Uncertain`
    /// while nobody can tell, and its writes stay pending.
    async fn settle_lost_save(&self, database: &Database) -> Result<(), StorageError> {
        let Some(mut resolving) = self.lost_save.lock().clone() else { return Ok(()) };
        let outcome = resolving.wait_for(Option::is_some).await.ok().and_then(|o| o.clone());
        match outcome {
            Some(SaveOutcome::Committed) => self.activation.lock().acknowledge(),
            Some(SaveOutcome::NeverApplied) => self.activation.lock().abandon(),
            Some(SaveOutcome::Unknown(why)) => return Err(StorageError::Uncertain(why)),
            // its resolver was stopped: the database's committed state is the truth
            None => {
                let (scopes, routing) = database.snapshot().await?;
                *self.activation.lock() = Activation::new(scopes, routing, false)?;
            }
        }
        *self.lost_save.lock() = None;
        Ok(())
    }

    /// Waits, bounded, for the releases its direct saves' dropped leases left.
    pub(crate) async fn wait_released(&self) -> bool {
        match &self.database {
            Some(database) => database.holder.wait_released().await,
            None => true,
        }
    }

    pub(crate) async fn lock_managed(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.operation.clone().lock_owned().await
    }

    pub(crate) fn begin_managed(
        &self,
        scopes: Vec<ContextScope>,
        routing: Option<Value>,
        _operation: &tokio::sync::OwnedMutexGuard<()>,
    ) -> Result<(), StorageError> {
        self.activation.lock().rebase(scopes, routing)?;
        self.clear_statements();
        self.turn.fetch_add(1, Ordering::AcqRel);
        self.managed.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn end_managed(
        &self,
        committed: bool,
        _operation: tokio::sync::OwnedMutexGuard<()>,
    ) {
        let mut activation = self.activation.lock();
        if committed {
            activation.acknowledge();
        } else {
            activation.abandon();
        }
        self.clear_statements();
        self.turn.fetch_add(1, Ordering::AcqRel);
        self.managed.store(false, Ordering::Release);
    }

    pub async fn bound(
        storage: Arc<dyn Storage>,
        thread_id: String,
        lane: String,
    ) -> Result<Self, StorageError> {
        let database = Database::new(storage, thread_id, lane);
        let (scopes, routing) = database.snapshot().await?;
        Ok(Self::with(Activation::new(scopes, routing, false)?, Some(database), false))
    }
}

impl HostContextActivation for ContextStore {
    type Transaction = Transaction;

    fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        Ok(Self::with(Activation::new(scopes, routing_state, read_only)?, None, false))
    }

    async fn load(
        &self,
        scope_id: String,
        identity: String,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError> {
        let _operation = self.operation.lock().await;
        if let Some(database) = &self.database
            && !self.managed.load(Ordering::Acquire)
        {
            self.settle_lost_save(database).await?;
            let (scopes, routing) = database.snapshot().await?;
            self.activation.lock().rebase(scopes, routing)?;
        }
        self.activation.lock().load(&scope_id, &identity, &fields)
    }

    async fn save(
        &self,
        scope_id: String,
        identity: String,
        revision: i64,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError> {
        let _operation = self.operation.lock().await;
        let direct = self.database.as_ref().filter(|_| !self.managed.load(Ordering::Acquire));
        if let Some(database) = direct {
            self.settle_lost_save(database).await?;
        }
        let values = self.activation.lock().save(&scope_id, &identity, revision, &fields)?;
        if let Some(database) = direct {
            let writes = self.activation.lock().writes();
            match database.commit(writes).await {
                Ok(()) => self.activation.lock().acknowledge(),
                Err(Unsaved::Lost(why, resolving)) => {
                    *self.lost_save.lock() = Some(resolving);
                    return Err(StorageError::Uncertain(why));
                }
                Err(Unsaved::Refused(error)) => {
                    if let Ok((scopes, routing)) = database.snapshot().await {
                        *self.activation.lock() = Activation::new(scopes, routing, false)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(values)
    }

    fn writes(&self) -> Vec<ContextScopeWrite> {
        self.activation.lock().writes()
    }

    fn statements(&self) -> Vec<AppStatement> {
        self.statements.lock().clone()
    }

    async fn transaction(&self, independent: bool) -> Result<Transaction, StorageError> {
        self.activation.lock().ensure_open()?;
        let database = self.database.clone().ok_or_else(|| {
            StorageError::Invalid("this context store has no thread database".into())
        })?;
        // Outside a managed turn, a joining transaction commits like an independent one.
        let _operation = self.operation.lock().await;
        let joins = (!independent && self.managed.load(Ordering::Acquire))
            .then(|| (self.clone(), self.turn.load(Ordering::Acquire)));
        Ok(Transaction::new(database, joins))
    }

    fn acknowledge(&self) {
        self.activation.lock().acknowledge();
        self.clear_statements();
    }

    fn closed(&self) -> bool {
        self.activation.lock().closed()
    }

    fn close(&self) {
        self.activation.lock().close();
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

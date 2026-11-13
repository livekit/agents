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
use crate::telemetry::Cx;
use crate::thread::run::RunOutcome;
use crate::thread::schema::{RecordAppliedWrite, SelectAppliedWrite};
use crate::{CommitOptions, Holder, RunOptions, Storage, StorageError, Thread, shortuuid};
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::storage::thread::AbandonReason;
use lk_agents_net::signal;

#[path = "store_session.rs"]
mod session;

#[derive(Clone)]
pub struct ContextStore {
    pub(crate) session_view: Arc<Mutex<Option<crate::session_state::SessionState>>>,
    owned_session: Arc<tokio::sync::Mutex<Option<session::OwnedSession>>>,
    session_sink: Arc<Mutex<Option<Arc<dyn lk_agents_abi::a2a::turn::HandlerPort>>>>,
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
    lost_save: Arc<Mutex<Option<signal::Cell<Option<SaveOutcome>>>>>,
}

/// The partition's thread database: direct saves outside a managed turn, and application SQL.
#[derive(Clone)]
pub(crate) struct Database {
    storage: Arc<dyn Storage>,
    thread_id: String,
    partition_key: String,
    /// The store's direct saves, one owner of the partition.
    holder: Holder,
    /// The operation the store serves: its writes' health metrics record under it.
    cx: Cx,
}

impl Database {
    fn new(storage: Arc<dyn Storage>, thread_id: String, partition_key: String, cx: Cx) -> Self {
        Self { storage, thread_id, partition_key, holder: Holder::new("context"), cx }
    }

    async fn open(&self) -> Result<Thread, StorageError> {
        Ok(self.storage.open(&self.thread_id).await?.with_telemetry(self.cx.clone()))
    }

    /// `f` over the thread opened for it, its schema applied, then closed.
    async fn with_thread<T>(
        &self,
        f: impl AsyncFnOnce(&Thread) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let thread = self.open().await?;
        let result = async {
            thread.ensure_schema().await?;
            f(&thread).await
        }
        .await;
        thread.close().await;
        result
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
            thread.resend(crate::telemetry::StorageWrite::Context).probed(write, applied).await
        })
        .await
    }

    async fn snapshot(&self) -> Result<(Vec<ContextScope>, Option<Value>), StorageError> {
        with_thread(&*self.storage, &self.thread_id, async |thread| {
            let scopes = thread.context_scopes(&self.partition_key).await?;
            let routing = thread
                .partition(&self.partition_key)
                .info()
                .await?
                .and_then(|partition_key| partition_key.state);
            let (routing, _) =
                crate::session_snapshot::hydrate(&**thread.conn(), &self.partition_key, routing)
                    .await?;
            Ok((scopes, routing))
        })
        .await
    }

    /// A direct save, one fenced run. One whose acknowledgment stayed lost keeps resolving in
    /// the background, on the holder's cleanup set, and is answered with its outcome's watch.
    async fn commit(&self, writes: Vec<ContextScopeWrite>) -> Result<(), Unsaved> {
        let thread = self.open().await?;
        let mut run = match self.stage(&thread, writes).await {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error.into());
            }
        };
        match run.commit(CommitOptions::default()).await {
            Err(StorageError::Uncertain(why)) => {
                let (resolved, resolving) = signal::once();
                let lost = why.clone();
                let spawner = thread.background.spawner.clone();
                self.holder.spawn_settling_cleanup(
                    &*spawner,
                    "context save resolution",
                    resolved,
                    async move {
                        let outcome = match run.resolve_later().await {
                            Some(Ok(_)) => SaveOutcome::Committed,
                            Some(Err(StorageError::Uncertain(why))) => SaveOutcome::Unknown(why),
                            // refused or rejected: the save cannot apply any more
                            Some(Err(_)) => SaveOutcome::NeverApplied,
                            None => SaveOutcome::Unknown(lost),
                        };
                        thread.close().await;
                        outcome
                    },
                );
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
        options.kind = crate::thread::run::PartitionKind::Context;
        let mut run = thread.partition(&self.partition_key).open(options, &self.holder).await?;
        // The lease serializes direct and conversational saves for this scope's partition.
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
                    partition_key: self.partition_key.clone(),
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
    Lost(Unacknowledged, signal::Cell<Option<SaveOutcome>>),
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
    pub(crate) fn acknowledge_session_state(&self) {
        self.activation.lock().acknowledge_session();
    }

    pub(crate) fn matches_partition(&self, thread_id: &str, partition_key: &str) -> bool {
        self.database
            .as_ref()
            .is_some_and(|db| db.thread_id == thread_id && db.partition_key == partition_key)
    }

    fn with(activation: Activation, database: Option<Database>, managed: bool) -> Self {
        Self {
            session_view: Arc::default(),
            owned_session: Arc::default(),
            session_sink: Arc::default(),
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
        partition_key: String,
        scopes: Vec<ContextScope>,
        routing: Option<Value>,
        read_only: bool,
        cx: Cx,
    ) -> Result<Self, StorageError> {
        let activation = Activation::new(scopes, routing, read_only)?;
        Ok(Self::with(activation, Some(Database::new(storage, thread_id, partition_key, cx)), true))
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
        let Some(resolving) = self.lost_save.lock().clone() else { return Ok(()) };
        let outcome = resolving.wait().await.ok();
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
            Some(database) => {
                database.holder.wait_released(&*database.storage.background().clock).await
            }
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

    /// The partition's context, its writes' health metrics recorded under `cx`.
    pub async fn bound(
        storage: Arc<dyn Storage>,
        thread_id: String,
        partition_key: String,
        cx: Cx,
    ) -> Result<Self, StorageError> {
        let database = Database::new(storage, thread_id, partition_key, cx);
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
                // The refused values never stand: the committed ones do, until the next load
                // reloads the partition.
                Err(Unsaved::Refused(error)) => {
                    self.activation.lock().abandon();
                    return Err(error);
                }
            }
        }
        Ok(values)
    }

    async fn load_session(&self) -> Result<Option<Value>, StorageError> {
        self.activation.lock().ensure_open()?;
        match &self.database {
            Some(database) => Ok(database.snapshot().await?.1),
            None => Ok(self.activation.lock().committed_session()),
        }
    }

    async fn load_session_snapshot(&self) -> Result<String, StorageError> {
        let value = self
            .load_session()
            .await?
            .unwrap_or_else(|| serde_json::json!({"version":1,"kind":"empty"}));
        serde_json::to_string(&value).map_err(|error| StorageError::Invalid(error.to_string()))
    }

    fn declare_binding(
        &self,
        context: lk_agents_abi::HostRef,
        store: lk_agents_abi::HostRef,
    ) -> lk_agents_abi::storage::session::BoundContext {
        lk_agents_abi::storage::session::BoundContext { context, store }
    }

    async fn flush_bound_session(
        &self,
        history: Option<Vec<lk_agents_abi::ChatItem>>,
    ) -> Result<(), StorageError> {
        self.activation.lock().ensure_open()?;
        let view = self.session_view.lock().clone();
        let Some(view) = view else { return Ok(()) };
        let history = match history {
            Some(history) => history,
            None => view.root_history()?,
        };
        self.flush_session(history).await
    }

    fn set_session_sink(
        &self,
        sink: Arc<dyn lk_agents_abi::a2a::turn::HandlerPort>,
    ) -> Result<(), StorageError> {
        self.activation.lock().ensure_open()?;
        *self.session_sink.lock() = Some(sink);
        Ok(())
    }

    async fn flush_session(
        &self,
        history: Vec<lk_agents_abi::ChatItem>,
    ) -> Result<(), StorageError> {
        let state = {
            let activation = self.activation.lock();
            activation.ensure_open()?;
            activation.session_state()
        };
        let Some(state) = state else { return Ok(()) };
        let sink = self.session_sink.lock().clone();
        if let Some(sink) = sink {
            sink.save_session(state, history).await?;
        } else if self.database.is_some() {
            self.checkpoint_owned_session(state, history).await?;
        }
        self.activation.lock().acknowledge_session();
        Ok(())
    }

    fn stage_session(&self, state: Value) -> Result<(), StorageError> {
        let view = self.session_view.lock().clone();
        let checked = view
            .as_ref()
            .map(|_| crate::session_state::SessionState::from_value(state.clone()))
            .transpose()?;
        let mut activation = self.activation.lock();
        activation.stage_session(state)?;
        if self.database.is_none() {
            activation.acknowledge_session();
        }
        drop(activation);
        if let Some((view, checked)) = view.zip(checked) {
            view.assign(&checked);
        }
        Ok(())
    }

    fn session_state(&self) -> Option<Value> {
        self.activation.lock().session_state()
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

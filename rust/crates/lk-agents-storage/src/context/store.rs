//! A lane's context field store as the host uses it: snapshot staging in a turn, fenced direct
//! saves outside one, and the application transactions over the same database.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lk_agents_abi::storage::context::{
    AppStatement, ContextActivation as HostContextActivation, ContextField, ContextScope,
    ContextScopeWrite, ContextValues,
};
use lk_agents_net::lock::Mutex;
use serde_json::Value;

use super::ContextActivation as Activation;
use super::transaction::Transaction;
use crate::run::RunOutcome;
use crate::sql::{Row, Statement};
use crate::{CommitOptions, RunOptions, Storage, StorageError, Thread, shortuuid};

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
}

/// The lane's thread database: direct saves outside a managed turn, and application SQL.
#[derive(Clone)]
pub(crate) struct Database {
    storage: Arc<dyn Storage>,
    thread_id: String,
    lane: String,
}

impl Database {
    /// `f` over the thread opened for it, its schema applied, then closed.
    async fn with_thread<T>(
        &self,
        f: impl AsyncFnOnce(&Thread) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let thread = self.storage.open(&self.thread_id, "application").await?;
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

    /// The statements in one atomic batch.
    pub(crate) async fn batch(&self, statements: Vec<Statement>) -> Result<(), StorageError> {
        self.with_thread(async |thread| thread.conn().batch(statements).await.map(drop)).await
    }

    async fn snapshot(&self) -> Result<(Vec<ContextScope>, Option<Value>), StorageError> {
        let thread = self.storage.open(&self.thread_id, "context").await?;
        let result = async {
            let scopes = thread.context_scopes(&self.lane).await?;
            let routing = thread.lane(&self.lane).info().await?.and_then(|lane| lane.state);
            Ok((scopes, routing))
        }
        .await;
        thread.close().await;
        result
    }

    async fn commit(&self, writes: Vec<ContextScopeWrite>) -> Result<(), StorageError> {
        let thread = self.storage.open(&self.thread_id, "context").await?;
        let result = async {
            let mut options = RunOptions::new(shortuuid("context_"), "");
            options.channel = Some("context".into());
            let mut run = thread.lane(&self.lane).open(options).await?;
            // The lease serializes direct and conversational saves for this scope's lane.
            // Check before any writes so a stale direct save has a typed refusal.
            for write in &writes {
                let revision = run
                    .context_scopes
                    .iter()
                    .find(|scope| scope.scope_id == write.scope.scope_id)
                    .map_or(0, |scope| scope.revision);
                if revision != write.expected_revision {
                    run.abandon(RunOutcome::Failed, "context-version-conflict").await?;
                    return Err(StorageError::VersionConflict {
                        lane: self.lane.clone(),
                        expected: write.expected_revision,
                    });
                }
            }
            run.scope_writes = writes;
            run.commit(CommitOptions::default()).await?;
            Ok(())
        }
        .await;
        thread.close().await;
        result
    }
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
        Ok(Self::with(activation, Some(Database { storage, thread_id, lane }), true))
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
        let database = Database { storage, thread_id, lane };
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
        let values = self.activation.lock().save(&scope_id, &identity, revision, &fields)?;
        if let Some(database) = &self.database
            && !self.managed.load(Ordering::Acquire)
        {
            let writes = self.activation.lock().writes();
            if let Err(error) = database.commit(writes).await {
                if !matches!(error, StorageError::Uncertain(_))
                    && let Ok((scopes, routing)) = database.snapshot().await
                {
                    *self.activation.lock() = Activation::new(scopes, routing, false)?;
                }
                return Err(error);
            }
            self.activation.lock().acknowledge();
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

    fn close(&self) {
        self.activation.lock().close();
    }
}

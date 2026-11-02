//! Scope identity, schema compatibility and revision-checked context staging.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::storage::context::{
    ContextField, ContextManifest, ContextScope, ContextScopeWrite, ContextValues,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::StorageError;
use crate::thread::db::STATE_LIMIT;

/// One activation's mutable scope state. Runtime owns serialization between callers.
pub struct ContextActivation {
    scopes: BTreeMap<String, ContextScope>,
    committed_scopes: BTreeMap<String, ContextScope>,
    base_revisions: BTreeMap<String, i64>,
    writes: BTreeMap<String, ContextScopeWrite>,
    routing_state: Option<Value>,
    read_only: bool,
    closed: bool,
}

impl ContextActivation {
    pub fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        let mut by_id = BTreeMap::new();
        let mut base_revisions = BTreeMap::new();
        for scope in scopes {
            if scope.scope_id.is_empty() || scope.revision < 0 {
                return Err(StorageError::Unavailable(
                    "invalid context scope identity or revision".into(),
                ));
            }
            base_revisions.insert(scope.scope_id.clone(), scope.revision);
            if by_id.insert(scope.scope_id.clone(), scope).is_some() {
                return Err(StorageError::Unavailable("duplicate context scope".into()));
            }
        }
        Ok(Self {
            committed_scopes: by_id.clone(),
            scopes: by_id,
            base_revisions,
            writes: BTreeMap::new(),
            routing_state,
            read_only,
            closed: false,
        })
    }

    pub(crate) fn ensure_open(&self) -> Result<(), StorageError> {
        if self.closed {
            return Err(StorageError::Invalid("the context store's activation has ended".into()));
        }
        Ok(())
    }

    pub fn load(
        &mut self,
        scope_id: &str,
        identity: &str,
        fields: &[ContextField],
    ) -> Result<ContextValues, StorageError> {
        self.ensure_open()?;
        if scope_id.is_empty() || identity.is_empty() {
            return Err(StorageError::Invalid(
                "context scope and schema identities are required".into(),
            ));
        }
        let mut scope = self.scopes.get(scope_id).cloned().unwrap_or_else(|| ContextScope {
            scope_id: scope_id.to_owned(),
            ..Default::default()
        });
        check_manifest(&mut scope, identity, fields)?;
        let values = fields
            .iter()
            .filter_map(|field| {
                scope.fields.get(&field.name).map(|value| (field.name.clone(), value.clone()))
            })
            .collect();
        let revision = scope.revision;
        self.scopes.insert(scope_id.to_owned(), scope);
        Ok(ContextValues { revision, fields: values })
    }

    pub fn save(
        &mut self,
        scope_id: &str,
        identity: &str,
        revision: i64,
        fields: &[ContextField],
    ) -> Result<ContextValues, StorageError> {
        self.ensure_open()?;
        if self.read_only {
            return Err(StorageError::Invalid(
                "context writes require an exclusive managed turn".into(),
            ));
        }
        let mut scope = self.scopes.get(scope_id).cloned().ok_or_else(|| {
            StorageError::Invalid("load_context must be called before save_context".into())
        })?;
        if scope.revision != revision {
            return Err(StorageError::VersionConflict {
                lane: scope_id.into(),
                expected: revision,
            });
        }
        if !scope.manifests.iter().any(|manifest| manifest.identity == identity) {
            return Err(StorageError::Invalid(
                "load_context must be called before save_context".into(),
            ));
        }
        check_manifest(&mut scope, identity, fields)?;
        scope.fields.extend(fields.iter().map(|field| (field.name.clone(), field.value.clone())));
        scope.revision = scope.revision.checked_add(1).ok_or_else(|| {
            StorageError::Invalid(format!("context scope {scope_id} exhausted its revision"))
        })?;
        check_state_size(
            self.routing_state.as_ref(),
            self.scopes.values().filter(|other| other.scope_id != scope_id).chain([&scope]),
        )?;
        let values = ContextValues { revision: scope.revision, fields: scope.fields.clone() };
        let write = ContextScopeWrite {
            expected_revision: *self.base_revisions.get(scope_id).unwrap_or(&0),
            scope: scope.clone(),
        };
        self.scopes.insert(scope_id.to_owned(), scope);
        self.writes.insert(scope_id.to_owned(), write);
        Ok(values)
    }

    pub fn writes(&self) -> Vec<ContextScopeWrite> {
        self.writes.values().cloned().collect()
    }

    pub fn acknowledge(&mut self) {
        for (id, write) in &self.writes {
            self.base_revisions.insert(id.clone(), write.scope.revision);
            self.committed_scopes.insert(id.clone(), write.scope.clone());
        }
        self.writes.clear();
    }

    pub fn rebase(
        &mut self,
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
    ) -> Result<(), StorageError> {
        self.ensure_open()?;
        if !self.writes.is_empty() {
            return Err(StorageError::Invalid("context saves are still pending".into()));
        }
        let incoming = Self::new(scopes, routing_state, self.read_only)?;
        if incoming.scopes == self.committed_scopes {
            self.routing_state = incoming.routing_state;
        } else {
            *self = incoming;
        }
        Ok(())
    }

    /// Discard staged fields after an abandoned activation without claiming a durable save.
    pub fn abandon(&mut self) {
        self.scopes = self.committed_scopes.clone();
        self.base_revisions =
            self.committed_scopes.iter().map(|(id, scope)| (id.clone(), scope.revision)).collect();
        self.writes.clear();
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

fn check_manifest(
    scope: &mut ContextScope,
    identity: &str,
    fields: &[ContextField],
) -> Result<(), StorageError> {
    let names: BTreeSet<_> = fields.iter().map(|field| field.name.as_str()).collect();
    if names.len() != fields.len() || names.contains("") {
        return Err(StorageError::Invalid(
            "persisted field names must be unique and nonempty".into(),
        ));
    }
    if let Some(previous) = scope.manifests.iter().find(|manifest| manifest.identity == identity) {
        for name in &previous.fields {
            if !names.contains(name.as_str()) {
                return Err(StorageError::Invalid(format!(
                    "context {identity} removed or renamed persisted field {name}; a migration is required"
                )));
            }
        }
    }
    for field in fields {
        if let Some(previous) = scope.contracts.get(&field.name)
            && previous != &field.schema
        {
            return Err(StorageError::Invalid(format!(
                "context scope {} has an incompatible contract for field {}",
                scope.scope_id, field.name
            )));
        }
    }
    for field in fields {
        scope.contracts.insert(field.name.clone(), field.schema.clone());
    }
    let manifest = ContextManifest {
        identity: identity.to_owned(),
        fields: names.into_iter().map(str::to_owned).collect(),
    };
    if let Some(previous) = scope.manifests.iter_mut().find(|m| m.identity == identity) {
        *previous = manifest;
    } else {
        scope.manifests.push(manifest);
    }
    Ok(())
}

/// The complete keyed data snapshot, counting every scope once and excluding schema metadata.
pub fn check_state_size<'a>(
    routing: Option<&Value>,
    scopes: impl IntoIterator<Item = &'a ContextScope>,
) -> Result<(), StorageError> {
    #[derive(Serialize)]
    struct Snapshot<'a> {
        scopes: BTreeMap<&'a str, &'a Map<String, Value>>,
        routing: Option<&'a Value>,
    }
    let snapshot = Snapshot {
        scopes: scopes.into_iter().map(|scope| (scope.scope_id.as_str(), &scope.fields)).collect(),
        routing,
    };
    struct Budget {
        bytes: usize,
        exceeded: bool,
    }
    impl std::io::Write for Budget {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self.bytes.saturating_add(bytes.len());
            if self.bytes >= STATE_LIMIT {
                self.exceeded = true;
                return Err(std::io::Error::other("state exceeds its byte budget"));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut budget = Budget { bytes: 0, exceeded: false };
    let result = serde_json::to_writer(&mut budget, &snapshot);
    if budget.exceeded {
        return Err(StorageError::TooLarge { what: "state".into(), limit: STATE_LIMIT });
    }
    result.map_err(|error| StorageError::NotSerializable("state".into(), error.to_string()))
}

pub mod db {
    //! Dedicated scope records, loaded with the lane and saved in its fenced transaction.

    use lk_agents_abi::storage::context::{ContextScope, ContextScopeWrite};
    use serde_json::Value;

    use crate::StorageError;
    use crate::sql::{Cells, Row, SqlValue, Statement};
    use crate::thread::db::Fence;

    // Scope records are framework-private schema, independent of the platform's v1 thread tables.
    pub const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS context_scopes (
        scope_id TEXT PRIMARY KEY,
        lane TEXT NOT NULL,
        revision INTEGER NOT NULL,
        fields JSONB NOT NULL,
        contracts JSONB NOT NULL,
        manifests JSONB NOT NULL
    )";

    pub fn select(lane: &str) -> Statement {
        Statement {
            sql: "SELECT scope_id, revision, json(fields), json(contracts), json(manifests) \
                  FROM context_scopes WHERE lane = ?1 ORDER BY scope_id"
                .into(),
            params: vec![lane.into()],
        }
    }

    pub fn from_rows(rows: &[Row]) -> Result<Vec<ContextScope>, StorageError> {
        rows.iter()
            .map(|row| {
                let columns =
                    Cells::new(row, &["scope_id", "revision", "fields", "contracts", "manifests"])?;
                let fields: Value = columns.get(2)?;
                let contracts: Value = columns.get(3)?;
                let manifests: Value = columns.get(4)?;
                Ok(ContextScope {
                    scope_id: columns.get(0)?,
                    revision: columns.get(1)?,
                    fields: serde_json::from_value(fields).map_err(decode)?,
                    contracts: serde_json::from_value(contracts).map_err(decode)?,
                    manifests: serde_json::from_value(manifests).map_err(decode)?,
                })
            })
            .collect()
    }

    fn decode(error: serde_json::Error) -> StorageError {
        StorageError::Unavailable(format!("context scope: {error}"))
    }

    /// A revision mismatch fails a NOT NULL constraint, rolling back the enclosing batch. A
    /// mismatched lane fence writes nothing, so the commit reports VersionConflict.
    pub fn write(
        lane: &str,
        write: &ContextScopeWrite,
        fence: Option<&Fence>,
    ) -> Result<Statement, StorageError> {
        if write.scope.scope_id.is_empty()
            || write.expected_revision < 0
            || write.scope.revision <= write.expected_revision
        {
            return Err(StorageError::Invalid("invalid context scope revision write".into()));
        }
        Ok(Statement {
            sql: "INSERT INTO context_scopes (scope_id, lane, revision, fields, contracts, manifests) \
                  SELECT ?1, ?2, \
                    CASE WHEN COALESCE((SELECT revision FROM context_scopes WHERE scope_id = ?1), 0) = ?3 \
                         THEN ?4 ELSE NULL END, jsonb(?5), jsonb(?6), jsonb(?7) \
                  WHERE ?8 IS NULL OR EXISTS \
                    (SELECT 1 FROM lanes WHERE lane = ?2 AND version = ?8 AND lease_token = ?9) \
                  ON CONFLICT(scope_id) DO UPDATE SET \
                    revision = CASE WHEN context_scopes.lane = excluded.lane \
                                    THEN excluded.revision ELSE NULL END, \
                    fields = excluded.fields, contracts = excluded.contracts, manifests = excluded.manifests"
                .into(),
            params: vec![
                write.scope.scope_id.as_str().into(),
                lane.into(),
                write.expected_revision.into(),
                write.scope.revision.into(),
                SqlValue::json(&write.scope.fields),
                SqlValue::json(&write.scope.contracts),
                SqlValue::json(&write.scope.manifests),
                fence.map(|fence| fence.version).into(),
                fence.map(|fence| fence.token.as_str()).into(),
            ],
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::backends::sqlite::{batch, query};
        use crate::context::ContextActivation;
        use lk_agents_abi::storage::context::ContextField;
        use serde_json::json;

        #[test]
        fn scope_saves_are_durable_and_stale_revisions_rollback_the_entire_batch() {
            let mut conn = rusqlite::Connection::open_in_memory().unwrap();
            batch(&mut conn, &crate::thread::db::bootstrap()).unwrap();
            let field = ContextField {
                name: "draft".into(),
                schema: json!({"type": "string"}),
                value: json!("saved"),
            };
            let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
            activation.load("root", "Booking", std::slice::from_ref(&field)).unwrap();
            activation.save("root", "Booking", 0, &[field]).unwrap();
            let first = activation.writes().remove(0);
            batch(&mut conn, &[write("desk", &first, None).unwrap()]).unwrap();
            let saved = from_rows(&query(&conn, &select("desk")).unwrap()).unwrap();
            assert_eq!(saved, vec![first.scope.clone()]);
            assert!(from_rows(&query(&conn, &select("other")).unwrap()).unwrap().is_empty());

            let stale = write("desk", &first, None).unwrap();
            let result =
                batch(&mut conn, &[Statement::raw("CREATE TABLE partial (body TEXT)"), stale]);
            assert!(result.is_err());
            assert!(
                query(
                    &conn,
                    &Statement::raw("SELECT name FROM sqlite_master WHERE name = 'partial'")
                )
                .unwrap()
                .is_empty()
            );
            assert_eq!(from_rows(&query(&conn, &select("desk")).unwrap()).unwrap(), saved);
        }
    }
}

pub mod store {
    //! A lane's context field store as the host uses it: snapshot staging in a turn, fenced direct
    //! saves outside one, and the application transactions over the same database.

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
    use crate::sql::{Row, Statement};
    use crate::thread::run::RunOutcome;
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
                    "the managed turn this transaction joined has ended; nothing was written"
                        .into(),
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

        fn closed(&self) -> bool {
            self.activation.lock().closed()
        }

        fn close(&self) {
            self.activation.lock().close();
        }
    }
}

pub mod transaction {
    //! Application SQL (docs/context.md#application-records-and-transactions). Both data planes
    //! execute reads and atomic batches, never an interactive transaction: reads observe committed
    //! records, writes are staged and applied together when the transaction ends, so a read after a
    //! write in the same transaction is refused rather than answered without that write.

    use parking_lot::Mutex;

    use lk_agents_abi::storage::context::{AppStatement, AppTransaction};
    use serde_json::{Number, Value};

    use super::store::{ContextStore, Database};
    use crate::StorageError;
    use crate::sql::{SqlValue, Statement};

    /// One application transaction: independent, or joining the managed turn it began in.
    pub struct Transaction {
        database: Database,
        /// The managed turn the writes join, and that turn's generation.
        joins: Option<(ContextStore, u64)>,
        state: Mutex<State>,
    }

    #[derive(Default)]
    struct State {
        staged: Vec<AppStatement>,
        ended: bool,
    }

    impl Transaction {
        pub(crate) fn new(database: Database, joins: Option<(ContextStore, u64)>) -> Self {
            Self { database, joins, state: Mutex::default() }
        }

        fn state(&self) -> parking_lot::MutexGuard<'_, State> {
            self.state.lock()
        }

        fn ensure_open(state: &State) -> Result<(), StorageError> {
            if state.ended {
                return Err(StorageError::Invalid("the transaction has ended".into()));
            }
            Ok(())
        }
    }

    impl AppTransaction for Transaction {
        async fn execute(&self, sql: String, params: Vec<Value>) -> Result<(), StorageError> {
            let write = AppStatement { sql, params };
            statement(&write)?;
            let mut state = self.state();
            Self::ensure_open(&state)?;
            state.staged.push(write);
            Ok(())
        }

        async fn fetch(
            &self,
            sql: String,
            params: Vec<Value>,
        ) -> Result<Vec<Vec<Value>>, StorageError> {
            let read = statement(&AppStatement { sql, params })?;
            {
                let state = self.state();
                Self::ensure_open(&state)?;
                if !state.staged.is_empty() {
                    return Err(StorageError::Invalid(
                        "a transaction reads committed records only: read before its first write, \
                         or in a transaction after this one commits"
                            .into(),
                    ));
                }
            }
            let rows = self.database.query(read).await?;
            rows.iter().map(|row| row.iter().map(cell).collect()).collect()
        }

        async fn commit(&self) -> Result<(), StorageError> {
            let staged = {
                let mut state = self.state();
                Self::ensure_open(&state)?;
                state.ended = true;
                std::mem::take(&mut state.staged)
            };
            if staged.is_empty() {
                return Ok(());
            }
            match &self.joins {
                Some((store, turn)) => store.join_turn(*turn, staged).await,
                None => {
                    let statements = staged.iter().map(statement).collect::<Result<_, _>>()?;
                    self.database.batch(statements).await
                }
            }
        }

        fn rollback(&self) {
            let mut state = self.state();
            state.ended = true;
            state.staged.clear();
        }
    }

    /// An application statement as the data plane executes it.
    pub fn statement(s: &AppStatement) -> Result<Statement, StorageError> {
        let params = s.params.iter().map(param).collect::<Result<_, _>>()?;
        Ok(Statement { sql: s.sql.clone().into(), params })
    }

    fn param(value: &Value) -> Result<SqlValue, StorageError> {
        let invalid = || {
            StorageError::Invalid(format!(
                "SQL parameter {value}: expected null, a boolean, an integer, a float or a string"
            ))
        };
        Ok(match value {
            Value::Null => SqlValue::Null,
            Value::Bool(b) => (*b).into(),
            Value::Number(n) if n.is_f64() => SqlValue::Double(n.as_f64().ok_or_else(invalid)?),
            Value::Number(n) => SqlValue::Int(n.as_i64().ok_or_else(invalid)?),
            Value::String(s) => SqlValue::Text(s.clone()),
            Value::Array(_) | Value::Object(_) => return Err(invalid()),
        })
    }

    fn cell(value: &SqlValue) -> Result<Value, StorageError> {
        Ok(match value {
            SqlValue::Null => Value::Null,
            SqlValue::Int(i) => (*i).into(),
            SqlValue::Double(d) => Number::from_f64(*d).map_or(Value::Null, Value::Number),
            SqlValue::Text(s) => s.clone().into(),
            SqlValue::Blob(_) => {
                return Err(StorageError::Invalid(
                    "a BLOB column cannot be read as an application row value: select hex(...)"
                        .into(),
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn field(name: &str, value: Value) -> ContextField {
        ContextField { name: name.into(), schema: json!({"type": "string"}), value }
    }

    #[test]
    fn scope_revisions_and_contracts_survive_restart_without_colliding() {
        let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
        let first = vec![field("draft", json!("first"))];
        let second = vec![field("draft", json!("second"))];
        for (id, fields) in [("booking_a", &first), ("booking_b", &second)] {
            let loaded = activation.load(id, "Booking", fields).unwrap();
            activation.save(id, "Booking", loaded.revision, fields).unwrap();
        }
        let scopes = activation.writes().into_iter().map(|write| write.scope).collect();
        let mut restored = ContextActivation::new(scopes, None, false).unwrap();
        let a = restored.load("booking_a", "Booking", &second).unwrap();
        let b = restored.load("booking_b", "Booking", &first).unwrap();
        assert_eq!((a.revision, a.fields["draft"].clone()), (1, json!("first")));
        assert_eq!((b.revision, b.fields["draft"].clone()), (1, json!("second")));
        restored.save("booking_a", "Booking", a.revision, &second).unwrap();
        assert!(matches!(
            restored.save("booking_a", "Booking", a.revision, &first),
            Err(StorageError::VersionConflict { .. })
        ));
        assert_eq!(restored.writes()[0].expected_revision, 1);
        assert!(matches!(
            restored.load("booking_a", "Booking", &[field("renamed", json!("default"))]),
            Err(StorageError::Invalid(_))
        ));
        let mut incompatible = field("draft", json!(0));
        incompatible.schema = json!({"type": "integer"});
        assert!(matches!(
            restored.load("booking_a", "Shared", &[incompatible]),
            Err(StorageError::Invalid(_))
        ));
        assert_eq!(restored.writes()[0].scope.fields["draft"], json!("second"));
    }

    #[test]
    fn aggregate_size_refusal_preserves_every_staged_scope() {
        let mut activation = ContextActivation::new(Vec::new(), None, false).unwrap();
        let fields = vec![field("draft", json!("é\\\"".repeat(6000)))];
        let a = activation.load("a", "Booking", &fields).unwrap();
        activation.save("a", "Booking", a.revision, &fields).unwrap();
        let before = activation.writes();
        let b = activation.load("b", "Booking", &fields).unwrap();
        assert!(matches!(
            activation.save("b", "Booking", b.revision, &fields),
            Err(StorageError::TooLarge { .. })
        ));
        assert_eq!(activation.writes(), before);
        activation.close();
        assert!(activation.load("a", "Booking", &fields).is_err());
    }

    #[test]
    fn the_complete_encoded_state_must_be_strictly_under_the_limit() {
        let mut scope = ContextScope { scope_id: "root".into(), ..Default::default() };
        let empty = serde_json::to_vec(&json!({
            "scopes": {"root": {"draft": ""}}, "routing": null,
        }))
        .unwrap()
        .len();
        scope.fields.insert("draft".into(), json!("x".repeat(STATE_LIMIT - empty - 1)));
        assert!(check_state_size(None, [&scope]).is_ok());
        scope.fields.insert("draft".into(), json!("x".repeat(STATE_LIMIT - empty)));
        assert!(matches!(check_state_size(None, [&scope]), Err(StorageError::TooLarge { .. })));

        scope.fields.insert("draft".into(), json!("x".repeat(32_000)));
        let routing = json!({"frames": "x".repeat(34_000)});
        assert!(matches!(
            check_state_size(Some(&routing), [&scope]),
            Err(StorageError::TooLarge { .. })
        ));
    }
}

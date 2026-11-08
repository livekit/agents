//! Persisted context scopes and the activation's field staging boundary.

use serde_json::{Map, Value};

use super::StorageError;

/// The host's declared field contract and its already validated JSON value.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextField {
    pub name: String,
    pub schema: Value,
    pub value: Value,
}

/// One application's declaration, retained to detect removed or renamed fields on reload.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextManifest {
    pub identity: String,
    pub fields: Vec<String>,
}

/// One scope's committed data. Revisions and schema metadata are separate from app values.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextScope {
    pub scope_id: String,
    pub revision: i64,
    pub fields: Map<String, Value>,
    pub contracts: Map<String, Value>,
    pub manifests: Vec<ContextManifest>,
}

/// A scope replacement conditional on the revision this activation loaded.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextScopeWrite {
    pub expected_revision: i64,
    pub scope: ContextScope,
}

/// One application SQL statement a managed transaction staged for the turn's commit. `params`
/// are JSON scalars: null, booleans, numbers and strings.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AppStatement {
    pub sql: String,
    pub params: Vec<Value>,
}

/// Values to hydrate in the host after the core accepted their schema contracts.
#[crate::host(record(module = "storage_context", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ContextValues {
    pub revision: i64,
    pub fields: Map<String, Value>,
}

/// Activation state and revision/schema decisions. Application type validation stays in the host.
#[crate::host(class(name = "_core_ContextActivation"))]
pub trait ContextActivation: Send + Sync + Sized + 'static {
    #[host(class(name = "_core_AppTransaction"))]
    type Transaction: Send + Sync + 'static;

    #[host(new)]
    fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError>;

    /// Check manifests before restoring values or accepting defaults for absent fields.
    #[host(await)]
    async fn load(
        &self,
        scope_id: String,
        identity: String,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError>;

    /// Stage the validated fields; a stale activation cannot overwrite a newer revision.
    #[host(await)]
    async fn save(
        &self,
        scope_id: String,
        identity: String,
        revision: i64,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError>;

    #[host(getter)]
    fn writes(&self) -> Vec<ContextScopeWrite>;

    /// The application statements managed transactions staged for the turn's commit.
    #[host(getter)]
    fn statements(&self) -> Vec<AppStatement>;

    /// Application SQL (docs/context.md#application-records-and-transactions): `independent`
    /// commits when the transaction ends; otherwise it joins the managed turn's commit, or
    /// commits like an independent one outside a managed turn.
    #[host(await)]
    async fn transaction(&self, independent: bool) -> Result<Self::Transaction, StorageError>;

    #[host(call)]
    fn acknowledge(&self);

    #[host(getter)]
    fn closed(&self) -> bool;

    /// End the activation, refusing subsequent loads or saves.
    #[host(call)]
    fn close(&self);
}

/// One application transaction. Reads observe committed records; writes are staged and
/// applied atomically when the transaction ends, so a read after a write in the same
/// transaction is refused.
#[crate::host(class(name = "_core_AppTransaction"))]
pub trait AppTransaction: Send + Sync + Sized + 'static {
    /// Stages one statement; `params` are JSON scalars.
    #[host(await)]
    async fn execute(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<(), StorageError>;

    /// One read's rows, each a list of column values in select order.
    #[host(await)]
    async fn fetch(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<Vec<Vec<Value>>, StorageError>;

    /// The first row of one read, `None` when it has none.
    #[host(await)]
    async fn fetch_one(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<Option<Vec<Value>>, StorageError>;

    /// Ends the transaction: its writes commit (independent) or join the turn's commit.
    #[host(await)]
    async fn commit(&self) -> Result<(), StorageError>;

    /// Ends the transaction, discarding its writes.
    #[host(call)]
    fn rollback(&self);
}

/// `storage.Transaction`: application SQL in one transaction
/// (docs/context.md#application-records-and-transactions).
///
/// Reads observe committed records. Writes are staged and applied together when the
/// transaction ends, so read before the first write: a read after one is refused.
#[crate::host(facade(
    path = "livekit.agents.storage",
    name = "Transaction",
    native = AppTransaction,
    from_native
))]
pub trait TransactionFacade {
    /// Runs one statement when the transaction commits.
    #[host(annotation(params = "collections.abc.Sequence[..storage.context.SqlValue] | None"))]
    async fn execute(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<(), StorageError>;

    /// The rows of one read.
    #[host(annotation(
        params = "collections.abc.Sequence[..storage.context.SqlValue] | None",
        ret = "list[..storage.context.Row]"
    ))]
    async fn fetch(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<Vec<Vec<Value>>, StorageError>;

    /// The first row of one read, `None` when it has none.
    #[host(annotation(
        params = "collections.abc.Sequence[..storage.context.SqlValue] | None",
        ret = "..storage.context.Row | None"
    ))]
    async fn fetch_one(
        &self,
        sql: String,
        #[host(default = None)] params: Option<Vec<Value>>,
    ) -> Result<Option<Vec<Value>>, StorageError>;
}

/// `storage.ContextStore`'s transactions, through its activation.
#[crate::host(facade(path = "livekit.agents.storage.context", name = "ContextStore", extended))]
pub trait ContextStoreFacade {
    /// The store's activation, which the store sets.
    #[host(delegate = "_activation")]
    type Activation: ContextActivation;

    /// A transaction, committed as the block ends and rolled back when it raises.
    #[host(
        via = Activation,
        native = transaction,
        exit = AppTransaction::commit,
        abort = AppTransaction::rollback,
        returns = "..storage.Transaction"
    )]
    async fn _transaction(&self, #[host(kw)] independent: bool) -> Result<(), StorageError>;
}

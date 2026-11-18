//! Native bound control resources. Neither credentials nor these capabilities are serialized.

use std::sync::Arc;

use crate::BoxFuture;
use crate::a2a::client::AgentClientError;
use crate::delegation::remote::{RemoteRecovery, RemoteTaskRef};
use crate::session::snapshot::SessionSnapshot;
use crate::storage::StorageError;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RemoteControlRef {
    pub remote_id: String,
    pub endpoint: String,
    pub task: RemoteTaskRef,
}

/// A wait-only receipt. Its producer lives in the admitted owner task or native command.
/// Concurrent explicit controls share the same immutable result while that attempt is pending.
pub trait RemoteCancellation: Send + Sync {
    fn pending(&self) -> bool;
    fn wait(&self) -> BoxFuture<'_, Result<(), AgentClientError>>;
}

/// Installed only by an explicit binding declaration. Loading or serializing state cannot
/// create this capability, and closing the context store invalidates it.
pub trait BoundSessionControl: Send + Sync {
    fn thread_id(&self) -> Result<Option<String>, StorageError>;
    fn ended(&self) -> BoxFuture<'_, ()>;
    fn commands(&self) -> Result<Option<Arc<dyn super::SessionCommands>>, StorageError>;

    /// Publish a candidate receipt without doing I/O. Only the returned candidate's owner
    /// starts work, after this method has released its lock.
    fn claim_cancellation(
        &self,
        target: &RemoteControlRef,
        candidate: Arc<dyn RemoteCancellation>,
    ) -> Result<Arc<dyn RemoteCancellation>, StorageError>;

    fn acquire(&self) -> BoxFuture<'_, Result<ControlOwner, StorageError>>;
}

/// One short fenced owner. It reloads the committed session under its lease and checkpoints
/// framework state only. Application field writes, SQL and delivered history are preserved.
/// Dropping the scope releases its storage ownership through the existing cleanup owner.
pub trait SessionControlScope: Send {
    fn thread_id(&self) -> &str;
    fn snapshot(&self) -> &SessionSnapshot;
    fn checkpoint(&mut self, state: SessionSnapshot) -> BoxFuture<'_, Result<(), StorageError>>;
    fn close(self: Box<Self>) -> BoxFuture<'static, Result<(), StorageError>>;
}

/// The current serialized conversation, or one acquired state-only fenced owner.
/// A control never opens a second lease while an existing owner can admit its journal I/O.
pub enum ControlOwner {
    Active(Arc<dyn super::SessionCommands>),
    Fenced(Box<dyn SessionControlScope>),
}

/// Resource admission accompanies the core's pure route without crossing the host boundary.
#[derive(Clone)]
pub enum RemoteCancellationAdmission {
    Confirmed,
    /// One already-owned live or retained operation; this is a wait-only receipt.
    Pending(Arc<dyn RemoteCancellation>),
    Retained {
        recovery: Option<RemoteRecovery>,
    },
}

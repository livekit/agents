//! Inspection of retained remote observations. Reads never dispatch protocol operations.

use std::collections::BTreeMap;

use crate::a2a::protocol::{Message, Task};
use crate::delegation::remote::{A2AJournalEntry, InputRequest};
use crate::storage::StorageError;

/// Retained communication with one configured remote, scoped to the selected conversation.
#[crate::host(class(
    name = "RemoteAgentState",
    module = "livekit.agents.storage.session_state",
    pid = false
))]
pub trait RemoteAgentView: Clone + Send + Sync + Sized + 'static {
    #[host(class(name = "RemoteTask"))]
    type Task: RemoteTaskView;
    #[host(class(name = "RemoteContextState"))]
    type Context: RemoteContextView<Task = Self::Task>;

    /// Pinned destination, without credentials.
    #[host(getter)]
    fn endpoint(&self) -> Result<String, StorageError>;
    /// Observed communication in local journal order.
    #[host(getter)]
    fn events(&self) -> Result<Vec<A2AJournalEntry>, StorageError>;
    /// Observed messages in either direction, indexed by message ID; includes contextless replies.
    #[host(getter)]
    fn messages(&self) -> Result<BTreeMap<String, Message>, StorageError>;
    /// Observed tasks, indexed by task ID; their protocol history may be absent.
    #[host(getter)]
    fn tasks(&self) -> Result<BTreeMap<String, Self::Task>, StorageError>;
    /// Grouping of the same observations by context ID, without another message history.
    #[host(getter)]
    fn contexts(&self) -> Result<BTreeMap<String, Self::Context>, StorageError>;
}

/// Observed messages and tasks grouped by their remote context ID.
#[crate::host(class(
    name = "RemoteContextState",
    module = "livekit.agents.storage.session_state",
    pid = false
))]
pub trait RemoteContextView: Clone + Send + Sync + Sized + 'static {
    #[host(class(name = "RemoteTask"))]
    type Task: RemoteTaskView;

    #[host(getter)]
    fn messages(&self) -> Result<BTreeMap<String, Message>, StorageError>;
    #[host(getter)]
    fn tasks(&self) -> Result<BTreeMap<String, Self::Task>, StorageError>;
}

/// A task's current protocol observation and its task-owned local routing facts.
#[crate::host(class(
    name = "RemoteTask",
    module = "livekit.agents.storage.session_state",
    pid = false
))]
pub trait RemoteTaskView: Clone + Send + Sync + Sized + 'static {
    /// The observed SDK Task, including an unspecified status when only deltas were received.
    #[host(getter)]
    fn task(&self) -> Result<Task, StorageError>;
    /// Retained input occurrences, indexed by occurrence ID independently of question wording.
    #[host(getter)]
    fn input_requests(&self) -> Result<BTreeMap<String, InputRequest>, StorageError>;
    /// Explicit local intent; only an observed task state confirms remote cancellation.
    #[host(getter)]
    fn cancel_requested(&self) -> Result<bool, StorageError>;
}

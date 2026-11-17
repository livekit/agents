use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::a2a::protocol::{Message, Task};
use lk_agents_abi::delegation::remote::{A2AJournalEntry, InputRequest, RemoteAgentState};
use lk_agents_abi::session::remote::{RemoteAgentView, RemoteContextView, RemoteTaskView};

use super::{SessionState, State, StorageError, invalid};

/// Native handles keep the same bound or detached owner as the enclosing session view.
#[derive(Clone)]
pub struct RemoteAgent {
    state: SessionState,
    remote_id: String,
}

#[derive(Clone)]
pub struct RemoteContext {
    remote: RemoteAgent,
    context_id: String,
}

#[derive(Clone)]
pub struct RemoteTask {
    remote: RemoteAgent,
    task_id: String,
}

impl SessionState {
    pub(super) fn remote_views(&self) -> BTreeMap<String, RemoteAgent> {
        let envelope = self.0.lock();
        let State::Conversation { state, .. } = &envelope.state else {
            return BTreeMap::new();
        };
        state
            .remote_agents
            .keys()
            .map(|id| (id.clone(), RemoteAgent { state: self.clone(), remote_id: id.clone() }))
            .collect()
    }
}

impl RemoteAgent {
    fn read<T>(&self, select: impl FnOnce(&RemoteAgentState) -> T) -> Result<T, StorageError> {
        let envelope = self.state.0.lock();
        let State::Conversation { state, .. } = &envelope.state else {
            return Err(invalid("the conversation no longer retains this remote agent"));
        };
        let remote = state
            .remote_agents
            .get(&self.remote_id)
            .ok_or_else(|| invalid("the conversation no longer retains this remote agent"))?;
        Ok(select(remote))
    }

    fn task_views(
        &self,
        state: &RemoteAgentState,
        context: Option<&str>,
    ) -> BTreeMap<String, RemoteTask> {
        state
            .tasks
            .iter()
            .filter(|(_, task)| context.is_none_or(|context| task.task.context_id == context))
            .map(|(id, _)| (id.clone(), RemoteTask { remote: self.clone(), task_id: id.clone() }))
            .collect()
    }
}

impl RemoteAgentView for RemoteAgent {
    type Task = RemoteTask;
    type Context = RemoteContext;

    fn endpoint(&self) -> Result<String, StorageError> {
        self.read(|state| state.endpoint.clone())
    }

    fn events(&self) -> Result<Vec<A2AJournalEntry>, StorageError> {
        self.read(|state| state.events.clone())
    }

    fn messages(&self) -> Result<BTreeMap<String, Message>, StorageError> {
        self.read(|state| state.messages.clone())
    }

    fn tasks(&self) -> Result<BTreeMap<String, Self::Task>, StorageError> {
        self.read(|state| self.task_views(state, None))
    }

    fn contexts(&self) -> Result<BTreeMap<String, Self::Context>, StorageError> {
        self.read(|state| {
            let ids: BTreeSet<_> = state
                .messages
                .values()
                .filter_map(|message| message.context_id.as_ref())
                .chain(state.tasks.values().map(|task| &task.task.context_id))
                .collect();
            ids.into_iter()
                .map(|id| {
                    (id.clone(), RemoteContext { remote: self.clone(), context_id: id.clone() })
                })
                .collect()
        })
    }
}

impl RemoteContextView for RemoteContext {
    type Task = RemoteTask;

    fn messages(&self) -> Result<BTreeMap<String, Message>, StorageError> {
        self.remote.read(|state| {
            state
                .messages
                .iter()
                .filter(|(_, message)| message.context_id.as_ref() == Some(&self.context_id))
                .map(|(id, message)| (id.clone(), message.clone()))
                .collect()
        })
    }

    fn tasks(&self) -> Result<BTreeMap<String, Self::Task>, StorageError> {
        self.remote.read(|state| self.remote.task_views(state, Some(&self.context_id)))
    }
}

impl RemoteTask {
    fn read<T>(
        &self,
        select: impl FnOnce(&lk_agents_abi::delegation::remote::RemoteTaskState) -> T,
    ) -> Result<T, StorageError> {
        self.remote
            .read(|state| state.tasks.get(&self.task_id).map(select))?
            .ok_or_else(|| invalid("the conversation no longer retains this remote task"))
    }
}

impl RemoteTaskView for RemoteTask {
    fn task(&self) -> Result<Task, StorageError> {
        self.read(|state| state.task.clone())
    }

    fn input_requests(&self) -> Result<BTreeMap<String, InputRequest>, StorageError> {
        self.read(|state| state.input_requests.clone())
    }

    fn cancel_requested(&self) -> Result<bool, StorageError> {
        self.read(|state| state.cancel_requested)
    }
}

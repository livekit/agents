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

    type Env = lk_agents_net::env::ProcessEnv;

    async fn cancel(
        &self,
        env: &Self::Env,
    ) -> Result<(), lk_agents_abi::a2a::client::AgentClientError> {
        use lk_agents_abi::a2a::client::AgentClientError;
        use lk_agents_abi::session::control::RemoteControlRef;
        self.remote
            .state
            .ensure_control_process()
            .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?;
        let expected = self.remote.state.0.lock().clone();
        let super::State::Conversation { state, .. } = &expected.state else {
            return Err(AgentClientError::InvalidInput(
                "the conversation no longer retains this task".into(),
            ));
        };
        let remote = state.remote_agents.get(&self.remote.remote_id).ok_or_else(|| {
            AgentClientError::InvalidInput(
                "the conversation no longer retains this remote agent".into(),
            )
        })?;
        let saved = remote.tasks.get(&self.task_id).ok_or_else(|| {
            AgentClientError::InvalidInput("the conversation no longer retains this task".into())
        })?;
        let task = lk_agents_abi::delegation::remote::RemoteTaskRef {
            task_id: saved.task.id.clone(),
            context_id: saved.task.context_id.clone(),
        };
        let evidence = remote.cancellation_evidence(&task);
        let target = RemoteControlRef {
            remote_id: self.remote.remote_id.clone(),
            endpoint: remote.endpoint.clone(),
            task,
        };
        let store = self.remote.state.2.lock().clone().ok_or_else(|| {
            AgentClientError::InvalidInput(
                "remote task cancellation requires an explicitly bound context store".into(),
            )
        })?;
        let control = env.remote_control.ok_or_else(|| {
            AgentClientError::InvalidInput("this runtime cannot execute bound task controls".into())
        })?;
        let result = control(
            env.clone(),
            store.control_binding(self.remote.state.clone()),
            target,
            evidence,
        )
        .wait()
        .await;
        // A separate managed activation may have joined another binding's receipt. Read the
        // committed framework state without a lease/write, then publish only by exact-view
        // CAS. Neither newer live progress nor local edits can be rolled back by this read.
        if !self.remote.state.control_unchanged(&store, &expected) {
            return result;
        }
        let refreshed =
            match lk_agents_abi::storage::context::ContextActivation::load_session(&store).await {
                Ok(Some(value)) => match super::SessionState::from_value(value) {
                    Ok(checked) => {
                        self.remote.state.publish_control_if(&store, &expected, &checked);
                        Ok(())
                    }
                    Err(error) => Err(AgentClientError::Protocol(error.to_string())),
                },
                Ok(None) => Ok(()),
                Err(error) => Err(AgentClientError::Protocol(error.to_string())),
            };
        result.and(refreshed)
    }
}

//! Generated host classes forward here; protocol records are converted once at the boundary.

use lk_agents_abi::a2a::client::{
    self as port, AgentClientError, Artifact, MessageInput, MessageMetadata, TaskUpdate,
};
use lk_agents_abi::{ApiConnectOptions, ChatItem, HOST_TEARDOWN_TIMEOUT};
use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::a2a::client::ClientError;
use lk_agents_core::a2a::rpc::TaskRef;

use super::{AgentClient, Reply};
use crate::env::ProcessEnv;
use crate::source::{PullError, Pulled};

pub(super) fn client_error(error: ClientError) -> AgentClientError {
    match error {
        ClientError::Transport(error) => AgentClientError::Provider(error.into()),
        ClientError::InvalidInput(message) => AgentClientError::InvalidInput(message),
        error => AgentClientError::Protocol(error.to_string()),
    }
}

impl port::AgentClient for AgentClient {
    type Reply = Reply;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        endpoint_url: String,
        token: Option<String>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self, AgentClientError> {
        Self::new(env, endpoint_url, token, conn_options.unwrap_or_default()).map_err(client_error)
    }

    async fn connect(&self) -> Result<port::AgentCard, AgentClientError> {
        let card = AgentClient::connect(self).await.map_err(client_error)?;
        lk_agents_core::a2a::client::tool_definition(&card).map_err(client_error)
    }

    fn offers_extension(&self) -> bool {
        AgentClient::offers_extension(self)
    }

    async fn send(
        &self,
        message: MessageInput,
        thread_id: Option<String>,
        task_id: Option<String>,
        metadata: Option<MessageMetadata>,
        chat_ctx: Option<Vec<ChatItem>>,
        traceparent: Option<String>,
    ) -> Result<Self::Reply, AgentClientError> {
        let mut message = message_to_wire(message)?;
        if let Some(metadata) = metadata {
            message
                .metadata
                .get_or_insert_default()
                .extend(lk_agents_core::a2a::client::metadata_to_wire(metadata)?);
        }
        let history = chat_ctx
            .map(|items| CallerHistory::from_chat_items(&items, None, &HistorySharing::default()))
            .transpose()
            .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?;
        AgentClient::send(
            self,
            message,
            thread_id.as_deref(),
            task_id.as_deref(),
            history.as_ref(),
            traceparent.as_deref(),
        )
        .await
        .map_err(client_error)
    }

    async fn _core_subscribe(
        &self,
        task_id: String,
        thread_id: String,
        traceparent: Option<String>,
    ) -> Result<Self::Reply, AgentClientError> {
        self.subscribe(TaskRef { id: task_id, context_id: thread_id }, None, traceparent.as_deref())
            .await
            .map_err(client_error)
    }

    async fn cancel(
        &self,
        task_id: String,
        thread_id: Option<String>,
        reason: Option<String>,
    ) -> Result<TaskUpdate, AgentClientError> {
        let thread_id = thread_id
            .or_else(|| self.inner.core.lock().last_context().map(str::to_owned))
            .ok_or_else(|| {
                AgentClientError::InvalidInput(
                    "cancel needs a thread_id before the first reply".into(),
                )
            })?;
        let update = AgentClient::cancel(
            self,
            TaskRef { id: task_id, context_id: thread_id },
            reason.as_deref(),
        )
        .await
        .map_err(client_error)?;
        Ok(update_from_wire(update))
    }

    async fn aclose(&self) {
        AgentClient::aclose(self).await;
    }
}

impl port::Reply for Reply {
    fn task_id(&self) -> String {
        self.task_id.clone()
    }
    fn thread_id(&self) -> String {
        self.thread_id.clone()
    }
    async fn next(&self) -> Result<Option<TaskUpdate>, AgentClientError> {
        match self.pull().await {
            Ok(Pulled::Item(update)) => Ok(Some(update_from_wire(update))),
            Ok(Pulled::End | Pulled::Cancelled) => Ok(None),
            Err(PullError::Overlap) => Err(AgentClientError::Overlap),
            Err(PullError::Failed(error)) => Err(AgentClientError::Provider(error)),
        }
    }
    async fn aclose(&self) {
        Reply::aclose(self, self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    }
}

pub fn message_to_wire(input: MessageInput) -> Result<a2a::Message, AgentClientError> {
    lk_agents_core::a2a::client::message_to_wire(input, crate::strings::shortuuid("msg_"))
}

pub use lk_agents_core::a2a::client::message_from_wire;

pub fn update_from_wire(update: lk_agents_core::a2a::client::TaskUpdate) -> TaskUpdate {
    TaskUpdate {
        state: match update.state {
            lk_agents_abi::storage::TaskState::Submitted => "submitted",
            lk_agents_abi::storage::TaskState::Working => "working",
            lk_agents_abi::storage::TaskState::InputRequired => "input-required",
            lk_agents_abi::storage::TaskState::Completed => "completed",
            lk_agents_abi::storage::TaskState::Failed => "failed",
            lk_agents_abi::storage::TaskState::Canceled => "canceled",
            lk_agents_abi::storage::TaskState::Rejected => "rejected",
        }
        .into(),
        text: update.text,
        task_id: update.task_id,
        context_id: update.context_id,
        metadata: update.metadata,
        artifacts: update
            .artifacts
            .into_iter()
            .map(|artifact| Artifact {
                artifact_id: artifact.artifact_id,
                name: artifact.name,
                description: artifact.description,
                parts: artifact
                    .parts
                    .into_iter()
                    .map(lk_agents_core::a2a::client::part_from_wire)
                    .collect(),
                metadata: artifact
                    .metadata
                    .map(|metadata| metadata.into_iter().collect())
                    .unwrap_or_default(),
            })
            .collect(),
    }
}

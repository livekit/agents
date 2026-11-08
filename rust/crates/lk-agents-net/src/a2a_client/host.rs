use lk_agents_abi::a2a::client::{
    self as port, AgentClientError, Artifact, MessageInput, MessageMetadata, TaskUpdate,
};
use lk_agents_abi::{ApiConnectOptions, ChatItem, HOST_TEARDOWN_TIMEOUT};
use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::a2a::rpc::TaskRef;

use super::{AgentClient, Reply};
use crate::env::ProcessEnv;
use crate::source::Pulled;

impl port::AgentClient for AgentClient {
    type Reply = Reply;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        endpoint_url: String,
        token: Option<String>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self, AgentClientError> {
        Ok(Self::new(env, endpoint_url, token, conn_options.unwrap_or_default())?)
    }

    async fn connect(&self) -> Result<port::AgentCard, AgentClientError> {
        let card = AgentClient::connect(self).await?;
        Ok(lk_agents_core::a2a::client::tool_definition(&card)?)
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
        cx: lk_agents_telemetry::Cx,
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
        Ok(AgentClient::send(
            self,
            message,
            thread_id.as_deref(),
            task_id.as_deref(),
            history.as_ref(),
            &cx,
        )
        .await?)
    }

    async fn _core_subscribe(
        &self,
        task_id: String,
        thread_id: String,
        cx: lk_agents_telemetry::Cx,
    ) -> Result<Self::Reply, AgentClientError> {
        Ok(self.subscribe(TaskRef { id: task_id, context_id: thread_id }, None, &cx).await?)
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
        .await?;
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
        let pull = self.try_pull().ok_or(AgentClientError::Overlap)?;
        Ok(match pull.await? {
            Pulled::Item(update) => Some(update_from_wire(update)),
            Pulled::End | Pulled::Cancelled => None,
        })
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
        state: update.state,
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

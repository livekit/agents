use super::*;
use crate::text_session::live::OwnerCommand;
use lk_agents_abi::session::snapshot::SessionSnapshot;

impl CommandPort {
    pub(crate) async fn cancel_live_tasks(
        &self,
        tasks: Vec<String>,
    ) -> Result<Vec<String>, StorageError> {
        let owner = self.live.as_ref().ok_or_else(|| {
            StorageError::Invalid("the conversation has no live endpoint owner".into())
        })?;
        let (done, receipt) = signal::once();
        owner
            .owner
            .send(OwnerCommand::CancelTasks { tasks, done })
            .await
            .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
        receipt.wait().await.map_err(|_| {
            StorageError::Invalid("the owner ended before applying task cancellation".into())
        })?
    }

    pub(crate) async fn close_live_owner(&self) -> Result<(), StorageError> {
        let owner = self.live.as_ref().ok_or_else(|| {
            StorageError::Invalid("the conversation has no live endpoint owner".into())
        })?;
        let (done, receipt) = signal::once();
        owner
            .owner
            .send(OwnerCommand::Close { done })
            .await
            .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
        receipt.wait().await.map_err(|_| {
            StorageError::Invalid("the conversation owner ended before cleanup completed".into())
        })?
    }

    pub(crate) async fn prepare_dialogue(
        &self,
        source: String,
        messages: Vec<lk_agents_abi::ChatMessage>,
    ) -> Result<SessionSnapshot, StorageError> {
        let owner = self.live.as_ref().ok_or_else(|| {
            StorageError::Invalid("the conversation has no live endpoint owner".into())
        })?;
        let (done, receipt) = signal::once();
        owner
            .owner
            .send(OwnerCommand::PrepareDialogue { source, messages, done })
            .await
            .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
        receipt.wait().await.map_err(|_| {
            StorageError::Invalid("the owner ended before preparing dialogue".into())
        })?
    }

    pub(crate) async fn commit_dialogue(&self, source: String) -> Result<(), StorageError> {
        let owner = self.live.as_ref().ok_or_else(|| {
            StorageError::Invalid("the conversation has no live endpoint owner".into())
        })?;
        let (done, receipt) = signal::once();
        owner
            .owner
            .send(OwnerCommand::CommitDialogue { source, done })
            .await
            .map_err(|_| StorageError::Invalid("the conversation owner ended".into()))?;
        receipt.wait().await.map_err(|_| {
            StorageError::Invalid("the owner ended before admitting committed dialogue".into())
        })?
    }
}

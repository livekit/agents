//! The live session lends its command sender, never its durable state or a host callback.

use super::VoiceSession;
use crate::session::CommandPort;
use lk_agents_abi::session::{InputRequestRef, SessionCommands, SessionInputCommand, ToolCallRef};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{BoxFuture, ChatMessage};

impl VoiceSession {
    fn conversation(&self) -> Result<CommandPort, StorageError> {
        self.running(|running| Ok(running.conversation.clone()))
            .map_err(|error| StorageError::Invalid(error.to_string()))
    }
}

impl SessionCommands for VoiceSession {
    fn checkpoint(&self) -> BoxFuture<'_, Result<serde_json::Value, StorageError>> {
        Box::pin(async move { self.conversation()?.checkpoint().await })
    }

    fn held_input<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> BoxFuture<'a, Result<Option<InputRequestRef>, StorageError>> {
        Box::pin(async move { self.conversation()?.held_input(remote_id, endpoint).await })
    }

    fn response<'a>(
        &'a self,
        reference: &'a InputRequestRef,
    ) -> BoxFuture<'a, Result<ChatMessage, StorageError>> {
        Box::pin(async move { self.conversation()?.response(reference).await })
    }

    fn input<'a>(
        &'a self,
        command: SessionInputCommand,
    ) -> BoxFuture<'a, Result<(), StorageError>> {
        Box::pin(async move { self.conversation()?.input(command).await })
    }

    fn recover<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        origin: &'a ToolCallRef,
    ) -> BoxFuture<
        'a,
        Result<Option<lk_agents_abi::delegation::remote::RemoteRecovery>, StorageError>,
    > {
        Box::pin(async move { self.conversation()?.recover(remote_id, endpoint, origin).await })
    }

    fn default_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> BoxFuture<'a, Result<Option<String>, StorageError>> {
        Box::pin(async move { self.conversation()?.default_context(remote_id, endpoint).await })
    }

    fn remote<'a>(
        &'a self,
        command: lk_agents_abi::delegation::remote::RemoteCommand,
    ) -> BoxFuture<'a, Result<Option<InputRequestRef>, StorageError>> {
        Box::pin(async move { self.conversation()?.remote(command).await })
    }
}

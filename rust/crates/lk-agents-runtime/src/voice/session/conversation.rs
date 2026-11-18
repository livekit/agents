//! The live session lends its command sender, never its durable state or a host callback.

use super::VoiceSession;
use crate::session::CommandPort;
use lk_agents_abi::BoxFuture;
use lk_agents_abi::session::{InputRequestRef, SessionCommands, SessionInputCommand};
use lk_agents_abi::storage::StorageError;

impl VoiceSession {
    fn conversation(&self) -> Result<CommandPort, StorageError> {
        self.running(|running| Ok(running.conversation.clone()))
            .map_err(|error| StorageError::Invalid(error.to_string()))
    }
}

impl SessionCommands for VoiceSession {
    fn ended(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        let owner = self.conversation();
        Box::pin(async move {
            if let Ok(owner) = owner {
                owner.ended().await;
            }
        })
    }

    fn routing_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        proposed: serde_json::Value,
        selected_task: Option<&'a str>,
        offered: &'a [lk_agents_abi::session::InputCandidate],
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::delegation::routing::CallRoutingRequest,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            self.conversation()?
                .routing_context(remote_id, endpoint, proposed, selected_task, offered)
                .await
        })
    }

    fn admit_call<'a>(
        &'a self,
        request: &'a lk_agents_abi::delegation::routing::CallRoutingRequest,
        verdict: Option<
            Result<
                lk_agents_abi::delegation::routing::CallRoute,
                lk_agents_abi::errors::ProviderError,
            >,
        >,
        usage: lk_agents_abi::usage::Usage,
        candidate: std::sync::Weak<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::a2a::remote::RemoteCallAdmission,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move {
            self.conversation()?.admit_call(request, verdict, usage, candidate).await
        })
    }

    fn routing_model(&self) -> Option<std::sync::Arc<dyn lk_agents_abi::Llm>> {
        self.conversation().ok().and_then(|port| port.routing_model())
    }

    fn register_remote_driver(
        &self,
        remote_id: &str,
        endpoint: &str,
        request_id: &str,
        driver: std::sync::Weak<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>,
    ) {
        // Registering a weak handle changes no saved state; preparation will refuse an ended owner.
        if let Ok(port) = self.conversation() {
            port.register_remote_driver(remote_id, endpoint, request_id, driver);
        }
    }

    fn remote_driver<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task: &'a lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            Option<std::sync::Arc<dyn lk_agents_abi::a2a::remote::RemoteTaskDriver>>,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move { self.conversation()?.remote_driver(remote_id, endpoint, task).await })
    }

    fn cancel_remote<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: std::sync::Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
        retry_after: Option<&'a str>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::session::control::RemoteCancellationAdmission,
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            let commands = self.conversation().map_err(|error| {
                lk_agents_abi::a2a::client::AgentClientError::InvalidInput(error.to_string())
            })?;
            commands.cancel_remote(target, scope, candidate, retry_after).await
        })
    }

    fn cancel_task<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        task_id: &'a str,
        scope: Option<&'a lk_agents_abi::delegation::remote::RemoteControlScope>,
        candidate: std::sync::Arc<dyn lk_agents_abi::session::control::RemoteCancellation>,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (
                lk_agents_abi::session::control::RemoteControlRef,
                lk_agents_abi::session::control::RemoteCancellationAdmission,
            ),
            lk_agents_abi::a2a::client::AgentClientError,
        >,
    > {
        Box::pin(async move {
            let commands = self.conversation().map_err(|error| {
                lk_agents_abi::a2a::client::AgentClientError::InvalidInput(error.to_string())
            })?;
            commands.cancel_task(remote_id, endpoint, task_id, scope, candidate).await
        })
    }

    fn cancel_outcome<'a>(
        &'a self,
        target: &'a lk_agents_abi::session::control::RemoteControlRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            lk_agents_abi::delegation::remote::RemoteCancelOutcome,
            lk_agents_abi::storage::StorageError,
        >,
    > {
        Box::pin(async move { self.conversation()?.cancel_outcome(target).await })
    }

    fn admit_input(
        &self,
        request: lk_agents_abi::session::InputRoutingRequest,
        verdict: Option<
            Result<lk_agents_abi::session::InputRoute, lk_agents_abi::errors::ProviderError>,
        >,
    ) -> BoxFuture<'_, Result<lk_agents_abi::session::InputAdmission, StorageError>> {
        Box::pin(async move { self.conversation()?.admit_input(request, verdict).await })
    }

    fn checkpoint(&self) -> BoxFuture<'_, Result<serde_json::Value, StorageError>> {
        Box::pin(async move { self.conversation()?.checkpoint().await })
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
        target: &'a lk_agents_abi::session::SessionRecovery,
    ) -> BoxFuture<
        'a,
        Result<Option<lk_agents_abi::delegation::remote::RemoteRecovery>, StorageError>,
    > {
        Box::pin(async move { self.conversation()?.recover(remote_id, endpoint, target).await })
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

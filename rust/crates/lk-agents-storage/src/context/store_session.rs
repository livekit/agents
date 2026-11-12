//! A standalone bound TextSession owns the same fenced run as a managed transport turn.

use super::*;
use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;

pub(super) struct OwnedSession {
    thread: Thread,
    run: crate::Run,
}

impl ContextStore {
    /// Managed A2A/voice executions already own their run. A standalone durable binding
    /// acquires here, before tools or model work, and keeps ownership through resource cleanup.
    pub async fn begin_session(&self) -> Result<(), StorageError> {
        self.activation.lock().ensure_open()?;
        if self.session_sink.lock().is_some() || self.database.is_none() {
            return Ok(());
        }
        let mut owned = self.owned_session.lock().await;
        if owned.is_some() {
            return Err(StorageError::Invalid("the store already owns a session run".into()));
        }
        if self.managed.load(Ordering::Acquire) {
            return Err(StorageError::Invalid(
                "the managed turn has no session checkpoint sink".into(),
            ));
        }
        let database = self.database.as_ref().expect("durable binding has a database");
        let operation = self.lock_managed().await;
        self.settle_lost_save(database).await?;
        let thread = database.open().await?;
        let run = match thread
            .partition(&database.partition_key)
            .open(RunOptions::new(shortuuid("session_"), ""), &database.holder)
            .await
        {
            Ok(run) => run,
            Err(error) => {
                thread.close().await;
                return Err(error);
            }
        };
        // Retain the acquired run before any further fallible operation.
        *owned = Some(OwnedSession { thread, run });
        let active = owned.as_ref().expect("session run was installed");
        self.begin_managed(
            active.run.context_scopes().to_vec(),
            active.run.state().cloned(),
            &operation,
        )
    }

    pub(super) async fn checkpoint_owned_session(
        &self,
        state: Value,
        history: Vec<lk_agents_abi::ChatItem>,
    ) -> Result<(), StorageError> {
        let mut owned = self.owned_session.lock().await;
        let active = owned.as_mut().ok_or_else(|| {
            StorageError::Invalid("durable session checkpoints require an active binding".into())
        })?;
        active.run.checkpoint_session(state, history.iter().map(encode_chat_item).collect()).await
    }

    /// Only a standalone binding is settled here. Managed transport turns settle externally.
    /// This runs on the resource owner's retained cleanup task, never on a cancellable waiter.
    pub async fn finish_session(&self, success: bool) -> Result<(), StorageError> {
        let mut owned = self.owned_session.lock().await;
        let Some(mut active) = owned.take() else { return Ok(()) };
        let operation = self.lock_managed().await;
        let mut result = if success {
            if let Some(state) = self.session_state() {
                active.run.set_state(state);
            }
            active.run.stage_scopes(self.writes());
            active.run.stage_statements(self.statements());
            active.run.commit(CommitOptions::default()).await.map(drop)
        } else {
            active.run.abandon(RunOutcome::Canceled, AbandonReason::RunDropped.as_str()).await
        };
        if matches!(result, Err(StorageError::Uncertain(_))) {
            // The run owns its resolver and fence until the outcome is known.
            if let Some(resolved) = active.run.resolve_later().await {
                result = resolved.map(drop);
            }
        }
        self.end_managed(success && result.is_ok(), operation);
        active.thread.close().await;
        result
    }
}

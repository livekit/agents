//! Control writes share the voice partition, including while no speech owns it.

use super::*;
use lk_agents_abi::session::SessionCheckpoint;
use serde_json::Value;

impl VoiceStore {
    pub(crate) async fn acquire_checkpoint(
        self: &Arc<Self>,
        local: VoiceHistory,
    ) -> Result<Arc<dyn SessionCheckpoint>, StorageError> {
        let mut state = self.state.lock().await;
        let result: Result<Arc<dyn SessionCheckpoint>, StorageError> = async {
            let history = self
                .begin_locked(&mut state, None, local, crate::thread::run::PartitionKind::Context)
                .await?;
            let active = state.active.as_mut().expect("acquisition installed the run");
            active.controls += 1;
            let run_id = active.run.id().to_owned();
            let background = Background::new(
                Arc::clone(&self.tasks),
                Arc::clone(&self.background.spawner),
                Arc::clone(&self.background.clock),
            );
            let checkpoint: Arc<dyn SessionCheckpoint> =
                Arc::new(crate::context::checkpoint::Checkpoint::voice(
                    Arc::clone(self),
                    run_id,
                    history,
                    &background,
                ));
            Ok(checkpoint)
        }
        .await;
        let diagnostics = std::mem::take(&mut state.diagnostics);
        drop(state);
        self.log_history(diagnostics);
        result
    }

    pub(crate) async fn save_checkpoint(
        &self,
        run_id: &str,
        mut history: VoiceHistory,
    ) -> Result<Value, StorageError> {
        let mut state = self.state.lock().await;
        let result = async {
            let active = state
                .active
                .as_ref()
                .filter(|active| active.run.id() == run_id && active.controls > 0)
                .ok_or_else(|| {
                    StorageError::Cancelled(
                        "the acquired voice checkpoint ended before its write".into(),
                    )
                })?;
            let stored = active
                .run
                .history()
                .iter()
                .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let current = self.activation.session_state();
            let routing = current.as_ref().ok_or_else(|| {
                StorageError::Invalid("the voice conversation has no state".into())
            })?;
            let agent = crate::session_state::agent_name(routing)?.ok_or_else(|| {
                StorageError::Invalid("voice checkpoint has no current agent".into())
            })?;
            // Keep the acquired snapshot's immutable baseline while including newly admitted
            // canonical words. The latest view is not permission to overwrite an unread row.
            for item in self
                .activation
                .bound_history()?
                .unwrap_or_default()
                .into_iter()
                .filter(conversational)
            {
                match history.items.iter_mut().find(|held| held.id() == item.id()) {
                    Some(held) => *held = item,
                    None => history.items.push(item),
                }
            }
            let merged: Vec<_> = state
                .reconcile(stored, history, &self.partition_key)?
                .into_iter()
                .filter(conversational)
                .collect();
            let routing = crate::session_state::voice_value(agent, merged.clone(), Some(routing))?;
            let rows: Vec<_> = merged.iter().map(encode_chat_item).collect();
            state
                .active
                .as_mut()
                .expect("the reconciled checkpoint retains its run")
                .run
                .checkpoint_session(routing.clone(), rows.clone())
                .await?;
            state.wrote(&rows)?;
            state.version =
                state.active.as_ref().expect("the checkpoint retains its run").run.version();
            self.activation.checkpointed_session(current, routing.clone())?;
            Ok(routing)
        }
        .await;
        let diagnostics = std::mem::take(&mut state.diagnostics);
        drop(state);
        self.log_history(diagnostics);
        result
    }

    pub(crate) async fn release_checkpoint(&self, run_id: &str) -> Result<(), StorageError> {
        let mut state = self.state.lock().await;
        let Some(active) = state.active.as_mut().filter(|active| active.run.id() == run_id) else {
            return Ok(());
        };
        active.controls =
            active.controls.checked_sub(1).expect("each guard releases its own count once");
        if active.controls > 0 || !active.speeches.is_empty() {
            return Ok(());
        }
        let mut active = state.active.take().expect("the last checkpoint owns the run");
        let checkpoint_only = active.operation.is_some();
        let operation = match active.operation.take() {
            Some(operation) => operation,
            None => self.activation.lock_managed().await,
        };
        if !checkpoint_only {
            // A speech finished while this checkpoint pinned its run. Its explicit application
            // saves still commit at the end of that managed execution.
            active.run.stage_scopes(self.activation.writes());
            active.run.stage_statements(self.activation.statements());
        }
        let saved = active.run.state().cloned();
        let result = active.run.commit(CommitOptions::default()).await;
        match &result {
            Ok(committed) => state.version = committed.version,
            Err(error) => self.release_failed_commit(&active.run, error).await,
        }
        self.released();
        active.thread.close().await;
        if checkpoint_only {
            let saved = if result.is_ok() { saved } else { self.activation.committed_session() };
            self.activation.end_checkpoint(saved, operation);
        } else {
            self.activation.end_managed(result.is_ok(), saved, operation);
        }
        result.map(drop)
    }
}

//! Framework model views share the transport ledger without owning its other handlers' rows.

use std::collections::BTreeMap;

use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::storage::voice::VoiceHistory;
use serde_json::Value;

use super::Run;
use crate::StorageError;

#[derive(Clone)]
pub struct SessionCheckpointData {
    pub state: Value,
    pub history: VoiceHistory,
}

impl Run {
    /// Prepare a narrow view using its own last acknowledged baseline. Omitted transport
    /// rows survive; another writer's changed canonical row is never claimed as our write.
    pub fn reconcile_session_view(
        &self,
        state: Value,
        history: VoiceHistory,
    ) -> Result<SessionCheckpointData, StorageError> {
        let stored = self
            .history()
            .iter()
            .map(|row| decode_chat_item(row.payload.clone()).map_err(StorageError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let revision = history.revision;
        let merged =
            crate::voice::reconcile_history(&stored, history, &BTreeMap::new()).map_err(|_| {
                StorageError::VersionConflict {
                    partition_key: self.partition_key.name().to_owned(),
                    expected: revision.0,
                }
            })?;
        Ok(SessionCheckpointData {
            state,
            history: VoiceHistory {
                stored: merged.items.clone(),
                items: merged.items,
                revision: (self.version(), revision.1.saturating_add(1)),
                ..Default::default()
            },
        })
    }

    /// A source may read its unchanged old snapshot through the current native owner. An
    /// explicit replacement cannot change the live machine behind that owner's command port.
    pub fn reconcile_session_edit(
        &self,
        proposed: Value,
        observed: Option<&Value>,
        history: VoiceHistory,
    ) -> Result<SessionCheckpointData, StorageError> {
        let current = self.state();
        if current != Some(&proposed) && observed != Some(&proposed) {
            return Err(StorageError::VersionConflict {
                partition_key: self.partition_key.name().to_owned(),
                expected: history.revision.0,
            });
        }
        let state = current.cloned().ok_or_else(|| {
            StorageError::Invalid("the native conversation has no saved checkpoint".into())
        })?;
        self.reconcile_session_view(state, history)
    }
}

impl SessionCheckpointData {
    pub async fn commit(self, run: &mut Run) -> Result<Self, StorageError> {
        let rows = self.history.items.iter().map(encode_chat_item).collect();
        run.checkpoint_session(self.state.clone(), rows).await?;
        Ok(self)
    }
}

//! Runtime-supplied field stores: snapshot staging in a turn, fenced direct saves outside one.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::storage::context::{
    ContextActivation as HostContextActivation, ContextField, ContextScope, ContextScopeWrite,
    ContextValues,
};
use lk_agents_storage::StorageError;
use lk_agents_storage::context::ContextActivation as Activation;
use lk_agents_storage::run::RunOutcome;
use serde_json::Value;

use super::{CommitOptions, RunOptions, Storage, shortuuid};

pub struct ContextActivation {
    activation: Mutex<Activation>,
    direct: Option<DirectStore>,
    operation: tokio::sync::Mutex<()>,
}

struct DirectStore {
    storage: Arc<dyn Storage>,
    thread_id: String,
    lane: String,
}

impl DirectStore {
    async fn snapshot(&self) -> Result<(Vec<ContextScope>, Option<Value>), StorageError> {
        let thread = self.storage.open(&self.thread_id, "context").await?;
        let result = async {
            let scopes = thread.context_scopes(&self.lane).await?;
            let routing = thread.lane(&self.lane).info().await?.and_then(|lane| lane.state);
            Ok((scopes, routing))
        }
        .await;
        thread.close().await;
        result
    }

    async fn commit(&self, writes: Vec<ContextScopeWrite>) -> Result<(), StorageError> {
        let thread = self.storage.open(&self.thread_id, "context").await?;
        let result = async {
            let mut options = RunOptions::new(shortuuid("context_"), "");
            options.channel = Some("context".into());
            let mut run = thread.lane(&self.lane).open(options).await?;
            // The lease serializes direct and conversational saves for this scope's lane.
            // Check before any writes so a stale direct save has a typed refusal.
            for write in &writes {
                let revision = run
                    .context_scopes
                    .iter()
                    .find(|scope| scope.scope_id == write.scope.scope_id)
                    .map_or(0, |scope| scope.revision);
                if revision != write.expected_revision {
                    run.abandon(RunOutcome::Failed, "context-version-conflict").await?;
                    return Err(StorageError::VersionConflict {
                        lane: self.lane.clone(),
                        expected: write.expected_revision,
                    });
                }
            }
            run.scope_writes = writes;
            run.commit(CommitOptions::default()).await?;
            Ok(())
        }
        .await;
        thread.close().await;
        result
    }
}

impl ContextActivation {
    pub async fn bound(
        storage: Arc<dyn Storage>,
        thread_id: String,
        lane: String,
    ) -> Result<Self, StorageError> {
        let direct = DirectStore { storage, thread_id, lane };
        let (scopes, routing) = direct.snapshot().await?;
        Ok(Self {
            activation: Mutex::new(Activation::new(scopes, routing, false)?),
            direct: Some(direct),
            operation: tokio::sync::Mutex::new(()),
        })
    }
}

#[async_trait]
impl HostContextActivation for ContextActivation {
    fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            activation: Mutex::new(Activation::new(scopes, routing_state, read_only)?),
            direct: None,
            operation: tokio::sync::Mutex::new(()),
        })
    }

    async fn load(
        &self,
        scope_id: String,
        identity: String,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError> {
        let _operation = self.operation.lock().await;
        if let Some(direct) = &self.direct {
            let (scopes, routing) = direct.snapshot().await?;
            self.activation.lock().expect("context activation").rebase(scopes, routing)?;
        }
        self.activation.lock().expect("context activation").load(&scope_id, &identity, &fields)
    }

    async fn save(
        &self,
        scope_id: String,
        identity: String,
        revision: i64,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError> {
        let _operation = self.operation.lock().await;
        let values = self
            .activation
            .lock()
            .expect("context activation")
            .save(&scope_id, &identity, revision, &fields)?;
        if let Some(direct) = &self.direct {
            let writes = self.activation.lock().expect("context activation").writes();
            if let Err(error) = direct.commit(writes).await {
                if !matches!(error, StorageError::Uncertain(_))
                    && let Ok((scopes, routing)) = direct.snapshot().await
                {
                    *self.activation.lock().expect("context activation") =
                        Activation::new(scopes, routing, false)?;
                }
                return Err(error);
            }
            self.activation.lock().expect("context activation").acknowledge();
        }
        Ok(values)
    }

    fn writes(&self) -> Vec<ContextScopeWrite> {
        self.activation.lock().expect("context activation").writes()
    }

    fn acknowledge(&self) {
        self.activation.lock().expect("context activation").acknowledge();
    }

    fn close(&self) {
        self.activation.lock().expect("context activation").close();
    }
}

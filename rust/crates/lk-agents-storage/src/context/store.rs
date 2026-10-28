//! A lane's context field store as the host uses it: snapshot staging in a turn, fenced direct
//! saves outside one.

use std::sync::{Arc, Mutex};

use lk_agents_abi::storage::context::{
    ContextActivation as HostContextActivation, ContextField, ContextScope, ContextScopeWrite,
    ContextValues,
};
use serde_json::Value;

use super::ContextActivation as Activation;
use crate::run::RunOutcome;
use crate::{CommitOptions, RunOptions, Storage, StorageError, shortuuid};

#[derive(Clone)]
pub struct ContextStore {
    activation: Arc<Mutex<Activation>>,
    direct: Option<DirectStore>,
    operation: Arc<tokio::sync::Mutex<()>>,
    managed: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Clone)]
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

impl ContextStore {
    pub(crate) async fn lock_managed(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.operation.clone().lock_owned().await
    }

    pub(crate) fn begin_managed(
        &self,
        scopes: Vec<ContextScope>,
        routing: Option<Value>,
        _operation: &tokio::sync::OwnedMutexGuard<()>,
    ) -> Result<(), StorageError> {
        self.activation.lock().expect("context activation").rebase(scopes, routing)?;
        self.managed.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    pub(crate) fn end_managed(
        &self,
        committed: bool,
        _operation: tokio::sync::OwnedMutexGuard<()>,
    ) {
        let mut activation = self.activation.lock().expect("context activation");
        if committed {
            activation.acknowledge();
        } else {
            activation.abandon();
        }
        self.managed.store(false, std::sync::atomic::Ordering::Release);
    }

    pub async fn bound(
        storage: Arc<dyn Storage>,
        thread_id: String,
        lane: String,
    ) -> Result<Self, StorageError> {
        let direct = DirectStore { storage, thread_id, lane };
        let (scopes, routing) = direct.snapshot().await?;
        Ok(Self {
            activation: Arc::new(Mutex::new(Activation::new(scopes, routing, false)?)),
            direct: Some(direct),
            operation: Arc::new(tokio::sync::Mutex::new(())),
            managed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

impl HostContextActivation for ContextStore {
    fn new(
        scopes: Vec<ContextScope>,
        routing_state: Option<Value>,
        read_only: bool,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            activation: Arc::new(Mutex::new(Activation::new(scopes, routing_state, read_only)?)),
            direct: None,
            operation: Arc::new(tokio::sync::Mutex::new(())),
            managed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    async fn load(
        &self,
        scope_id: String,
        identity: String,
        fields: Vec<ContextField>,
    ) -> Result<ContextValues, StorageError> {
        let _operation = self.operation.lock().await;
        if let Some(direct) = &self.direct
            && !self.managed.load(std::sync::atomic::Ordering::Acquire)
        {
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
        if let Some(direct) = &self.direct
            && !self.managed.load(std::sync::atomic::Ordering::Acquire)
        {
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

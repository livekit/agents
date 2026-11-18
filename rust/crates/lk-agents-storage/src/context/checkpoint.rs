//! A checkpoint owns its partition until the admitted state has been written or abandoned.

use std::sync::Arc;

use lk_agents_abi::session::SessionCheckpoint;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::voice::VoiceHistory;
use lk_agents_abi::{BoxFuture, Closed};
use lk_agents_net::owned::{Background, Owned};
use lk_agents_net::signal;
use parking_lot::Mutex;
use serde_json::Value;

use super::store::ContextStore;
use crate::StorageError;
use crate::voice::VoiceStore;

pub(crate) struct Checkpoint(Mutex<Scope>);

enum Scope {
    Memory(ContextStore),
    Voice {
        owner: Owned<VoiceScope>,
        released: signal::Cell<Option<Result<(), StorageError>>>,
        history: VoiceHistory,
        background: Background,
    },
    Taken,
}

struct VoiceScope {
    store: Arc<VoiceStore>,
    run_id: String,
    done: signal::Producer<Option<Result<(), StorageError>>>,
}

impl Checkpoint {
    pub(crate) fn memory(store: ContextStore) -> Self {
        Self(Mutex::new(Scope::Memory(store)))
    }

    pub(crate) fn voice(
        store: Arc<VoiceStore>,
        run_id: String,
        history: VoiceHistory,
        background: &Background,
    ) -> Self {
        let (done, released) = signal::once();
        let owner = Owned::new(
            background,
            VoiceScope { store, run_id, done },
            Box::new(|scope, _deadline| {
                Box::pin(async move {
                    // The fenced storage operation has its own bound; an outer timeout must not
                    // cut its release in half.
                    let result = scope.store.release_checkpoint(&scope.run_id).await;
                    crate::telemetry::cleanup_failed(
                        "voice",
                        "releasing a conversation checkpoint",
                        result.clone(),
                    );
                    let closed = if result.is_ok() { Closed::Clean } else { Closed::Failed };
                    scope.done.resolve(result);
                    closed
                })
            }),
        );
        Self(Mutex::new(Scope::Voice { owner, released, history, background: background.clone() }))
    }
}

impl SessionCheckpoint for Checkpoint {
    fn history(&self) -> Option<VoiceHistory> {
        match &*self.0.lock() {
            Scope::Voice { history, .. } => Some(history.clone()),
            Scope::Memory(_) | Scope::Taken => None,
        }
    }

    fn commit(&self) -> BoxFuture<'_, Result<Value, StorageError>> {
        let scope = std::mem::replace(&mut *self.0.lock(), Scope::Taken);
        let background = match &scope {
            Scope::Voice { background, .. } => Some(background.clone()),
            Scope::Memory(_) | Scope::Taken => None,
        };
        let write = async move {
            match scope {
                Scope::Memory(store) => {
                    store.flush_session(store.bound_history()?.unwrap_or_default()).await?;
                    store.session_state().ok_or_else(|| {
                        StorageError::Invalid("the conversation has no staged state".into())
                    })
                }
                Scope::Voice { owner, released, history, .. } => {
                    let written = owner.store.save_checkpoint(&owner.run_id, history).await;
                    let closed = owner.close().await;
                    let released = released.wait().await.map_err(|_| {
                        StorageError::Unavailable("the checkpoint release task ended".into())
                    })?;
                    let saved = written?;
                    released?;
                    match closed {
                        Closed::Clean => Ok(saved),
                        Closed::Failed | Closed::TimedOut => Err(StorageError::Unavailable(
                            "the conversation checkpoint did not close cleanly".into(),
                        )),
                    }
                }
                Scope::Taken => {
                    Err(StorageError::Invalid("the checkpoint was already used".into()))
                }
            }
        };
        let Some(background) = background else { return Box::pin(write) };
        // Once admitted, storage owns the bounded write and its release. Dropping a caller's
        // waiter must not interrupt a committed write before its exact acknowledgement.
        let (done, result) = signal::once();
        let admitted = background.tasks.spawn_settling(
            &*background.spawner,
            "voice conversation checkpoint write",
            done,
            write,
        );
        Box::pin(async move {
            if !admitted {
                return Err(StorageError::Cancelled("the voice checkpoint owner ended".into()));
            }
            result.wait().await.map_err(|_| {
                StorageError::Unavailable(
                    "the voice checkpoint writer ended before its receipt".into(),
                )
            })?
        })
    }
}

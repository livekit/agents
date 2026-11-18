use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as HostBinding};
use lk_agents_abi::text_session::{SessionSnapshot, SessionStateSink};
use lk_agents_abi::{BoxFuture, ChatItem};

use crate::StorageError;
use crate::context::store::ContextStore;
use crate::session_state::SessionState;

pub struct SessionBinding {
    store: ContextStore,
    view: SessionState,
    closed: AtomicBool,
}

impl SessionBinding {
    /// The same native conversation owner without a database or application save policy.
    pub fn memory() -> Self {
        let store =
            ContextStore::new(Vec::new(), None, false).expect("an empty writable context is valid");
        let view = SessionState::new(None).expect("an empty session state is valid");
        Self::new(store, view).expect("a fresh memory context has no existing binding")
    }

    fn ensure_open(&self) -> Result<(), StorageError> {
        if self.closed.load(Ordering::Acquire) || self.store.closed() {
            return Err(StorageError::Invalid("the session binding has ended".into()));
        }
        Ok(())
    }
}

impl HostBinding for SessionBinding {
    type Activation = ContextStore;
    type View = SessionState;
    fn new(store: ContextStore, view: SessionState) -> Result<Self, StorageError> {
        let current = store
            .session_state()
            .unwrap_or_else(|| serde_json::json!({"version":1,"kind":"empty"}));
        // Validate the activation and envelope before claiming the binding.
        let checked = SessionState::from_value(current.clone())?;
        store.stage_session(current)?;
        {
            let mut active = store.session_view.lock();
            if active.is_some() {
                return Err(StorageError::Invalid(
                    "this store already has an active session binding".into(),
                ));
            }
            *active = Some(view.clone());
        }
        view.assign(&checked);
        Ok(Self { store, view, closed: AtomicBool::new(false) })
    }
    fn update(&self, state: String) -> Result<(), StorageError> {
        self.ensure_open()?;
        let checked = SessionState::new(Some(state))?;
        let value = serde_json::from_str(&checked._to_json())
            .map_err(|error| StorageError::Invalid(error.to_string()))?;
        self.store.stage_session(value)
    }
    async fn flush(&self, history: Option<Vec<ChatItem>>) -> Result<(), StorageError> {
        self.ensure_open()?;
        let history = match history {
            Some(history) => history,
            None => self.view.root_history()?,
        };
        self.store.flush_session(history).await
    }
    fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
    fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.view.unbind_commands();
            let mut active = self.store.session_view.lock();
            if active.as_ref().is_some_and(|view| view.same_view(&self.view)) {
                *active = None;
            }
        }
    }
}

impl SessionStateSink for SessionBinding {
    fn bind_commands(
        &self,
        commands: std::sync::Weak<dyn lk_agents_abi::session::SessionCommands>,
    ) -> Result<(), StorageError> {
        self.ensure_open()?;
        self.view.bind_commands(commands)
    }

    fn checkpoint_voice<'a>(
        &'a self,
        history: lk_agents_abi::storage::voice::VoiceHistory,
    ) -> BoxFuture<
        'a,
        Result<std::sync::Arc<dyn lk_agents_abi::session::SessionCheckpoint>, StorageError>,
    > {
        Box::pin(async move {
            self.ensure_open()?;
            self.store.voice_checkpoint(history).await
        })
    }

    fn stage_voice(&self, state: SessionSnapshot) -> Result<(), StorageError> {
        self.ensure_open()?;
        let mut next = crate::session_state::snapshot_value(state)?;
        if let Some(current) = self.store.session_state()
            && let Some(pending) = current.get("pending_user_turns")
        {
            next["pending_user_turns"] = pending.clone();
        }
        self.store.stage_session(next)
    }

    fn restore(&self) -> Result<Option<SessionSnapshot>, StorageError> {
        self.ensure_open()?;
        let state = self
            .store
            .session_state()
            .unwrap_or_else(|| serde_json::json!({"version": 1, "kind": "empty"}));
        let checked = SessionState::from_value(state)?;
        let snapshot = checked._snapshot_data()?;
        self.view.assign(&checked);
        Ok(snapshot)
    }

    fn publish<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        state: SessionSnapshot,
    ) -> BoxFuture<'a, Result<serde_json::Value, StorageError>> {
        Box::pin(async move {
            self.ensure_open()?;
            self.store.stage_session(crate::session_state::snapshot_value(state)?)?;
            self.flush(None).await?;
            self.store
                .committed_session()
                .ok_or_else(|| StorageError::Invalid("the conversation has no checkpoint".into()))
        })
    }
}

impl Drop for SessionBinding {
    fn drop(&mut self) {
        self.close();
    }
}

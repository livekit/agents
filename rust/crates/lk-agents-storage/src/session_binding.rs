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
    fn update_voice(&self, agent: String, history: Vec<ChatItem>) -> Result<(), StorageError> {
        self.ensure_open()?;
        self.store.stage_session(crate::session_state::voice_value(
            agent,
            history,
            self.store.session_state().as_ref(),
        )?)
    }
    fn validate_voice(&self, agent: String) -> Result<(), StorageError> {
        self.ensure_open()?;
        if let Some(state) = self.view._snapshot_data()?
            && (state.active_frame_id != state.root_frame_id
                || !state.ancestors.is_empty()
                || state.frames.len() > 1
                || state.frames.values().any(|agent| !agent.pending_calls.is_empty()))
        {
            return Err(StorageError::Invalid(
                "saved local agent work must continue through TextSession or bound Agent.run"
                    .into(),
            ));
        }
        if let Some(saved) = self.view.root_agent()?.map(|agent| agent.agent_name)
            && saved != agent
        {
            return Err(StorageError::Invalid(format!(
                "saved voice agent {saved:?} does not match configured agent {agent:?}"
            )));
        }
        Ok(())
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
            let mut active = self.store.session_view.lock();
            if active.as_ref().is_some_and(|view| view.same_view(&self.view)) {
                *active = None;
            }
        }
    }
}

impl SessionStateSink for SessionBinding {
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
    ) -> BoxFuture<'a, Result<(), StorageError>> {
        Box::pin(async move {
            self.ensure_open()?;
            self.store.stage_session(crate::session_state::snapshot_value(state)?)?;
            self.flush(None).await
        })
    }
}

impl Drop for SessionBinding {
    fn drop(&mut self) {
        self.close();
    }
}

//! Native session views. A clone shares the live cell; snapshot serialization detaches it.
use crate::StorageError;
use lk_agents_abi::storage::session::{PersistedSessionState, SessionKind};
use lk_agents_abi::text_session::{FrameId, SessionFrame, TextSessionState};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Serialize, Deserialize)]
struct Envelope {
    version: u32,
    #[serde(flatten)]
    state: State,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum State {
    Empty,
    Text { state: TextSessionState },
    Voice { agent: String, history_ids: Vec<String> },
}

#[derive(Clone)]
pub struct SessionState(Arc<Mutex<Envelope>>);

impl SessionState {
    pub(crate) fn from_value(value: serde_json::Value) -> Result<Self, StorageError> {
        let envelope: Envelope = serde_json::from_value(value).map_err(invalid)?;
        if envelope.version != 1 {
            return Err(invalid("unsupported session state version"));
        }
        Ok(Self(Arc::new(Mutex::new(envelope))))
    }

    pub(crate) fn assign(&self, other: &Self) {
        let state = other.0.lock().clone();
        *self.0.lock() = state;
    }

    pub(crate) fn root_history(&self) -> Result<Vec<lk_agents_abi::ChatItem>, StorageError> {
        match &self.0.lock().state {
            State::Text { state } => state
                .frames
                .get(&state.root_frame_id)
                .map(|frame| frame.history.clone())
                .ok_or_else(|| invalid("session state has no root frame")),
            _ => Ok(Vec::new()),
        }
    }

    pub(crate) fn same_view(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

pub(crate) fn text_value(state: TextSessionState) -> Result<serde_json::Value, StorageError> {
    serde_json::to_value(Envelope { version: 1, state: State::Text { state } }).map_err(invalid)
}

pub(crate) fn voice_value(
    agent: String,
    history_ids: Vec<String>,
) -> Result<serde_json::Value, StorageError> {
    serde_json::to_value(Envelope { version: 1, state: State::Voice { agent, history_ids } })
        .map_err(invalid)
}

fn invalid(error: impl std::fmt::Display) -> StorageError {
    StorageError::Invalid(error.to_string())
}

fn decode(data: &str) -> Result<Envelope, StorageError> {
    let envelope: Envelope = serde_json::from_str(data).map_err(invalid)?;
    if envelope.version != 1 {
        return Err(invalid("unsupported session state version"));
    }
    Ok(envelope)
}

fn encode(state: State) -> Result<String, StorageError> {
    serde_json::to_string(&Envelope { version: 1, state }).map_err(invalid)
}

impl PersistedSessionState for SessionState {
    fn new(data: Option<String>) -> Result<Self, StorageError> {
        let state = match data {
            Some(data) => decode(&data)?,
            None => Envelope { version: 1, state: State::Empty },
        };
        Ok(Self(Arc::new(Mutex::new(state))))
    }
    fn kind(&self) -> SessionKind {
        match self.0.lock().state {
            State::Empty => SessionKind::Empty,
            State::Text { .. } => SessionKind::Text,
            State::Voice { .. } => SessionKind::Voice,
        }
    }
    fn active_agent(&self) -> Option<String> {
        match &self.0.lock().state {
            State::Text { state } => {
                state.frames.get(&state.active_frame_id).map(|frame| frame.agent_name.clone())
            }
            State::Voice { agent, .. } => Some(agent.clone()),
            State::Empty => None,
        }
    }
    fn session_id(&self) -> Option<String> {
        match &self.0.lock().state {
            State::Text { state } => Some(state.session_id.clone()),
            _ => None,
        }
    }
    fn execution_id(&self) -> Option<String> {
        match &self.0.lock().state {
            State::Text { state } => Some(state.execution_id.clone()),
            _ => None,
        }
    }
    fn root_frame_id(&self) -> Result<FrameId, StorageError> {
        match &self.0.lock().state {
            State::Text { state } => Ok(state.root_frame_id.clone()),
            _ => Err(invalid("only a text session has a root frame")),
        }
    }
    fn active_frame_id(&self) -> Result<FrameId, StorageError> {
        match &self.0.lock().state {
            State::Text { state } => Ok(state.active_frame_id.clone()),
            _ => Err(invalid("only a text session has an active frame")),
        }
    }
    fn ancestors(&self) -> Vec<FrameId> {
        match &self.0.lock().state {
            State::Text { state } => state.ancestors.clone(),
            _ => Vec::new(),
        }
    }
    fn frames(&self) -> BTreeMap<FrameId, SessionFrame> {
        match &self.0.lock().state {
            State::Text { state } => state.frames.clone(),
            _ => BTreeMap::new(),
        }
    }
    fn snapshot(&self) -> String {
        self._to_json()
    }
    fn _to_json(&self) -> String {
        serde_json::to_string(&*self.0.lock()).expect("validated session state serializes")
    }
    fn _text_state(&self) -> Result<Option<TextSessionState>, StorageError> {
        match &self.0.lock().state {
            State::Empty => Ok(None),
            State::Text { state } => Ok(Some(state.clone())),
            State::Voice { .. } => Err(invalid("a voice snapshot cannot resume a text session")),
        }
    }
    fn _update(&self, data: String) -> Result<(), StorageError> {
        *self.0.lock() = decode(&data)?;
        Ok(())
    }
    fn _from_json(data: String) -> Result<String, StorageError> {
        let state = decode(&data)?;
        serde_json::to_string(&state).map_err(invalid)
    }
    fn _from_text(state: TextSessionState) -> Result<String, StorageError> {
        encode(State::Text { state })
    }
    fn _from_voice(agent: String, history_ids: Vec<String>) -> Result<String, StorageError> {
        encode(State::Voice { agent, history_ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cloned_views_follow_updates_but_snapshots_detach() {
        let view = SessionState::new(None).unwrap();
        let shared = view.clone();
        let snapshot = SessionState::new(Some(view.snapshot())).unwrap();
        view._update(SessionState::_from_voice("desk".into(), vec![]).unwrap()).unwrap();
        assert_eq!(shared.active_agent().as_deref(), Some("desk"));
        assert_eq!(snapshot.kind(), SessionKind::Empty);
        assert!(view._text_state().is_err());
        assert!(view._update(r#"{"version":2,"kind":"empty"}"#.into()).is_err());
        assert_eq!(view.kind(), SessionKind::Voice);
    }
}

use super::*;

impl PersistedSessionState for SessionState {
    type RemoteAgent = super::remote::RemoteAgent;

    fn new(data: Option<String>) -> Result<Self, StorageError> {
        let value = data
            .map(|data| serde_json::from_str(&data).map_err(invalid))
            .transpose()?
            .unwrap_or_else(|| serde_json::json!({"version":1,"kind":"empty"}));
        Self::from_value(value)
    }
    fn kind(&self) -> SessionKind {
        if matches!(self.0.lock().state, State::Empty) {
            SessionKind::Empty
        } else {
            SessionKind::Conversation
        }
    }
    fn root_agent(&self) -> Result<Option<lk_agents_abi::session::AgentState>, StorageError> {
        self.agent_view(false)
    }
    fn current_agent(&self) -> Result<Option<lk_agents_abi::session::AgentState>, StorageError> {
        self.agent_view(true)
    }
    fn agents(&self) -> Result<BTreeMap<String, lk_agents_abi::session::AgentState>, StorageError> {
        self.agent_views()
    }
    fn remote_agents(&self) -> BTreeMap<String, Self::RemoteAgent> {
        self.remote_views()
    }
    fn chat_history(&self) -> Vec<ChatItem> {
        match &self.0.lock().state {
            State::Conversation { state, .. } => {
                state.chat_history.iter().cloned().map(ChatItem::Message).collect()
            }
            _ => Vec::new(),
        }
    }
    fn session_id(&self) -> Option<String> {
        match &self.0.lock().state {
            State::Conversation { state, .. } => Some(state.session_id.clone()),
            _ => None,
        }
    }
    fn snapshot(&self) -> String {
        self._to_json()
    }
    fn _to_json(&self) -> String {
        serde_json::to_string(&*self.0.lock()).expect("validated session state serializes")
    }
    fn _snapshot_data(&self) -> Result<Option<SessionSnapshot>, StorageError> {
        Ok(match &self.0.lock().state {
            State::Conversation { state, .. } => Some(state.as_ref().clone()),
            _ => None,
        })
    }
    fn _update(&self, data: String) -> Result<(), StorageError> {
        let checked = Self::new(Some(data))?;
        self.assign(&checked);
        Ok(())
    }
    fn _from_json(data: String) -> Result<String, StorageError> {
        Ok(Self::new(Some(data))?._to_json())
    }
}

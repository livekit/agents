//! The remote interaction binding owns synchronization only; A2A owns every decision.

use std::sync::Mutex;

use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::a2a::delegation::{
    DelegationUpdate, InteractionId, RemoteInteraction as Policy, retained,
};
use lk_agents_abi::a2a::client::{AgentClientError, Message, TaskUpdate};
use lk_agents_abi::a2a::remote::{RemoteEvent, RemoteInteraction as Port};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ChatItem, FunctionCall};
use serde::Deserialize;
use serde_json::{Map, Value};

use super::host::message_from_wire;

pub struct RemoteInteraction {
    inner: Mutex<State>,
}

struct State {
    policy: Policy,
    call: FunctionCall,
    livekit: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextRequest {
    request: String,
}

impl Port for RemoteInteraction {
    fn new(
        endpoint: String,
        tool_id: String,
        function_call: ChatItem,
        history: Vec<ChatItem>,
    ) -> Result<Self, AgentClientError> {
        let ChatItem::FunctionCall(function_call) = function_call else {
            return Err(AgentClientError::InvalidInput(
                "a remote interaction needs its parent function call".into(),
            ));
        };
        let id = InteractionId {
            delegation_id: lk_agents_abi::utils::strings::shortuuid("dlg_"),
            endpoint: endpoint.trim_end_matches('/').into(),
            tool_id,
            parent_call_id: function_call.call_id.clone(),
        };
        let policy = retained::for_call(id, &history).map_err(protocol)?;
        Ok(Self { inner: Mutex::new(State { policy, call: function_call, livekit: false }) })
    }

    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: Vec<ChatItem>,
    ) -> Result<Message, AgentClientError> {
        let mut state = self.inner.lock().map_err(protocol)?;
        if !arguments.is_object() {
            return Err(AgentClientError::InvalidInput(
                "remote tool arguments must be an object".into(),
            ));
        }
        let text = if structured {
            String::new()
        } else {
            serde_json::from_value::<TextRequest>(arguments.clone())
                .map_err(|error| AgentClientError::InvalidInput(error.to_string()))?
                .request
        };
        let original_call = state.policy.snapshot().id.parent_call_id;
        let caller_history = offers_extension
            .then(|| {
                CallerHistory::from_chat_items(
                    &history,
                    Some(&original_call),
                    &HistorySharing::default(),
                )
            })
            .transpose()
            .map_err(protocol)?;
        let mut message = state
            .policy
            .message(
                text,
                lk_agents_abi::utils::strings::shortuuid("msg_"),
                None,
                "",
                None,
                offers_extension,
                caller_history.as_ref(),
            )
            .map_err(protocol)?;
        if structured {
            message.parts[0] = a2a::Part::data(arguments);
        }
        state.livekit = offers_extension;
        Ok(message_from_wire(message))
    }

    fn update(&self, update: TaskUpdate) -> Result<RemoteEvent, AgentClientError> {
        let mut state = self.inner.lock().map_err(protocol)?;
        let update = retained::update_from_host(update).map_err(protocol)?;
        Ok(match state.policy.update(&update).map_err(protocol)? {
            DelegationUpdate::Pending => RemoteEvent::Pending,
            DelegationUpdate::Progress(text) => RemoteEvent::Progress { text },
            DelegationUpdate::InputRequired { request_id, question, verbatim } => {
                let verbatim_full = verbatim.as_deref() == Some(question.as_str());
                let outcome = ToolOutcome::InputRequired {
                    question: question.clone(),
                    verbatim: verbatim_full,
                    tool: Some(state.call.name.clone()),
                };
                RemoteEvent::InputRequired {
                    request_id,
                    question,
                    verbatim,
                    verbatim_full,
                    outcome,
                }
            }
            DelegationUpdate::Completed(outcome) | DelegationUpdate::Failed(outcome) => {
                RemoteEvent::Outcome { outcome }
            }
            DelegationUpdate::Cancelled => RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled },
            DelegationUpdate::Follow(task) => {
                RemoteEvent::Follow { task_id: task.id, thread_id: task.context_id }
            }
        })
    }

    fn cancelled(&self) -> Result<(), AgentClientError> {
        let mut state = self.inner.lock().map_err(protocol)?;
        state.policy.cancel();
        state.policy.cancellation_settled();
        Ok(())
    }

    fn extra(&self) -> Result<Map<String, Value>, AgentClientError> {
        let state = self.inner.lock().map_err(protocol)?;
        retained::call_extra(&state.policy, &state.call, state.livekit).map_err(protocol)
    }
}

fn protocol(error: impl std::fmt::Display) -> AgentClientError {
    AgentClientError::Protocol(error.to_string())
}

impl lk_agents_abi::a2a::remote::RemoteConfiguration for RemoteInteraction {
    fn _core_remote_token(
        endpoint: String,
        token: Option<String>,
    ) -> Result<Option<String>, AgentClientError> {
        use lk_agents_abi::a2a::auth::Credentials as CredentialsApi;

        if token.is_some() {
            return Ok(token);
        }
        let Some(project_url) = std::env::var("LIVEKIT_URL").ok().filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let Some(api_key) = std::env::var("LIVEKIT_API_KEY").ok().filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let Some(api_secret) =
            std::env::var("LIVEKIT_API_SECRET").ok().filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let Ok(mut project) = url::Url::parse(&project_url) else {
            return Ok(None);
        };
        let scheme = match project.scheme() {
            "ws" | "http" => "http",
            "wss" | "https" => "https",
            _ => return Ok(None),
        };
        if project.set_scheme(scheme).is_err()
            || !project.username().is_empty()
            || project.password().is_some()
        {
            return Ok(None);
        }
        let destination = super::checked_url(&endpoint).map_err(protocol)?;
        if project.origin() != destination.origin() {
            return Ok(None);
        }
        crate::a2a_auth::Credentials::new(api_key, api_secret)
            .project_token("delegate".into(), 3600)
            .map(Some)
            .map_err(protocol)
    }
}

//! The remote interaction binding owns synchronization only; A2A owns every decision.

use std::sync::Mutex;

use lk_agents_abi::a2a::client::{AgentClientError, Message, MessagePart, TaskUpdate};
use lk_agents_abi::a2a::remote::{RemoteEvent, RemoteInteraction as Port};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ChatItem, FunctionCall};
use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::a2a::delegation::{
    DelegationUpdate, InteractionId, RemoteInteraction as Policy, retained,
};
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
            delegation_id: crate::strings::shortuuid("dlg_"),
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
        thread_id: Option<String>,
        token: Option<String>,
    ) -> Result<Message, AgentClientError> {
        let mut state = self.inner.lock().map_err(protocol)?;
        // sharing the caller's thread needs a grant this project signed for it, verified here
        let authority = match (thread_id.as_deref(), token.as_deref()) {
            (Some(thread), Some(token)) => {
                let endpoint = state.policy.snapshot().id.endpoint;
                let project = Project::of(&endpoint)?.ok_or_else(|| {
                    AgentClientError::InvalidInput(
                        "sharing a thread needs the endpoint on the configured project".into(),
                    )
                })?;
                let assertion = super::authority::AuthorityAssertion::Grant {
                    project_url: &project.url,
                    api_key: &project.api_key,
                    api_secret: &project.api_secret,
                    token,
                };
                let now = std::time::SystemTime::now();
                Some(
                    super::authority::verify(&endpoint, thread, DELEGATE, assertion, now)
                        .map_err(protocol)?,
                )
            }
            _ => None,
        };
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
                crate::strings::shortuuid("msg_"),
                thread_id.as_deref().filter(|_| authority.is_some()),
                DELEGATE,
                authority.as_ref(),
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

    fn answer(&self, request_id: String, text: String) -> Result<Message, AgentClientError> {
        let mut state = self.inner.lock().map_err(protocol)?;
        state.policy.accepts_reply(&request_id).map_err(protocol)?;
        let livekit = state.livekit;
        let message = state
            .policy
            .message(text, crate::strings::shortuuid("msg_"), None, "", None, livekit, None)
            .map_err(protocol)?;
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

impl lk_agents_abi::a2a::client::MessageContent for RemoteInteraction {
    fn _core_a2a_message_text(message: Message) -> String {
        message
            .parts
            .iter()
            .filter_map(|part| match part {
                MessagePart::TextPart { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn _core_a2a_message_data(message: Message) -> Option<Map<String, Value>> {
        message.parts.into_iter().find_map(|part| match part {
            MessagePart::DataPart { data: Value::Object(data), .. } => Some(data),
            _ => None,
        })
    }
}

impl lk_agents_abi::a2a::remote::RemoteConfiguration for RemoteInteraction {
    fn _core_remote_token(
        endpoint: String,
        token: Option<String>,
        thread_id: Option<String>,
    ) -> Result<Option<String>, AgentClientError> {
        use lk_agents_abi::a2a::auth::{AuthGrant, Credentials as CredentialsApi};

        if token.is_some() {
            return Ok(token);
        }
        let Some(project) = Project::of(&endpoint)? else { return Ok(None) };
        let credentials = crate::a2a_auth::Credentials::new(project.api_key, project.api_secret);
        match thread_id.filter(|id| !id.is_empty()) {
            // the caller's thread, shared with the endpoint (docs/internals/storage.md "Two
            // brains, one thread"): an endpoint call alone grants no thread access
            Some(thread_id) => credentials.thread_grant_token(
                AuthGrant {
                    thread_ids: vec![thread_id],
                    read: true,
                    write: true,
                    ..AuthGrant::default()
                },
                DELEGATE.into(),
                3600,
            ),
            // no thread to share: enough to read the card
            None => credentials.project_token(DELEGATE.into(), 3600),
        }
        .map(Some)
        .map_err(protocol)
    }
}

/// The identity of the tokens a remote tool mints for itself.
const DELEGATE: &str = "delegate";

/// The configured LiveKit project (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`), when
/// the endpoint is on its origin.
struct Project {
    url: String,
    api_key: String,
    api_secret: String,
}

impl Project {
    fn of(endpoint: &str) -> Result<Option<Self>, AgentClientError> {
        let var = |name| std::env::var(name).ok().filter(|value: &String| !value.is_empty());
        let (Some(url), Some(api_key), Some(api_secret)) =
            (var("LIVEKIT_URL"), var("LIVEKIT_API_KEY"), var("LIVEKIT_API_SECRET"))
        else {
            return Ok(None);
        };
        let Ok(mut project) = url::Url::parse(&url) else { return Ok(None) };
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
        let destination = super::checked_url(endpoint).map_err(protocol)?;
        if project.origin() != destination.origin() {
            return Ok(None);
        }
        Ok(Some(Project { url: project.to_string(), api_key, api_secret }))
    }
}

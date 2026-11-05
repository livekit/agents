//! A tool call's delegation to a remote agent: A2A owns every decision; this runs it over the
//! endpoint client and the host's progress and ask-the-person callbacks.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::watch;

use lk_agents_abi::a2a::client::{
    self as port, AgentClientError, Message, MessageInput, MessagePart, TaskUpdate,
};
use lk_agents_abi::a2a::remote::{DelegationHost, RemoteEvent, RemoteInteraction as Port};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::invoke::{CallContext, Cancellation as _};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::{ApiConnectOptions, ChatItem, FunctionCall, HOST_TEARDOWN_TIMEOUT};
use lk_agents_core::a2a::caller_history::{CallerHistory, HistorySharing};
use lk_agents_core::a2a::delegation::{
    DelegationUpdate, InteractionId, RemoteInteraction as Policy, retained,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use super::host::message_from_wire;
use super::{AgentClient, Reply};
use crate::env::ProcessEnv;
use crate::task::Cancellation;
use crate::telemetry;

pub struct RemoteInteraction {
    inner: Interaction,
    /// The running delegation's stop and settlement.
    running: Mutex<Option<(Arc<Cancellation>, watch::Receiver<bool>)>>,
}

/// The interaction's decisions, shared with its running delegation.
#[derive(Clone)]
struct Interaction(Arc<Mutex<State>>);

struct State {
    policy: Policy,
    call: FunctionCall,
    livekit: bool,
    project: Project,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextRequest {
    request: String,
}

impl Port for RemoteInteraction {
    type Env = ProcessEnv;

    fn new(
        endpoint: String,
        tool_id: String,
        function_call: ChatItem,
        history: Vec<ChatItem>,
        project_url: String,
        api_key: String,
        api_secret: String,
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
        let project = Project { url: project_url, api_key, api_secret };
        let state = State { policy, call: function_call, livekit: false, project };
        Ok(Self { inner: Interaction(Arc::new(Mutex::new(state))), running: Mutex::new(None) })
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
        self.inner.message(arguments, structured, offers_extension, &history, thread_id, token)
    }

    async fn run(
        &self,
        env: &ProcessEnv,
        host: Arc<dyn DelegationHost>,
        arguments: Value,
        structured: bool,
        history: Vec<ChatItem>,
        token: Option<String>,
        thread_id: Option<String>,
        conn_options: Option<ApiConnectOptions>,
        live: bool,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Result<RemoteEvent, AgentClientError> {
        let endpoint = self.inner.0.lock().policy.snapshot().id.endpoint;
        let client =
            AgentClient::new(env, endpoint, token.clone(), conn_options.unwrap_or_default())
                .map_err(super::host::client_error)?;
        let request = Request {
            arguments,
            structured,
            history,
            shared: thread_id.map(|thread| (thread, token)),
            live,
        };
        let stop = Arc::new(Cancellation::default());
        let (done, settled) = watch::channel(false);
        *self.running.lock() = Some((stop.clone(), settled));
        let delegation = Delegation { interaction: self.inner.clone(), client, host, context, cx };
        // Its own task: a dropped `run` leaves it to the host, which stops it in the same turn
        // (`cancelled()`), and `settled()` waits for its cleanup.
        let task = (env.runtime)().map_err(unavailable)?.spawn(async move {
            let result = delegation.run(request, &stop).await;
            done.send_replace(true);
            result
        });
        task.await.unwrap_or_else(|error| Err(AgentClientError::Protocol(error.to_string())))
    }

    fn cancelled(&self) -> Result<(), AgentClientError> {
        if let Some((stop, _)) = &*self.running.lock() {
            stop.cancel();
        }
        let mut state = self.inner.0.lock();
        state.policy.cancel();
        state.policy.cancellation_settled();
        Ok(())
    }

    async fn settled(&self) {
        let running = self.running.lock().as_ref().map(|(_, settled)| settled.clone());
        if let Some(mut settled) = running
            && tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, settled.wait_for(|done| *done))
                .await
                .is_err()
        {
            log::error!(target: "livekit.agents", "delegation cleanup incomplete");
        }
    }

    fn extra(&self) -> Result<Map<String, Value>, AgentClientError> {
        self.inner.extra()
    }
}

/// What `run` sends: the call's arguments, the caller's history and, when the caller's thread is
/// shared, that thread and the grant for it.
struct Request {
    arguments: Value,
    structured: bool,
    history: Vec<ChatItem>,
    shared: Option<(String, Option<String>)>,
    live: bool,
}

/// One running delegation over its own endpoint client.
struct Delegation {
    interaction: Interaction,
    client: AgentClient,
    host: Arc<dyn DelegationHost>,
    context: CallContext,
    cx: telemetry::Cx,
}

impl Delegation {
    /// The delegation until its outcome or `stop`; its reply and client are closed either way.
    async fn run(
        &self,
        request: Request,
        stop: &Cancellation,
    ) -> Result<RemoteEvent, AgentClientError> {
        let mut reply = None;
        let result = tokio::select! {
            biased;
            () = stop.cancelled() => Ok(RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled }),
            result = self.drive(request, &mut reply) => result,
        };
        if let Some(reply) = reply {
            port::Reply::aclose(&reply).await;
        }
        port::AgentClient::aclose(&self.client).await;
        result
    }

    async fn drive(
        &self,
        request: Request,
        reply: &mut Option<Reply>,
    ) -> Result<RemoteEvent, AgentClientError> {
        let client = &self.client;
        port::AgentClient::connect(client).await?;
        let (thread_id, token) = request.shared.unzip();
        let message = self.interaction.message(
            request.arguments,
            request.structured,
            client.offers_extension(),
            &request.history,
            thread_id,
            token.flatten(),
        )?;
        *reply = Some(self.send(message).await?);
        if request.live {
            self.host.delegated(&self.cx, &self.context).await.map_err(host_error)?;
        }
        loop {
            let event = match reply.as_ref() {
                Some(current) => self.follow(current).await?,
                None => None,
            };
            if let Some(done) = reply.take() {
                port::Reply::aclose(&done).await;
            }
            *reply = Some(match event {
                Some(RemoteEvent::Follow { task_id, thread_id }) => {
                    let cx = self.cx.clone();
                    port::AgentClient::_core_subscribe(client, task_id, thread_id, cx).await?
                }
                Some(RemoteEvent::InputRequired {
                    request_id, question, verbatim_full, ..
                }) if request.live => {
                    let answer = self.host.ask(&self.cx, &self.context, question, verbatim_full);
                    let answer = answer.await.map_err(host_error)?;
                    self.send(self.interaction.answer(&request_id, answer)?).await?
                }
                Some(event @ (RemoteEvent::InputRequired { .. } | RemoteEvent::Outcome { .. })) => {
                    return Ok(event);
                }
                _ => {
                    return Err(AgentClientError::Protocol(
                        "the remote interaction ended without an outcome".into(),
                    ));
                }
            });
        }
    }

    async fn send(&self, message: Message) -> Result<Reply, AgentClientError> {
        let message = MessageInput::Message(Box::new(message));
        let cx = self.cx.clone();
        port::AgentClient::send(&self.client, message, None, None, None, None, cx).await
    }

    /// The reply's updates, each reported to the host, until one ends it.
    async fn follow(&self, reply: &Reply) -> Result<Option<RemoteEvent>, AgentClientError> {
        while let Some(update) = port::Reply::next(reply).await? {
            let event = self.interaction.update(update)?;
            let text = match &event {
                RemoteEvent::Progress { text } => Some(text.clone()),
                _ => None,
            };
            let extra = self.interaction.extra()?;
            let progress = self.host.progress(&self.cx, &self.context, text, extra);
            progress.await.map_err(host_error)?;
            if !matches!(event, RemoteEvent::Pending | RemoteEvent::Progress { .. }) {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }
}

impl Interaction {
    fn message(
        &self,
        arguments: Value,
        structured: bool,
        offers_extension: bool,
        history: &[ChatItem],
        thread_id: Option<String>,
        token: Option<String>,
    ) -> Result<Message, AgentClientError> {
        let mut state = self.0.lock();
        // sharing the caller's thread needs a grant this project signed for it, verified here
        let authority = match (thread_id.as_deref(), token.as_deref()) {
            (Some(thread), Some(token)) => {
                let endpoint = state.policy.snapshot().id.endpoint;
                let project = state.project.serving(&endpoint)?.ok_or_else(|| {
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
                    history,
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

    /// The person's answer to the retained input request `request_id`, on its task; refused
    /// once that request was settled or superseded.
    fn answer(&self, request_id: &str, text: String) -> Result<Message, AgentClientError> {
        let mut state = self.0.lock();
        state.policy.accepts_reply(request_id).map_err(protocol)?;
        let livekit = state.livekit;
        let message = state
            .policy
            .message(text, crate::strings::shortuuid("msg_"), None, "", None, livekit, None)
            .map_err(protocol)?;
        Ok(message_from_wire(message))
    }

    fn update(&self, update: TaskUpdate) -> Result<RemoteEvent, AgentClientError> {
        let mut state = self.0.lock();
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

    fn extra(&self) -> Result<Map<String, Value>, AgentClientError> {
        let state = self.0.lock();
        retained::call_extra(&state.policy, &state.call, state.livekit).map_err(protocol)
    }
}

/// A host callback's failure, raised as it was.
fn host_error(error: HostError) -> AgentClientError {
    AgentClientError::Provider(ProviderError::Host {
        error,
        api_kind: None,
        retryable: false,
        status_code: None,
    })
}

fn unavailable(error: String) -> AgentClientError {
    AgentClientError::Provider(ProviderError::Local(
        lk_agents_abi::errors::ApiError::new(error).with_retryable(false),
    ))
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
    fn _core_delegation_tool(
        name: String,
        description: String,
    ) -> Result<lk_agents_abi::a2a::client::AgentCard, AgentClientError> {
        lk_agents_core::a2a::client::delegation_tool(name, description)
            .map_err(super::host::client_error)
    }

    fn _core_remote_token(
        endpoint: String,
        project_url: String,
        api_key: String,
        api_secret: String,
        token: Option<String>,
        thread_id: Option<String>,
    ) -> Result<Option<String>, AgentClientError> {
        use lk_agents_abi::a2a::auth::{AuthGrant, Credentials as CredentialsApi};

        if token.is_some() {
            return Ok(token);
        }
        let project = Project { url: project_url, api_key, api_secret };
        let Some(project) = project.serving(&endpoint)? else { return Ok(None) };
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

/// The server's LiveKit project as the host passes it; empty values when there is none.
struct Project {
    url: String,
    api_key: String,
    api_secret: String,
}

impl Project {
    /// The project with its URL as an HTTP origin, when `endpoint` is on that origin.
    fn serving(&self, endpoint: &str) -> Result<Option<Self>, AgentClientError> {
        let Project { url, api_key, api_secret } = self;
        if url.is_empty() || api_key.is_empty() || api_secret.is_empty() {
            return Ok(None);
        }
        let (api_key, api_secret) = (api_key.clone(), api_secret.clone());
        let Ok(mut project) = url::Url::parse(url) else { return Ok(None) };
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

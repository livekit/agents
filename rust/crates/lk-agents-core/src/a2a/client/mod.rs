use lk_agents_abi::a2a_protocol as a2a;
mod messages;
pub use messages::{message_from_wire, message_to_wire, metadata_to_wire, part_from_wire};
mod tool;
pub use tool::{delegation_tool, take_task_id, tool_definition};

use a2a::{
    AgentCard, Artifact, CancelTaskRequest, JsonRpcId, JsonRpcRequest, JsonRpcResponse, Message,
    Part, PartContent, SendMessageRequest, StreamResponse, SubscribeToTaskRequest, TaskState,
};
use lk_agents_abi::a2a::metadata::{EXTENSION_URI, MERGED_INTO_KEY, PROGRESS_KEY, REASON_KEY};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::a2a::rpc::TaskRef;

// Neither a connection nor an outstanding task lives on a reusable remote-tool definition.

/// Connection, timeout and status failures are the transport's `ApiError` kinds.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ClientError {
    #[error("invalid A2A input: {0}")]
    InvalidInput(String),
    #[error("A2A error {code}: {message}")]
    Rpc { code: i32, message: String, data: Option<Value> },
    /// The client or the peer broke the protocol: closed, not connected, an invalid or
    /// incomplete reply.
    #[error("{0}")]
    Protocol(String),
    #[error(transparent)]
    Transport(#[from] lk_agents_abi::errors::ApiError),
}

impl ClientError {
    pub fn closed() -> Self {
        Self::Protocol("the A2A client is closed".into())
    }

    pub fn incomplete() -> Self {
        Self::Protocol("the A2A stream ended before an outcome".into())
    }

    pub fn invalid_response(error: impl std::fmt::Display) -> Self {
        Self::Protocol(format!("invalid A2A response: {error}"))
    }

    fn not_connected() -> Self {
        Self::Protocol("connect the A2A client before sending a request".into())
    }
}

/// The host's view of a client failure.
impl From<ClientError> for lk_agents_abi::a2a::client::AgentClientError {
    fn from(error: ClientError) -> Self {
        match error {
            ClientError::InvalidInput(message) => Self::InvalidInput(message),
            ClientError::Transport(error) => Self::Provider(error.into()),
            error => Self::Protocol(error.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

/// A prepared request is retained unchanged when a transport retries it, including messageId.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub id: Option<JsonRpcId>,
    pub streaming: bool,
    /// Safe to repeat without repeating remote work; message and cancel requests are not.
    pub replay_safe: bool,
    /// Expected task identity after resolving explicit, message and implicit routing.
    pub task: Option<TaskRef>,
    /// A routed thread remains authoritative even when the peer creates a new task.
    pub context_id: Option<String>,
    /// Unary task methods return a raw task rather than a streaming response wrapper.
    pub task_response: bool,
}

impl Request {
    /// A streaming event is not a unary SendMessage result. Validate independently of any
    /// conversation observer so low-level clients enforce the same SDK operation contract.
    pub fn validate_response(&self, response: &StreamResponse) -> Result<(), ClientError> {
        if self.task_response && !matches!(response, StreamResponse::Task(_)) {
            return Err(ClientError::invalid_response("a task method returned another response"));
        }
        if !self.streaming
            && !self.task_response
            && !matches!(response, StreamResponse::Task(_) | StreamResponse::Message(_))
        {
            return Err(ClientError::invalid_response("unary send returned a stream delta"));
        }
        Ok(())
    }

    /// The request carrying the caller's span as its W3C `traceparent`, so the endpoint's work
    /// continues the caller's trace.
    pub fn traced(mut self, caller: Option<lk_agents_abi::telemetry::record::SpanContext>) -> Self {
        if let Some(caller) = caller {
            self.headers.push(("traceparent".into(), caller.traceparent()));
        }
        self
    }
}

/// A connection's card and last acknowledged thread. Ordinary endpoint authorization remains
/// the endpoint's responsibility; delegation's separate sharing authority is not imposed here.
pub struct Client {
    endpoint: String,
    token: Option<String>,
    card: Option<AgentCard>,
    rpc_url: Option<String>,
    last_context: Option<String>,
    closed: bool,
}

impl Client {
    pub fn new(endpoint: String, token: Option<String>) -> Result<Self, ClientError> {
        check_url(&endpoint)?;
        if token.as_ref().is_some_and(|t| t.contains(['\r', '\n'])) {
            return Err(ClientError::InvalidInput(
                "a bearer token cannot contain a newline".into(),
            ));
        }
        Ok(Self {
            endpoint: if endpoint.contains('?') {
                endpoint
            } else {
                endpoint.trim_end_matches('/').to_owned()
            },
            token,
            card: None,
            rpc_url: None,
            last_context: None,
            closed: false,
        })
    }

    pub fn last_context(&self) -> Option<&str> {
        self.last_context.as_deref()
    }

    pub fn card(&self) -> Option<&AgentCard> {
        self.card.as_ref()
    }

    pub fn offers_extension(&self) -> bool {
        self.card
            .as_ref()
            .and_then(|c| c.capabilities.extensions.as_ref())
            .is_some_and(|exts| exts.iter().any(|ext| ext.uri == EXTENSION_URI))
    }

    pub fn card_request(&self) -> Result<Request, ClientError> {
        self.ensure_open()?;
        let (path, query) = self
            .endpoint
            .split_once('?')
            .map_or((self.endpoint.as_str(), None), |(path, query)| (path, Some(query)));
        let mut url = format!("{}/.well-known/agent-card.json", path.trim_end_matches('/'));
        if let Some(query) = query {
            url.push('?');
            url.push_str(query);
        }
        Ok(Request {
            method: HttpMethod::Get,
            url,
            headers: self.headers(false),
            body: Vec::new(),
            id: None,
            streaming: false,
            replay_safe: true,
            task: None,
            context_id: None,
            task_response: false,
        })
    }

    /// The card is a capability hint, never proof of authority. Runtime validates its URL and
    /// credentials' destination before accepting the connection.
    pub fn accept_card(&mut self, body: &[u8]) -> Result<&AgentCard, ClientError> {
        self.ensure_open()?;
        if self.card.is_none() {
            let card: AgentCard =
                serde_json::from_slice(body).map_err(ClientError::invalid_response)?;
            let interface = card
                .supported_interfaces
                .iter()
                .find(|i| {
                    i.protocol_binding == a2a::TRANSPORT_PROTOCOL_JSONRPC
                        && i.protocol_version == "1.0"
                })
                .ok_or_else(|| {
                    ClientError::invalid_response("the card offers no A2A 1.0 JSONRPC interface")
                })?;
            check_url(&interface.url).map_err(ClientError::invalid_response)?;
            self.rpc_url = Some(interface.url.clone());
            self.card = Some(card);
        }
        Ok(self.card.as_ref().expect("card was accepted"))
    }

    /// Validate before discovery or any other I/O, using the same routing as `send`.
    pub fn validate_message(
        &self,
        message: &Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<(), ClientError> {
        self.message_route(message, thread_id, task_id).map(|_| ())
    }

    /// Explicit routing takes precedence over routing in the message; otherwise continue the
    /// last acknowledged thread. A pending task must always have its context beside it.
    pub fn send(
        &self,
        mut message: Message,
        thread_id: Option<&str>,
        task_id: Option<&str>,
        request_id: JsonRpcId,
    ) -> Result<Request, ClientError> {
        self.ensure_connected()?;
        let (context_id, task_id) = self.message_route(&message, thread_id, task_id)?;
        let routing = (context_id.map(str::to_owned), task_id.map(str::to_owned));
        message.context_id = routing.0;
        message.task_id = routing.1;
        let streaming = self.card.as_ref().is_some_and(|c| c.capabilities.streaming == Some(true));
        self.resume_send(
            SendMessageRequest { message, configuration: None, metadata: None, tenant: None },
            streaming,
            request_id,
        )
    }

    /// An unsent durable intent keeps its original SDK body and method. Discovery can change
    /// capabilities, but cannot rewrite the saved operation into different business work.
    pub fn resume_send(
        &self,
        request: SendMessageRequest,
        streaming: bool,
        request_id: JsonRpcId,
    ) -> Result<Request, ClientError> {
        self.ensure_connected()?;
        self.validate_message(&request.message, None, None)?;
        let message = &request.message;
        if message.task_id.is_some() && message.context_id.is_none() {
            return Err(ClientError::InvalidInput("a saved task message needs its context".into()));
        }
        if streaming && !self.card.as_ref().is_some_and(|c| c.capabilities.streaming == Some(true))
        {
            return Err(ClientError::InvalidInput(
                "the peer no longer supports the saved streaming operation".into(),
            ));
        }
        let task = message
            .task_id
            .as_ref()
            .zip(message.context_id.as_ref())
            .map(|(id, context_id)| TaskRef { id: id.clone(), context_id: context_id.clone() });
        let context_id = message.context_id.clone();
        let method = if streaming {
            a2a::methods::SEND_STREAMING_MESSAGE
        } else {
            a2a::methods::SEND_MESSAGE
        };
        self.rpc(method, request, request_id, streaming, task, context_id)
    }

    pub fn cancel(
        &self,
        task: &TaskRef,
        reason: Option<&str>,
        id: JsonRpcId,
    ) -> Result<Request, ClientError> {
        check_task(task)?;
        let metadata = reason.map(|reason| {
            [(REASON_KEY.to_owned(), Value::String(reason.into()))].into_iter().collect()
        });
        self.resume_cancel(
            task,
            CancelTaskRequest { id: task.id.clone(), metadata, tenant: None },
            id,
        )
    }

    /// An undispatched control keeps its exact SDK body. The saved task association supplies
    /// the context extension; no new cancellation is inferred from an uncertain operation.
    pub fn resume_cancel(
        &self,
        task: &TaskRef,
        request: CancelTaskRequest,
        id: JsonRpcId,
    ) -> Result<Request, ClientError> {
        check_task(task)?;
        if request.id != task.id {
            return Err(ClientError::InvalidInput(
                "the saved cancellation belongs to another task".into(),
            ));
        }
        self.rpc(
            a2a::methods::CANCEL_TASK,
            TaskParams { request, context_id: &task.context_id, offset: None },
            id,
            false,
            Some(task.clone()),
            Some(task.context_id.clone()),
        )
    }

    pub fn get_task(&self, task: &TaskRef, id: JsonRpcId) -> Result<Request, ClientError> {
        check_task(task)?;
        self.rpc(
            a2a::methods::GET_TASK,
            a2a::GetTaskRequest { id: task.id.clone(), history_length: None, tenant: None },
            id,
            false,
            Some(task.clone()),
            Some(task.context_id.clone()),
        )
    }

    pub fn subscribe(
        &self,
        task: &TaskRef,
        offset: Option<&str>,
        id: JsonRpcId,
    ) -> Result<Request, ClientError> {
        check_task(task)?;
        self.rpc(
            a2a::methods::SUBSCRIBE_TO_TASK,
            TaskParams {
                request: SubscribeToTaskRequest { id: task.id.clone(), tenant: None },
                context_id: &task.context_id,
                offset,
            },
            id,
            true,
            Some(task.clone()),
            Some(task.context_id.clone()),
        )
    }

    /// Only an acknowledged task or message updates the connection's implicit routing.
    pub fn acknowledge(&mut self, update: &TaskUpdate) -> Result<(), ClientError> {
        self.acknowledge_context(&update.context_id)
    }

    pub fn acknowledge_context(&mut self, context_id: &str) -> Result<(), ClientError> {
        self.ensure_open()?;
        if !context_id.is_empty() {
            self.last_context = Some(context_id.to_owned());
        }
        Ok(())
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    fn ensure_open(&self) -> Result<(), ClientError> {
        if self.closed { Err(ClientError::closed()) } else { Ok(()) }
    }

    fn ensure_connected(&self) -> Result<(), ClientError> {
        self.ensure_open()?;
        if self.card.is_none() { Err(ClientError::not_connected()) } else { Ok(()) }
    }

    fn message_route<'a>(
        &'a self,
        message: &'a Message,
        thread_id: Option<&'a str>,
        task_id: Option<&'a str>,
    ) -> Result<(Option<&'a str>, Option<&'a str>), ClientError> {
        self.ensure_open()?;
        if message.message_id.is_empty()
            || message.parts.is_empty()
            || message.role == a2a::Role::Unspecified
        {
            return Err(ClientError::InvalidInput(
                "a message needs an id, a role and content parts".into(),
            ));
        }
        let context_id =
            thread_id.or(message.context_id.as_deref()).or(self.last_context.as_deref());
        let task_id = task_id.or(message.task_id.as_deref());
        if context_id == Some("")
            || task_id == Some("")
            || (task_id.is_some() && context_id.is_none())
        {
            return Err(ClientError::InvalidInput(
                "a pending task needs nonempty task and context ids".into(),
            ));
        }
        Ok((context_id, task_id))
    }

    fn headers(&self, streaming: bool) -> Vec<(String, String)> {
        let mut headers = vec![
            ("A2A-Version".into(), "1.0".into()),
            (
                "Accept".into(),
                if streaming { "text/event-stream" } else { "application/json" }.into(),
            ),
        ];
        if let Some(token) = &self.token {
            headers.push(("Authorization".into(), format!("Bearer {token}")));
        }
        headers
    }

    fn rpc(
        &self,
        method: &str,
        params: impl Serialize,
        id: JsonRpcId,
        streaming: bool,
        task: Option<TaskRef>,
        context_id: Option<String>,
    ) -> Result<Request, ClientError> {
        self.ensure_connected()?;
        if id == JsonRpcId::Null {
            return Err(ClientError::InvalidInput("a request needs a JSON-RPC id".into()));
        }
        let params =
            serde_json::to_value(params).map_err(|e| ClientError::InvalidInput(e.to_string()))?;
        let envelope = JsonRpcRequest::new(id.clone(), method, Some(params));
        let body =
            serde_json::to_vec(&envelope).map_err(|e| ClientError::InvalidInput(e.to_string()))?;
        let mut headers = self.headers(streaming);
        headers.push(("Content-Type".into(), "application/json".into()));
        Ok(Request {
            method: HttpMethod::Post,
            url: self.rpc_url.clone().ok_or_else(ClientError::not_connected)?,
            headers,
            body,
            id: Some(id),
            streaming,
            replay_safe: matches!(method, a2a::methods::SUBSCRIBE_TO_TASK | a2a::methods::GET_TASK),
            task,
            context_id,
            task_response: matches!(method, a2a::methods::CANCEL_TASK | a2a::methods::GET_TASK),
        })
    }
}

/// LiveKit's task addressing additions on the upstream request records.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TaskParams<'a, T> {
    #[serde(flatten)]
    request: T,
    context_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<&'a str>,
}

fn check_url(url: &str) -> Result<(), ClientError> {
    if !(url.starts_with("https://") || url.starts_with("http://"))
        || url.chars().any(char::is_whitespace)
    {
        return Err(ClientError::InvalidInput(
            "an endpoint must be an absolute HTTP(S) URL".into(),
        ));
    }
    Ok(())
}

fn check_task(task: &TaskRef) -> Result<(), ClientError> {
    if task.id.is_empty() || task.context_id.is_empty() {
        return Err(ClientError::InvalidInput("task methods require task and context ids".into()));
    }
    Ok(())
}

/// A task's progress or observed outcome. `follow_task` names the task that owns a merged
/// answer; it must be followed rather than treating this task as an empty successful answer.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskUpdate {
    pub state: TaskState,
    pub text: String,
    pub progress: bool,
    pub task_id: String,
    pub context_id: String,
    pub metadata: Map<String, Value>,
    pub artifacts: Vec<Artifact>,
    /// Original SDK direct reply or status message, without flattening data/file parts.
    pub message: Option<Message>,
    pub follow_task: Option<TaskRef>,
}

/// One validated SDK response, retained beside its presentation projection. Artifact events
/// and direct message replies keep their protocol identity even when presentation has none.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplyEvent {
    pub response: StreamResponse,
    pub update: Option<TaskUpdate>,
}

impl ReplyEvent {
    pub fn route(&self) -> TaskRef {
        let (id, context_id) = match &self.response {
            StreamResponse::Task(task) => (task.id.clone(), task.context_id.clone()),
            StreamResponse::Message(message) => (
                message.task_id.clone().unwrap_or_default(),
                message.context_id.clone().unwrap_or_default(),
            ),
            StreamResponse::StatusUpdate(event) => {
                (event.task_id.clone(), event.context_id.clone())
            }
            StreamResponse::ArtifactUpdate(event) => {
                (event.task_id.clone(), event.context_id.clone())
            }
        };
        TaskRef { id, context_id }
    }
}

impl TaskUpdate {
    pub fn is_settled(&self) -> bool {
        self.state.is_terminal()
            || matches!(self.state, TaskState::InputRequired | TaskState::AuthRequired)
    }
}

/// One response stream's identity, artifact assembly and observed outcome.
#[derive(Debug, Default)]
pub struct ReplyState {
    task: Option<TaskRef>,
    expected_context: Option<String>,
    artifacts: Vec<Artifact>,
    settled: bool,
}

impl ReplyState {
    pub fn new(task: Option<TaskRef>) -> Self {
        let expected_context = task.as_ref().map(|task| task.context_id.clone());
        Self { task, expected_context, ..Self::default() }
    }

    pub fn for_request(request: &Request) -> Self {
        let mut state = Self::new(request.task.clone());
        if request.context_id.is_some() {
            state.expected_context = request.context_id.clone();
        }
        state
    }

    pub fn task_ref(&self) -> Option<TaskRef> {
        self.task.clone()
    }

    pub fn settled(&self) -> bool {
        self.settled
    }

    /// JSON-RPC validation is shared for streaming and ordinary responses. No response from a
    /// different request or task is accepted into this reply.
    pub fn decode(
        &mut self,
        body: &[u8],
        id: &JsonRpcId,
    ) -> Result<Option<TaskUpdate>, ClientError> {
        self.decode_event(body, id, false).map(|event| event.update)
    }

    /// Protocol observations reach the owner before it decides what to present or persist.
    /// `task_response` names unary SDK methods whose result is a Task without a union wrapper.
    pub fn decode_event(
        &mut self,
        body: &[u8],
        id: &JsonRpcId,
        task_response: bool,
    ) -> Result<ReplyEvent, ClientError> {
        let response = decode_response(body, id, task_response)?;
        let update = self.accept(response.clone())?;
        Ok(ReplyEvent { response, update })
    }

    /// `CancelTask` and `GetTask` return a task directly, rather than a streaming wrapper.
    pub fn decode_task(&mut self, body: &[u8], id: &JsonRpcId) -> Result<TaskUpdate, ClientError> {
        let result = response_result(body, id)?;
        self.accept(StreamResponse::Task(
            serde_json::from_value(result).map_err(ClientError::invalid_response)?,
        ))?
        .ok_or_else(|| ClientError::invalid_response("a task response did not produce an update"))
    }

    pub fn accept(&mut self, event: StreamResponse) -> Result<Option<TaskUpdate>, ClientError> {
        if self.settled {
            return Err(ClientError::invalid_response("the task already supplied its outcome"));
        }
        match event {
            StreamResponse::Task(task) => {
                self.identify(&task.id, &task.context_id)?;
                if let Some(artifacts) = task.artifacts {
                    self.artifacts = artifacts;
                }
                self.status(
                    task.status,
                    task.metadata.map(|m| m.into_iter().collect()).unwrap_or_default(),
                )
            }
            StreamResponse::StatusUpdate(event) => {
                self.identify(&event.task_id, &event.context_id)?;
                self.status(
                    event.status,
                    event.metadata.map(|m| m.into_iter().collect()).unwrap_or_default(),
                )
            }
            StreamResponse::ArtifactUpdate(event) => self.artifact_update(event),
            StreamResponse::Message(message) => self.direct_reply(message),
        }
    }

    fn artifact_update(
        &mut self,
        event: a2a::TaskArtifactUpdateEvent,
    ) -> Result<Option<TaskUpdate>, ClientError> {
        self.identify(&event.task_id, &event.context_id)?;
        let existing =
            self.artifacts.iter_mut().find(|a| a.artifact_id == event.artifact.artifact_id);
        match existing {
            Some(artifact) if event.append == Some(true) => {
                append_artifact(artifact, event.artifact)
            }
            Some(artifact) => *artifact = event.artifact,
            None => self.artifacts.push(event.artifact),
        }
        Ok(None)
    }

    /// A message instead of a task: the remote answered at once and the reply is settled.
    fn direct_reply(&mut self, message: Message) -> Result<Option<TaskUpdate>, ClientError> {
        if self.task.is_some() {
            return Err(ClientError::invalid_response(
                "a task stream cannot turn into an unrelated message",
            ));
        }
        if self
            .expected_context
            .as_ref()
            .is_some_and(|context| Some(context) != message.context_id.as_ref())
        {
            return Err(ClientError::invalid_response("the message belongs to another context"));
        }
        self.settled = true;
        let text = parts_text(&message.parts);
        Ok(Some(TaskUpdate {
            state: TaskState::Completed,
            text,
            progress: false,
            task_id: message.task_id.clone().unwrap_or_default(),
            context_id: message.context_id.clone().unwrap_or_default(),
            metadata: short_metadata(
                message.metadata.clone().map(|m| m.into_iter().collect()).unwrap_or_default(),
            ),
            artifacts: Vec::new(),
            message: Some(message),
            follow_task: None,
        }))
    }

    /// Transport EOF is an observation, never evidence that remote work completed.
    pub fn finish(&self) -> Result<(), ClientError> {
        if self.settled { Ok(()) } else { Err(ClientError::incomplete()) }
    }

    fn identify(&mut self, id: &str, context_id: &str) -> Result<(), ClientError> {
        if id.is_empty() || context_id.is_empty() {
            return Err(ClientError::invalid_response("a task update needs task and context ids"));
        }
        if self.expected_context.as_deref().is_some_and(|expected| expected != context_id) {
            return Err(ClientError::invalid_response("the update belongs to another context"));
        }
        match &self.task {
            Some(task) if task.id != id || task.context_id != context_id => {
                Err(ClientError::invalid_response("the update belongs to another task or context"))
            }
            Some(_) => Ok(()),
            None => {
                self.task = Some(TaskRef { id: id.into(), context_id: context_id.into() });
                Ok(())
            }
        }
    }

    fn status(
        &mut self,
        status: a2a::TaskStatus,
        mut metadata: Map<String, Value>,
    ) -> Result<Option<TaskUpdate>, ClientError> {
        let state = status.state;
        let progress = state == TaskState::Working
            && (metadata.get(PROGRESS_KEY).and_then(Value::as_bool) == Some(true)
                || status
                    .message
                    .as_ref()
                    .and_then(|message| message.metadata.as_ref())
                    .and_then(|metadata| metadata.get(PROGRESS_KEY))
                    .and_then(Value::as_bool)
                    == Some(true));
        if let Some(message) = &status.message
            && let Some(message_metadata) = &message.metadata
        {
            let mut keys: Vec<_> = message_metadata.iter().collect();
            keys.sort_by_key(|(key, _)| *key);
            for (key, value) in keys {
                metadata.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        let task = self.task.as_ref().expect("status has an identified task");
        let follow_task = match metadata.get(MERGED_INTO_KEY) {
            None => None,
            Some(Value::String(id))
                if !id.is_empty() && id != &task.id && state == TaskState::Completed =>
            {
                Some(TaskRef { id: id.clone(), context_id: task.context_id.clone() })
            }
            Some(_) => {
                return Err(ClientError::invalid_response(
                    "mergedInto must name a different task on a completed outcome",
                ));
            }
        };
        self.settled = state.is_terminal()
            || matches!(state, TaskState::InputRequired | TaskState::AuthRequired);
        let text = if follow_task.is_some() {
            String::new()
        } else if state == TaskState::Completed && !self.artifacts.is_empty() {
            answer_text(&self.artifacts)
        } else {
            status.message.as_ref().map(|m| parts_text(&m.parts)).unwrap_or_default()
        };
        Ok(Some(TaskUpdate {
            state,
            text,
            progress,
            task_id: task.id.clone(),
            context_id: task.context_id.clone(),
            metadata: short_metadata(metadata),
            artifacts: self.artifacts.clone(),
            message: status.message,
            follow_task,
        }))
    }
}

/// Decode the protocol value independently of presentation normalization. Native observers
/// retain this value independently of how a consumer presents its structured content.
pub fn decode_response(
    body: &[u8],
    id: &JsonRpcId,
    task_response: bool,
) -> Result<StreamResponse, ClientError> {
    let result = response_result(body, id)?;
    if task_response {
        Ok(StreamResponse::Task(
            serde_json::from_value(result).map_err(ClientError::invalid_response)?,
        ))
    } else {
        serde_json::from_value(result).map_err(ClientError::invalid_response)
    }
}

/// An appended chunk adds its parts and replaces only the descriptive fields it carries.
pub(crate) fn append_artifact(into: &mut Artifact, update: Artifact) {
    into.parts.extend(update.parts);
    if update.name.is_some() {
        into.name = update.name;
    }
    if update.description.is_some() {
        into.description = update.description;
    }
    if update.metadata.is_some() {
        into.metadata = update.metadata;
    }
    if update.extensions.is_some() {
        into.extensions = update.extensions;
    }
}

fn response_result(body: &[u8], id: &JsonRpcId) -> Result<Value, ClientError> {
    let response: JsonRpcResponse =
        serde_json::from_slice(body).map_err(ClientError::invalid_response)?;
    if response.jsonrpc != "2.0" || &response.id != id {
        return Err(ClientError::invalid_response(
            "the JSON-RPC response does not match its request",
        ));
    }
    match (response.result, response.error) {
        (None, Some(error)) => {
            Err(ClientError::Rpc { code: error.code, message: error.message, data: error.data })
        }
        (Some(result), None) => Ok(result),
        _ => Err(ClientError::invalid_response(
            "a JSON-RPC response needs exactly one result or error",
        )),
    }
}

fn parts_text(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match &part.content {
            PartContent::Text(text) => Some(text.clone()),
            PartContent::Data(value) => Some(value.to_string()),
            PartContent::Raw(_) | PartContent::Url(_) => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn answer_text(artifacts: &[Artifact]) -> String {
    let named = artifacts.iter().any(|a| a.name.as_deref() == Some("answer"));
    artifacts
        .iter()
        .filter(|a| !named || a.name.as_deref() == Some("answer"))
        .map(|a| parts_text(&a.parts))
        .collect::<Vec<_>>()
        .join("\n")
}

fn short_metadata(metadata: Map<String, Value>) -> Map<String, Value> {
    let prefix = format!("{EXTENSION_URI}/");
    metadata
        .into_iter()
        .filter_map(|(key, value)| {
            // Shorthand is an API presentation only: a peer's raw shorthand must not impersonate
            // the namespaced key that carries framework semantics.
            if key.starts_with("livekit/") {
                return None;
            }
            let key =
                key.strip_prefix(&prefix).map(|suffix| format!("livekit/{suffix}")).unwrap_or(key);
            Some((key, value))
        })
        .collect()
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;

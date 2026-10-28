//! The A2A client's synchronous protocol state. Runtime executes [`Request`]s and feeds each
//! response to its own [`ReplyState`]; neither a connection nor an outstanding task lives on a
//! reusable remote-tool definition.

use a2a::{
    AgentCard, Artifact, CancelTaskRequest, JsonRpcId, JsonRpcRequest, JsonRpcResponse, Message,
    Part, PartContent, SendMessageRequest, StreamResponse, SubscribeToTaskRequest,
};
use lk_agents_storage::run::{EXTENSION_URI, MERGED_INTO_KEY, REASON_KEY, TaskState};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::rpc::{TaskRef, task_state};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ClientError {
    #[error("the A2A client is closed")]
    Closed,
    #[error("connect the A2A client before sending a request")]
    NotConnected,
    #[error("invalid A2A input: {0}")]
    InvalidInput(String),
    #[error("invalid A2A response: {0}")]
    InvalidResponse(String),
    #[error("A2A error {code}: {message}")]
    Rpc { code: i32, message: String, data: Option<Value> },
    #[error("the A2A stream ended before an outcome")]
    Incomplete,
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
    /// Expected task identity after resolving explicit, message and implicit routing.
    pub task: Option<TaskRef>,
    /// A routed thread remains authoritative even when the peer creates a new task.
    pub context_id: Option<String>,
    /// Unary task methods return a raw task rather than a streaming response wrapper.
    pub task_response: bool,
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
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            token,
            card: None,
            rpc_url: None,
            last_context: None,
            closed: false,
        })
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
        Ok(Request {
            method: HttpMethod::Get,
            url: format!("{}/.well-known/agent-card.json", self.endpoint),
            headers: self.headers(false),
            body: Vec::new(),
            id: None,
            streaming: false,
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
            let card: AgentCard = serde_json::from_slice(body).map_err(malformed)?;
            let interface = card
                .supported_interfaces
                .iter()
                .find(|i| {
                    i.protocol_binding == a2a::TRANSPORT_PROTOCOL_JSONRPC
                        && i.protocol_version == "1.0"
                })
                .ok_or_else(|| {
                    ClientError::InvalidResponse(
                        "the card offers no A2A 1.0 JSONRPC interface".into(),
                    )
                })?;
            check_url(&interface.url).map_err(|e| ClientError::InvalidResponse(e.to_string()))?;
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
        let task = message
            .task_id
            .as_ref()
            .zip(message.context_id.as_ref())
            .map(|(id, context_id)| TaskRef { id: id.clone(), context_id: context_id.clone() });
        let context_id = message.context_id.clone();
        let streaming = self.card.as_ref().is_some_and(|c| c.capabilities.streaming == Some(true));
        let method = if streaming {
            a2a::methods::SEND_STREAMING_MESSAGE
        } else {
            a2a::methods::SEND_MESSAGE
        };
        self.rpc(
            method,
            SendMessageRequest { message, configuration: None, metadata: None, tenant: None },
            request_id,
            streaming,
            task,
            context_id,
        )
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
        self.rpc(
            a2a::methods::CANCEL_TASK,
            TaskParams {
                request: CancelTaskRequest { id: task.id.clone(), metadata, tenant: None },
                context_id: &task.context_id,
                offset: None,
            },
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
        self.ensure_open()?;
        if !update.context_id.is_empty() {
            self.last_context = Some(update.context_id.clone());
        }
        Ok(())
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    fn ensure_open(&self) -> Result<(), ClientError> {
        if self.closed { Err(ClientError::Closed) } else { Ok(()) }
    }

    fn ensure_connected(&self) -> Result<(), ClientError> {
        self.ensure_open()?;
        if self.card.is_none() { Err(ClientError::NotConnected) } else { Ok(()) }
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
            url: self.rpc_url.clone().ok_or(ClientError::NotConnected)?,
            headers,
            body,
            id: Some(id),
            streaming,
            task,
            context_id,
            task_response: method == a2a::methods::CANCEL_TASK,
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

fn malformed(error: impl std::fmt::Display) -> ClientError {
    ClientError::InvalidResponse(error.to_string())
}

/// A task's progress or observed outcome. `follow_task` names the task that owns a merged
/// answer; it must be followed rather than treating this task as an empty successful answer.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskUpdate {
    pub state: TaskState,
    pub text: String,
    pub task_id: String,
    pub context_id: String,
    pub metadata: Map<String, Value>,
    pub artifacts: Vec<Artifact>,
    pub follow_task: Option<TaskRef>,
}

impl TaskUpdate {
    pub fn is_settled(&self) -> bool {
        self.state.is_terminal() || self.state == TaskState::InputRequired
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

    pub fn task_id(&self) -> Option<&str> {
        self.task.as_ref().map(|t| t.id.as_str())
    }

    pub fn context_id(&self) -> Option<&str> {
        self.task.as_ref().map(|t| t.context_id.as_str()).or(self.expected_context.as_deref())
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
        let result = response_result(body, id)?;
        self.accept(serde_json::from_value(result).map_err(malformed)?)
    }

    /// `CancelTask` and `GetTask` return a task directly, rather than a streaming wrapper.
    pub fn decode_task(&mut self, body: &[u8], id: &JsonRpcId) -> Result<TaskUpdate, ClientError> {
        let result = response_result(body, id)?;
        self.accept(StreamResponse::Task(serde_json::from_value(result).map_err(malformed)?))?
            .ok_or_else(|| malformed("a task response did not produce an update"))
    }

    pub fn accept(&mut self, event: StreamResponse) -> Result<Option<TaskUpdate>, ClientError> {
        if self.settled {
            return Err(malformed("the task already supplied its outcome"));
        }
        match event {
            StreamResponse::Task(task) => {
                self.identify(&task.id, &task.context_id)?;
                self.artifacts = task.artifacts.unwrap_or_default();
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
            StreamResponse::ArtifactUpdate(event) => {
                self.identify(&event.task_id, &event.context_id)?;
                let existing =
                    self.artifacts.iter_mut().find(|a| a.artifact_id == event.artifact.artifact_id);
                match existing {
                    Some(artifact) if event.append == Some(true) => {
                        artifact.parts.extend(event.artifact.parts);
                        if event.artifact.name.is_some() {
                            artifact.name = event.artifact.name;
                        }
                        if event.artifact.description.is_some() {
                            artifact.description = event.artifact.description;
                        }
                        if event.artifact.metadata.is_some() {
                            artifact.metadata = event.artifact.metadata;
                        }
                    }
                    Some(artifact) => *artifact = event.artifact,
                    None => self.artifacts.push(event.artifact),
                }
                Ok(None)
            }
            StreamResponse::Message(message) => {
                if self.task.is_some() {
                    return Err(malformed("a task stream cannot turn into an unrelated message"));
                }
                if self
                    .expected_context
                    .as_ref()
                    .is_some_and(|context| Some(context) != message.context_id.as_ref())
                {
                    return Err(malformed("the message belongs to another context"));
                }
                self.settled = true;
                let text = parts_text(&message.parts);
                let artifacts = vec![Artifact {
                    artifact_id: message.message_id,
                    name: None,
                    description: None,
                    parts: message.parts,
                    metadata: None,
                    extensions: None,
                }];
                Ok(Some(TaskUpdate {
                    state: TaskState::Completed,
                    text,
                    task_id: message.task_id.unwrap_or_default(),
                    context_id: message.context_id.unwrap_or_default(),
                    metadata: short_metadata(
                        message.metadata.map(|m| m.into_iter().collect()).unwrap_or_default(),
                    ),
                    artifacts,
                    follow_task: None,
                }))
            }
        }
    }

    /// Transport EOF is an observation, never evidence that remote work completed.
    pub fn finish(&self) -> Result<(), ClientError> {
        if self.settled { Ok(()) } else { Err(ClientError::Incomplete) }
    }

    fn identify(&mut self, id: &str, context_id: &str) -> Result<(), ClientError> {
        if id.is_empty() || context_id.is_empty() {
            return Err(malformed("a task update needs task and context ids"));
        }
        if self.expected_context.as_deref().is_some_and(|expected| expected != context_id) {
            return Err(malformed("the update belongs to another context"));
        }
        match &self.task {
            Some(task) if task.id != id || task.context_id != context_id => {
                Err(malformed("the update belongs to another task or context"))
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
        let state = task_state(&status.state)
            .ok_or_else(|| malformed("the peer reported an unsupported task state"))?;
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
                return Err(malformed(
                    "mergedInto must name a different task on a completed outcome",
                ));
            }
        };
        self.settled = state.is_terminal() || state == TaskState::InputRequired;
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
            task_id: task.id.clone(),
            context_id: task.context_id.clone(),
            metadata: short_metadata(metadata),
            artifacts: self.artifacts.clone(),
            follow_task,
        }))
    }
}

fn response_result(body: &[u8], id: &JsonRpcId) -> Result<Value, ClientError> {
    let response: JsonRpcResponse = serde_json::from_slice(body).map_err(malformed)?;
    if response.jsonrpc != "2.0" || &response.id != id {
        return Err(malformed("the JSON-RPC response does not match its request"));
    }
    match (response.result, response.error) {
        (None, Some(error)) => {
            Err(ClientError::Rpc { code: error.code, message: error.message, data: error.data })
        }
        (Some(result), None) => Ok(result),
        _ => Err(malformed("a JSON-RPC response needs exactly one result or error")),
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
mod tests;

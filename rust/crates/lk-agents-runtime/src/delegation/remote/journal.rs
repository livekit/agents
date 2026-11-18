use std::sync::Arc;

use lk_agents_abi::BoxFuture;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::{SessionCommands, ToolCallRef};
use lk_agents_core::a2a::client::{ClientError, Request};
use lk_agents_net::a2a_client::observer::RequestObserver;
use parking_lot::Mutex;

/// Each transport exchange reports into the conversation's serialized command port.
pub struct Journal {
    owner: Arc<dyn SessionCommands>,
    endpoint: String,
    remote_id: String,
    request_id: String,
    origin: Option<ToolCallRef>,
    input_request: Option<InputRequestRef>,
    observed_input: Mutex<Option<InputRequestRef>>,
    control_scope: Option<RemoteControlScope>,
}

impl Journal {
    pub fn new(
        owner: Arc<dyn SessionCommands>,
        endpoint: String,
        remote_id: String,
        request_id: String,
        origin: Option<ToolCallRef>,
        input_request: Option<InputRequestRef>,
    ) -> Self {
        Self {
            owner,
            endpoint,
            remote_id,
            request_id,
            origin,
            input_request,
            observed_input: Mutex::new(None),
            control_scope: None,
        }
    }

    pub(super) fn with_control_scope(mut self, scope: Option<RemoteControlScope>) -> Self {
        self.control_scope = scope;
        self
    }

    /// Only this exchange's acknowledged observation can be attached to its presentation.
    pub fn observed_input(&self) -> Option<InputRequestRef> {
        self.observed_input.lock().clone()
    }

    async fn apply(&self, input: RemoteInput) -> Result<Option<InputRequestRef>, ClientError> {
        self.owner
            .remote(RemoteCommand {
                remote_id: self.remote_id.clone(),
                endpoint: self.endpoint.clone(),
                input,
            })
            .await
            .map_err(|error| ClientError::Protocol(error.to_string()))
    }
}

impl RequestObserver for Journal {
    fn prepared<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<(), ClientError>> {
        Box::pin(async move {
            let rpc: a2a::JsonRpcRequest =
                serde_json::from_slice(&request.body).map_err(ClientError::invalid_response)?;
            let params = rpc.params.ok_or_else(|| {
                ClientError::InvalidInput(
                    "an observed operation needs SDK request parameters".into(),
                )
            })?;
            use a2a::jsonrpc::methods;
            let body = match rpc.method.as_str() {
                methods::SEND_MESSAGE | methods::SEND_STREAMING_MESSAGE => A2ARequest::Send(
                    serde_json::from_value(params).map_err(ClientError::invalid_response)?,
                ),
                methods::GET_TASK => A2ARequest::Get(
                    serde_json::from_value(params).map_err(ClientError::invalid_response)?,
                ),
                methods::LIST_TASKS => A2ARequest::List(
                    serde_json::from_value(params).map_err(ClientError::invalid_response)?,
                ),
                methods::CANCEL_TASK => A2ARequest::Cancel(
                    serde_json::from_value(params).map_err(ClientError::invalid_response)?,
                ),
                methods::SUBSCRIBE_TO_TASK => A2ARequest::Subscribe(
                    serde_json::from_value(params).map_err(ClientError::invalid_response)?,
                ),
                _ => {
                    return Err(ClientError::InvalidInput(
                        "the client operation has no journal codec".into(),
                    ));
                }
            };
            let input_request =
                matches!(body, A2ARequest::Send(_)).then(|| self.input_request.clone()).flatten();
            let control_scope =
                matches!(body, A2ARequest::Send(_)).then(|| self.control_scope.clone()).flatten();
            self.apply(RemoteInput::Prepare {
                request_id: self.request_id.clone(),
                request: body,
                streaming: request.streaming,
                origin: self.origin.clone(),
                input_request,
                control_scope: control_scope.map(Box::new),
            })
            .await
            .map(|_| ())
        })
    }

    fn dispatch_started(&self) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async {
            self.apply(RemoteInput::DispatchStarted { request_id: self.request_id.clone() })
                .await
                .map(|_| ())
        })
    }

    fn response<'a>(
        &'a self,
        request: &'a Request,
        response: a2a::StreamResponse,
    ) -> BoxFuture<'a, Result<a2a::StreamResponse, ClientError>> {
        Box::pin(async move {
            let raw = response.clone();
            let response = if request.task_response {
                match response {
                    a2a::StreamResponse::Task(task) => A2AResponse::Task(task),
                    other => A2AResponse::Stream(other),
                }
            } else if request.streaming {
                A2AResponse::Stream(response)
            } else {
                match response {
                    a2a::StreamResponse::Task(task) => {
                        A2AResponse::Send(a2a::SendMessageResponse::Task(task))
                    }
                    a2a::StreamResponse::Message(message) => {
                        A2AResponse::Send(a2a::SendMessageResponse::Message(message))
                    }
                    // Preserve typed peer content even when it belongs to the wrong SDK
                    // operation. Core journals that evidence before refusing its projection.
                    other => A2AResponse::Stream(other),
                }
            };
            let input = self
                .apply(RemoteInput::Response {
                    request_id: self.request_id.clone(),
                    response: Box::new(response),
                })
                .await?;
            *self.observed_input.lock() = input;
            let target = match &raw {
                a2a::StreamResponse::Task(task) => Some(RemoteTaskRef {
                    task_id: task.id.clone(),
                    context_id: task.context_id.clone(),
                }),
                a2a::StreamResponse::StatusUpdate(update) => Some(RemoteTaskRef {
                    task_id: update.task_id.clone(),
                    context_id: update.context_id.clone(),
                }),
                a2a::StreamResponse::ArtifactUpdate(update) => Some(RemoteTaskRef {
                    task_id: update.task_id.clone(),
                    context_id: update.context_id.clone(),
                }),
                // A direct message remains a message, never a synthetic task result.
                a2a::StreamResponse::Message(_) => None,
            };
            let Some(target) = target else { return Ok(raw) };
            let accepted = self
                .owner
                .recover(
                    &self.remote_id,
                    &self.endpoint,
                    &lk_agents_abi::session::SessionRecovery::Observation(target),
                )
                .await
                .map_err(|error| ClientError::Protocol(error.to_string()))?;
            match accepted {
                None => Ok(raw),
                Some(RemoteRecovery::Observed(response)) => {
                    let a2a::SendMessageResponse::Task(task) = *response else {
                        return Err(ClientError::Protocol(
                            "task observation recovery returned a direct message".into(),
                        ));
                    };
                    // A late input-required poll cannot bind or present a retired question.
                    *self.observed_input.lock() = None;
                    Ok(a2a::StreamResponse::Task(task))
                }
                Some(_) => Err(ClientError::Protocol(
                    "task observation recovery returned a transport operation".into(),
                )),
            }
        })
    }

    fn ended(&self, error: Option<ClientError>) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            let input = match error {
                Some(ClientError::Rpc { code, message, data }) => RemoteInput::Error {
                    request_id: self.request_id.clone(),
                    error: RemoteFailure::Rpc { error: a2a::JsonRpcError { code, message, data } },
                },
                Some(ClientError::Transport(error)) => RemoteInput::Error {
                    request_id: self.request_id.clone(),
                    error: RemoteFailure::Transport {
                        kind: error.error_kind,
                        message: error.message,
                        status_code: error.status_code,
                    },
                },
                Some(error) => RemoteInput::Error {
                    request_id: self.request_id.clone(),
                    error: RemoteFailure::Protocol { message: error.to_string() },
                },
                None => RemoteInput::Ended { request_id: self.request_id.clone() },
            };
            self.apply(input).await.map(|_| ())
        })
    }
}

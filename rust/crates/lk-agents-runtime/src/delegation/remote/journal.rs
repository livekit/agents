use std::sync::Arc;

use lk_agents_abi::BoxFuture;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::delegation::remote::*;
use lk_agents_abi::session::ToolCallRef;
use lk_agents_core::a2a::client::{ClientError, Request};
use lk_agents_net::a2a_client::observer::RequestObserver;

/// Each transport exchange reports into the conversation's serialized command port.
pub struct Journal {
    pub owner: Arc<dyn RemoteCommands>,
    pub endpoint: String,
    pub remote_id: String,
    pub request_id: String,
    pub origin: Option<ToolCallRef>,
}

impl Journal {
    async fn apply(&self, input: RemoteInput) -> Result<(), ClientError> {
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
            self.apply(RemoteInput::Prepare {
                request_id: self.request_id.clone(),
                request: body,
                streaming: request.streaming,
                origin: self.origin.clone(),
                input_request: None,
            })
            .await
        })
    }

    fn dispatch_started(&self) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(self.apply(RemoteInput::DispatchStarted { request_id: self.request_id.clone() }))
    }

    fn response<'a>(
        &'a self,
        request: &'a Request,
        response: a2a::StreamResponse,
    ) -> BoxFuture<'a, Result<(), ClientError>> {
        Box::pin(async move {
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
            self.apply(RemoteInput::Response {
                request_id: self.request_id.clone(),
                response: Box::new(response),
            })
            .await
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
            self.apply(input).await
        })
    }
}

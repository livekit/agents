//! The model: every `chat()` registers an invocation with the actual request's item ids and
//! tool names; the response stream reads whatever the test delivers to that request.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use lk_agents_abi::{
    ChatRequest, Llm, LlmChunk, LlmStream, Ts,
    errors::ProviderError,
    utils::stream::{Closed, StreamSource},
};
use tokio::sync::mpsc;

use super::{Control, Registry};
use crate::{
    input::{AgentText, Input},
    record::{Detail, Kind},
};

/// The id this fake puts on every chunk of a request's response. The driver copies it into the
/// metrics it reports for that request, which is how the harness learns, from the driver itself,
/// which speech a request was made for.
const RESPONSE: &str = "response-";

/// The chat a response id names, if this fake minted it.
pub fn request_of(response_id: &str) -> Option<u32> {
    response_id.strip_prefix(RESPONSE)?.parse().ok()
}

/// The response id for the n-th chat.
pub(super) fn response_id(nth: u32) -> String {
    format!("{RESPONSE}{nth}")
}

pub struct FakeLlm {
    registry: Arc<Registry>,
}

impl FakeLlm {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeLlm { registry })
    }
}

struct Response {
    slot: super::Slot,
    receiver: mpsc::UnboundedReceiver<Result<LlmChunk, ProviderError>>,
    registry: Arc<Registry>,
    closed: AtomicBool,
}

#[async_trait::async_trait]
impl Llm for FakeLlm {
    fn label(&self) -> String {
        "fake::llm".into()
    }

    async fn chat(&self, request: &ChatRequest) -> Result<LlmStream, ProviderError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let slot = self.registry.register(
            Kind::Chat,
            Detail::Chat {
                ctx_ids: request.chat_ctx.items.iter().map(|i| i.id().to_string()).collect(),
                speech: None,
                tools: request.tools.iter().map(|t| t.name.clone()).collect(),
            },
            Control::Chat(sender),
        );
        Ok(Box::new(Response {
            slot,
            receiver,
            registry: self.registry.clone(),
            closed: AtomicBool::new(false),
        }))
    }
}

#[async_trait::async_trait]
impl StreamSource<LlmChunk> for Response {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        self.receiver.recv().await.transpose()
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        self.closed.store(true, Ordering::SeqCst);
        self.registry.mark_closed(self.slot);
        Closed::Clean
    }
}

/// One model delivery onto a registered chat request's stream. True when the stream still had
/// a reader. `End` and `Fail` close the stream; the registry drops the control afterwards.
pub(super) fn deliver(
    sender: &mpsc::UnboundedSender<Result<LlmChunk, ProviderError>>,
    nth: u32,
    what: &Input,
    serial: u32,
) -> bool {
    let response_id = response_id(nth);
    match what {
        Input::LlmText { text: AgentText(text), .. } => sender
            .send(Ok(LlmChunk {
                id: response_id,
                delta: Some(lk_agents_abi::ChoiceDelta {
                    content: Some(text.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .is_ok(),
        Input::LlmToolCall { name, args, .. } => sender
            .send(Ok(LlmChunk {
                id: response_id,
                delta: Some(lk_agents_abi::ChoiceDelta {
                    tool_calls: vec![lk_agents_abi::FunctionToolCall {
                        name: name.clone(),
                        arguments: args.0.clone(),
                        call_id: format!("call-{nth}-{serial}"),
                        extra: None,
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .is_ok(),
        Input::LlmEnd { .. } => !sender.is_closed(),
        Input::LlmFail { .. } => sender
            .send(Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                "injected provider failure",
            ))))
            .is_ok(),
        _ => false,
    }
}

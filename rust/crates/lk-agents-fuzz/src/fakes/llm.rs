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

impl Llm for FakeLlm {
    fn label(&self) -> String {
        "fake::llm".into()
    }

    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
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
            }) as LlmStream)
        })
    }
}

impl StreamSource<LlmChunk> for Response {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { self.receiver.recv().await.transpose() })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.closed.store(true, Ordering::SeqCst);
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

/// The completion tokens every reply reports.
pub const COMPLETION_TOKENS: u32 = 10;

/// One model delivery onto a registered chat request's stream, in the chunk shapes the inference
/// gateway sends: each `LlmText` is one wire chunk naming the assistant role
/// (the case cuts a reply into the token-sized pieces the model streams, at the model's pace), a
/// tool call comes whole in one chunk, and the end is an empty chunk (the finish) followed by one
/// carrying only the usage. True when the stream still had a reader. `End` and `Fail` close the
/// stream; the registry drops the control afterwards.
pub(super) fn deliver(
    sender: &mpsc::UnboundedSender<Result<LlmChunk, ProviderError>>,
    nth: u32,
    what: &Input,
    serial: u32,
) -> bool {
    let response_id = response_id(nth);
    let assistant = || lk_agents_abi::ChoiceDelta {
        role: Some(lk_agents_abi::Role::Assistant),
        ..Default::default()
    };
    match what {
        Input::LlmText { text: AgentText(text), .. } => sender
            .send(Ok(LlmChunk {
                id: response_id,
                delta: Some(lk_agents_abi::ChoiceDelta {
                    content: Some(text.clone()),
                    ..assistant()
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
                    ..assistant()
                }),
                ..Default::default()
            }))
            .is_ok(),
        // The finish (an empty delta), then the usage alone, as the gateway ends a stream.
        Input::LlmEnd { .. } => {
            let finish = LlmChunk {
                id: response_id.clone(),
                delta: Some(lk_agents_abi::ChoiceDelta {
                    content: Some(String::new()),
                    ..assistant()
                }),
                ..Default::default()
            };
            let usage = LlmChunk {
                id: response_id,
                usage: Some(lk_agents_abi::CompletionUsage {
                    completion_tokens: COMPLETION_TOKENS,
                    ..Default::default()
                }),
                ..Default::default()
            };
            sender.send(Ok(finish)).is_ok() && sender.send(Ok(usage)).is_ok()
        }
        Input::LlmFail { .. } => sender
            .send(Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                "injected provider failure",
            ))))
            .is_ok(),
        _ => false,
    }
}

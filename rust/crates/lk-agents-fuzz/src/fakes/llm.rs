use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use lk_agents_abi::{
    ChatRequest, Llm, LlmChunk, LlmStream, Ts,
    errors::ProviderError,
    utils::stream::{Closed, StreamOutput},
};
use tokio::sync::mpsc;

use super::{
    Control, Registry,
    hazard::{Site, strike},
};
use crate::{
    input::{AgentText, Input},
    record::{Detail, Kind},
};

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
    hazard: super::hazard::StreamHazard,
}

impl Llm for FakeLlm {
    fn label(&self) -> String {
        "fake::llm".into()
    }

    fn failed(&self, _error: ProviderError, recoverable: bool) {
        self.registry.record_failed(Kind::Chat, recoverable);
    }

    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        request: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            if let Some(fault) = self.registry.hazard(Site::ChatOpen) {
                return strike(fault, Site::ChatOpen).await;
            }
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
                hazard: self.registry.stream_hazard(Site::ChatNext, Site::ChatClose),
            }) as LlmStream)
        })
    }
}

impl StreamOutput<LlmChunk> for Response {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::ChatNext, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return strike(fault, Site::ChatNext).await;
            }
            self.receiver.recv().await.transpose()
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::ChatClose, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return self.registry.strike_close(fault, Site::ChatClose, deadline).await;
            }
            self.closed.store(true, Ordering::SeqCst);
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Response {}

/// The completion tokens every reply reports.
pub const COMPLETION_TOKENS: u64 = 10;

/// One model delivery onto a chat stream, in the gateway's chunk shapes: one chunk per `LlmText`,
/// a tool call whole, and the end as an empty finish chunk then a usage-only chunk. True when the
/// stream still had a reader.
pub(super) fn deliver(
    sender: &mpsc::UnboundedSender<Result<LlmChunk, ProviderError>>,
    nth: u32,
    response_id: String,
    what: &Input,
    serial: u32,
) -> bool {
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
                usage: Some(Box::new(lk_agents_abi::usage::TokenUsage {
                    output_tokens: COMPLETION_TOKENS,
                    ..Default::default()
                })),
                ..Default::default()
            };
            sender.send(Ok(finish)).is_ok() && sender.send(Ok(usage)).is_ok()
        }
        Input::LlmFail { .. } => sender
            .send(Err(ProviderError::Local(lk_agents_abi::errors::ApiError::new(
                lk_agents_abi::errors::ErrorKind::Server,
                "injected provider failure",
            ))))
            .is_ok(),
        _ => false,
    }
}

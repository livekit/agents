use super::*;
use lk_agents_abi::a2a::turn::HandlerPort;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{BoxFuture, ChatMessage};
use lk_agents_net::task::ScopedTask;

#[derive(Clone)]
pub(crate) enum Target {
    Handler(Arc<dyn HandlerPort>),
    Delegate {
        host: Arc<dyn lk_agents_abi::a2a::remote::DelegationHost>,
        cx: telemetry::Cx,
        context: lk_agents_abi::invoke::CallContext,
    },
    Stream(crate::run_stream::Output),
}
impl Target {
    pub(super) fn is_stream(&self) -> bool {
        matches!(self, Self::Stream(_))
    }
    /// Registers receipt ownership synchronously before the following text event can be read.
    fn prepare(&self, message: ChatMessage) -> BoxFuture<'static, Result<(), ProviderError>> {
        match self {
            Self::Handler(port) => {
                let receipt = port.say(message);
                Box::pin(async move { receipt.await.map_err(Into::into) })
            }
            Self::Stream(output) => output.prepare(message),
            Self::Delegate { host, cx, context } => {
                let (host, cx, context) = (host.clone(), cx.clone(), context.clone());
                let text = message.raw_text_content().unwrap_or_default();
                Box::pin(async move {
                    host.say(&cx, &context, text, Default::default()).await.map_err(Into::into)
                })
            }
        }
    }
    pub(crate) async fn progress(&self, text: String, silent: bool) {
        if let Self::Handler(port) = self {
            port.progress_with_policy(text, !silent).await;
        }
    }
}

pub(super) enum Pending {
    Admitted(super::progress::Receipt),
    Delivering {
        receipt: super::progress::Receipt,
        result: signal::Cell<Option<Result<(), ProviderError>>>,
        task: Option<ScopedTask>,
    },
    Delivered {
        receipt: super::progress::Receipt,
        _task: Option<ScopedTask>,
    },
}
fn ended() -> ProviderError {
    HostError::new("the exact output's delivery owner has ended").into()
}

impl Execution {
    pub(super) fn deliver_output(
        &mut self,
        output: lk_agents_core::agent_run::ToolOutputId,
        message: ChatMessage,
    ) {
        let Some(Pending::Admitted(receipt)) = self.tool_outputs.remove(&output) else {
            return;
        };
        let future = match &self.output {
            Some(target) => target.prepare(message),
            None => Box::pin(async {
                Err(HostError::new(
                    "exact tool output requires a live voice, text stream or A2A output",
                )
                .into())
            }),
        };
        let (done, result) = signal::once();
        let task = self.progress.spawn_output(done, future);
        // A refused spawn drops its producer. The same death input clears core ownership;
        // it cannot leave a pending source checkpoint merely because no task was admitted.
        self.tool_outputs.insert(output, Pending::Delivering { receipt, result, task });
    }
    pub(super) fn settle_output(
        &mut self,
        output: lk_agents_core::agent_run::ToolOutputId,
        result: Result<(), ProviderError>,
    ) {
        let Some(pending) = self.tool_outputs.remove(&output) else {
            return;
        };
        let receipt = match pending {
            Pending::Admitted(receipt)
            | Pending::Delivering { receipt, .. }
            | Pending::Delivered { receipt, .. } => receipt,
        };
        receipt.finish(result);
    }
    /// Both the host invoker and exact subwork report death to their existing driver.
    pub(crate) async fn settled(&mut self) -> Option<AgentRunEvent> {
        tokio::select! {
            biased;
            (output, result) = delivered(&self.tool_outputs) => {
                let pending = self.tool_outputs.remove(&output).expect("selected owned output");
                let Pending::Delivering { receipt, task, .. } = pending else { unreachable!("selected delivering output") };
                self.tool_outputs.insert(output, Pending::Delivered { receipt, _task: task });
                Some(AgentRunEvent::ToolOutputDelivered { output, result })
            }
            _ = self.invoker.settled() => None,
        }
    }
}
async fn delivered(
    outputs: &BTreeMap<lk_agents_core::agent_run::ToolOutputId, Pending>,
) -> (lk_agents_core::agent_run::ToolOutputId, Result<(), ProviderError>) {
    let futures: Vec<_> = outputs
        .iter()
        .filter_map(|(id, pending)| {
            let Pending::Delivering { result, .. } = pending else {
                return None;
            };
            Some(Box::pin(
                async move { (*id, result.wait().await.unwrap_or_else(|_| Err(ended()))) },
            ))
        })
        .collect();
    if futures.is_empty() {
        std::future::pending().await
    } else {
        futures_util::future::select_all(futures).await.0
    }
}

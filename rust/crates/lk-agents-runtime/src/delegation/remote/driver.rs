//! Live task messages have one consumer: the original delegation's native driver.

use std::sync::Arc;

use lk_agents_abi::BoxFuture;
use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::{DelegationInput, RemoteResponseClaim, RemoteTaskDriver};
use lk_agents_abi::session::ToolCallRef;
use lk_agents_net::signal;
use parking_lot::Mutex;
use tokio::sync::mpsc;

pub(super) struct Driver {
    sender: mpsc::Sender<Command>,
}

pub(super) struct Command {
    pub message: Message,
    pub origin: Option<ToolCallRef>,
    pub done: Completion,
    pub response_claim: Option<Arc<RemoteResponseClaim>>,
}

pub(super) enum Message {
    Input(DelegationInput),
    Cancel(lk_agents_abi::delegation::remote::RemoteTaskRef),
    Reconcile,
    Resume {
        request_id: String,
        request: Box<lk_agents_abi::a2a::protocol::SendMessageRequest>,
        streaming: bool,
        input_request: Option<lk_agents_abi::session::InputRequestRef>,
    },
}

type Outcome = Arc<Mutex<Option<Result<(), AgentClientError>>>>;

/// Its producer moves with the command. Dropping the driver or an in-flight command ends
/// the wait, including when the frontend that submitted the message has already gone away.
pub(super) struct Completion {
    outcome: Outcome,
    done: signal::Producer<Option<()>>,
}

impl Completion {
    pub fn resolve(self, outcome: Result<(), AgentClientError>) {
        *self.outcome.lock() = Some(outcome);
        self.done.resolve(());
    }
}

impl Driver {
    pub fn channel() -> (Arc<Self>, mpsc::Receiver<Command>) {
        let (sender, receiver) = mpsc::channel(16);
        (Arc::new(Self { sender }), receiver)
    }

    async fn submit(
        &self,
        message: Message,
        response_claim: Option<Arc<RemoteResponseClaim>>,
        origin: Option<ToolCallRef>,
    ) -> Result<(), AgentClientError> {
        let outcome = Arc::new(Mutex::new(None));
        let (done, receipt) = signal::once();
        self.sender
            .send(Command {
                message,
                origin,
                done: Completion { outcome: outcome.clone(), done },
                response_claim,
            })
            .await
            .map_err(|_| ended())?;
        receipt.wait().await.map_err(|_| ended())?;
        outcome.lock().take().expect("only the submitting waiter takes a settled outcome")
    }
}

impl RemoteTaskDriver for Driver {
    fn is_live(&self) -> bool {
        !self.sender.is_closed()
    }

    fn input<'a>(
        &'a self,
        input: DelegationInput,
        response_claim: Option<Arc<RemoteResponseClaim>>,
        origin: Option<ToolCallRef>,
    ) -> BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(self.submit(Message::Input(input), response_claim, origin))
    }

    fn resume<'a>(
        &'a self,
        request_id: String,
        request: Box<lk_agents_abi::a2a::protocol::SendMessageRequest>,
        streaming: bool,
        input_request: Option<lk_agents_abi::session::InputRequestRef>,
        origin: Option<ToolCallRef>,
    ) -> BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(self.submit(
            Message::Resume { request_id, request, streaming, input_request },
            None,
            origin,
        ))
    }

    fn cancel<'a>(
        &'a self,
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
    ) -> BoxFuture<'a, Result<(), AgentClientError>> {
        Box::pin(self.submit(Message::Cancel(task), None, None))
    }

    fn reconcile(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            // This command observes work already admitted by a duplicate exchange. Its
            // failure belongs to the original call, never a retry of that accepted work.
            match self.submit(Message::Reconcile, None, None).await {
                Ok(()) | Err(_) => {}
            }
        })
    }
}

fn ended() -> AgentClientError {
    AgentClientError::InvalidInput(
        "the remote task's live observer ended before this command settled; inspect its saved task before continuing".into(),
    )
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;

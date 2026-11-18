//! Live task messages have one consumer: the original delegation's native driver.

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use lk_agents_abi::BoxFuture;
use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::{DelegationInput, RemoteResponseClaim, RemoteTaskDriver};
use lk_agents_abi::session::ToolCallRef;
use lk_agents_abi::session::control::RemoteCancellation;
use lk_agents_net::owned::Background;
use lk_agents_net::signal;
use parking_lot::Mutex;
use tokio::sync::mpsc;

pub(super) struct Driver {
    sender: mpsc::Sender<Command>,
    configuration: super::configuration::Configuration,
    enqueues: Background,
    cancellations: Mutex<
        BTreeMap<lk_agents_abi::delegation::remote::RemoteTaskRef, Weak<dyn RemoteCancellation>>,
    >,
}

pub(super) struct Command {
    pub message: Message,
    pub origin: Option<ToolCallRef>,
    pub done: Completion,
    pub response_claim: Option<Arc<RemoteResponseClaim>>,
}

pub(super) enum Message {
    Input(DelegationInput),
    Cancel {
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
        retry_after: Option<String>,
    },
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
pub(super) enum Completion {
    Submission {
        outcome: Outcome,
        done: signal::Producer<Option<()>>,
    },
    Cancellation {
        done: signal::Producer<Option<Result<(), AgentClientError>>>,
        _receipt: Arc<dyn RemoteCancellation>,
    },
}

impl Completion {
    pub fn resolve(self, outcome: Result<(), AgentClientError>) {
        match self {
            Self::Submission { outcome: stored, done } => {
                *stored.lock() = Some(outcome);
                done.resolve(());
            }
            Self::Cancellation { done, .. } => done.resolve(outcome),
        }
    }
}

impl Driver {
    pub fn channel(
        background: &Background,
        configuration: super::configuration::Configuration,
    ) -> (Arc<Self>, mpsc::Receiver<Command>) {
        let (sender, receiver) = mpsc::channel(16);
        let enqueues =
            Background::new(Arc::default(), background.spawner.clone(), background.clock.clone());
        (
            Arc::new(Self { sender, configuration, enqueues, cancellations: Mutex::default() }),
            receiver,
        )
    }

    pub async fn close(&self) {
        self.enqueues.tasks.close();
        let late = self.enqueues.tasks.drain(&*self.enqueues.clock, self.enqueues.deadline()).await;
        if !late.is_empty() {
            log::warn!(target: lk_agents_abi::LOG_TARGET,
                "remote task control enqueue cleanup incomplete: {late:?}");
        }
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
                done: Completion::Submission { outcome: outcome.clone(), done },
                response_claim,
            })
            .await
            .map_err(|_| ended())?;
        receipt.wait().await.map_err(|_| ended())?;
        outcome.lock().take().expect("only the submitting waiter takes a settled outcome")
    }
}

impl RemoteTaskDriver for Driver {
    fn same_configuration(&self, other: &dyn RemoteTaskDriver) -> bool {
        (other as &dyn std::any::Any)
            .downcast_ref::<Self>()
            .is_some_and(|other| self.configuration == other.configuration)
    }

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

    fn cancellation(
        &self,
        task: lk_agents_abi::delegation::remote::RemoteTaskRef,
        retry_after: Option<String>,
    ) -> Arc<dyn RemoteCancellation> {
        let (done, candidate) = super::control::Receipt::pair();
        {
            let mut inspected = Vec::new();
            let mut pending = self.cancellations.lock();
            pending.retain(|_, receipt| {
                receipt.upgrade().is_some_and(|receipt| {
                    let active = receipt.pending();
                    inspected.push(receipt);
                    active
                })
            });
            if let Some(previous) = pending.get(&task).and_then(Weak::upgrade) {
                return previous;
            }
            pending.insert(task.clone(), Arc::downgrade(&candidate));
            drop(pending);
            drop(inspected);
        }
        let sender = self.sender.clone();
        // Admission owns enqueue progress before it releases its receipt. A lost waiter
        // cannot retract the intent, and closing the consumer settles a capacity wait.
        let queued_receipt = candidate.clone();
        self.enqueues.spawn("remote task cancellation enqueue", async move {
            let command = Command {
                message: Message::Cancel { task, retry_after },
                origin: None,
                done: Completion::Cancellation { done, _receipt: queued_receipt },
                response_claim: None,
            };
            if let Err(error) = sender.send(command).await {
                error.0.done.resolve(Err(ended()));
            }
        });
        candidate
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

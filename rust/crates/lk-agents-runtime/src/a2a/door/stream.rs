//! Publishes a request's live response events; slow or disconnected observers are detached.

use a2a::{JsonRpcId, StreamResponse};
use lk_agents_abi::a2a::door::{DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::TurnEvent;
use lk_agents_core::a2a::TurnResult;
use lk_agents_core::a2a::wire::{self, Delivery, Publication};
use lk_agents_storage::thread::schema::TaskRow;
use tokio::sync::{Mutex, mpsc};

use lk_agents_net::clock::unix_now_ms;
use lk_agents_storage::shortuuid;

pub type Sender = mpsc::Sender<Result<DoorFrame, DoorError>>;

pub struct Publisher {
    request_id: JsonRpcId,
    context: String,
    /// The task a continuation names, announced in place of its receipt.
    continuing: Option<String>,
    state: Mutex<(Publication, Option<Sender>)>,
}

impl Publisher {
    pub fn new(
        request_id: JsonRpcId,
        context: String,
        continuing: Option<String>,
        sender: Option<Sender>,
    ) -> Self {
        let publication = Publication::new(shortuuid("art_"));
        Self { request_id, context, continuing, state: Mutex::new((publication, sender)) }
    }

    async fn emit(&self, events: Vec<StreamResponse>, task_id: &str) {
        for event in events {
            let frame = wire::success(self.request_id.clone(), event, None);
            let mut state = self.state.lock().await;
            let Some(sender) = state.1.as_ref() else { continue };
            let attached = match wire::deliver(self.request_id.clone(), frame, sender.capacity()) {
                Delivery::Send(frame) => sender.try_send(Ok(frame)).is_ok(),
                Delivery::Detach(failure) => {
                    let _ = sender.try_send(Ok(failure));
                    log::warn!(target: "livekit.agents", taskId = task_id; "detached a slow A2A observer");
                    false
                }
            };
            if !attached {
                state.1 = None;
            }
        }
    }

    /// The turn's answer and final status.
    pub async fn finish(&self, result: &TurnResult) {
        let events = self.state.lock().await.0.finish(result, &self.context, unix_now_ms());
        self.emit(events, &result.task.task_id).await;
    }

    /// The message's task is durable, before admission waits or application work.
    pub async fn accepted(&self, task: &TaskRow) {
        let task_id = self.continuing.as_deref().unwrap_or(&task.task_id);
        let announced = self.state.lock().await.0.announce(task_id, &self.context, unix_now_ms());
        self.emit(announced.into_iter().collect(), task_id).await;
    }

    /// A handler's progress or answer token.
    pub async fn event(&self, event: &TurnEvent) {
        let (TurnEvent::Progress { task_id, .. } | TurnEvent::Token { task_id, .. }) = event;
        let published =
            self.state.lock().await.0.event(event, &self.context, shortuuid("msg_"), unix_now_ms());
        self.emit(published.into_iter().collect(), task_id).await;
    }
}

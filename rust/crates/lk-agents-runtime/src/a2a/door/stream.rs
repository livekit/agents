//! A request's response stream: each event `wire::Publication` decides is appended to the durable
//! stream, then sent to the live observer, which is detached once it falls behind.

use std::sync::Arc;

use a2a::{JsonRpcId, StreamResponse};
use lk_agents_abi::a2a::door::{DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::TurnEvent;
use lk_agents_abi::storage::StreamInfo;
use lk_agents_core::a2a::TurnResult;
use lk_agents_core::a2a::wire::{self, Cursor, Delivery, Publication};
use lk_agents_storage::StorageError;
use lk_agents_storage::thread::schema::TaskRow;
use tokio::sync::{Mutex, mpsc};

use lk_agents_net::clock::unix_now_ms;
use lk_agents_storage::{ResponseStreams, shortuuid};

pub type Sender = mpsc::Sender<Result<DoorFrame, DoorError>>;

pub struct Publisher {
    streams: Arc<dyn ResponseStreams>,
    pub stream: StreamInfo,
    request_id: JsonRpcId,
    context: String,
    /// The task a continuation names, announced in place of its receipt.
    continuing: Option<String>,
    state: Mutex<(Publication, Option<Sender>)>,
}

impl Publisher {
    pub fn new(
        streams: Arc<dyn ResponseStreams>,
        stream: StreamInfo,
        request_id: JsonRpcId,
        context: String,
        continuing: Option<String>,
        sender: Option<Sender>,
    ) -> Self {
        let publication = Publication::new(shortuuid("art_"));
        Self {
            streams,
            stream,
            request_id,
            context,
            continuing,
            state: Mutex::new((publication, sender)),
        }
    }

    async fn emit(&self, events: Vec<StreamResponse>, task_id: &str) -> Result<(), StorageError> {
        for event in events {
            let bytes = serde_json::to_vec(&event).expect("A2A event");
            let stream_id = &self.stream.stream_id;
            let offset = self.streams.append(stream_id, &bytes).await?;
            let cursor = Cursor { task_id: task_id.into(), stream_id: stream_id.clone(), offset };
            let frame = wire::success(self.request_id.clone(), event, Some(&cursor.encode()));
            let mut state = self.state.lock().await;
            let Some(sender) = state.1.as_ref() else { continue };
            let attached = match wire::deliver(self.request_id.clone(), frame, sender.capacity()) {
                Delivery::Send(frame) => sender.try_send(Ok(frame)).is_ok(),
                Delivery::Detach(failure) => {
                    let _ = sender.try_send(Ok(failure));
                    log::warn!(target: "livekit.agents", taskId = task_id; "detached a slow A2A observer; the response remains durable");
                    false
                }
            };
            if !attached {
                state.1 = None;
            }
        }
        Ok(())
    }

    /// The turn's answer, then the stream closes.
    pub async fn finish(&self, result: &TurnResult) -> Result<(), StorageError> {
        let events = self.state.lock().await.0.finish(result, &self.context, unix_now_ms());
        self.emit(events, &result.task.task_id).await?;
        self.streams.close(&self.stream.stream_id).await
    }

    /// The message's task is durable, before admission waits or application work.
    pub async fn accepted(&self, task: &TaskRow) -> Result<(), StorageError> {
        let task_id = self.continuing.as_deref().unwrap_or(&task.task_id);
        let announced = self.state.lock().await.0.announce(task_id, &self.context, unix_now_ms());
        self.emit(announced.into_iter().collect(), task_id).await
    }

    /// A handler's progress or answer token.
    pub async fn event(&self, event: &TurnEvent) -> Result<(), StorageError> {
        let (TurnEvent::Progress { task_id, .. } | TurnEvent::Token { task_id, .. }) = event else {
            return Ok(());
        };
        let published =
            self.state.lock().await.0.event(event, &self.context, shortuuid("msg_"), unix_now_ms());
        self.emit(published.into_iter().collect(), task_id).await
    }
}

//! A turn's durable event publication and the typed host stream that supplies its progress.

use std::sync::Arc;

use a2a::{JsonRpcId, StreamResponse};
use async_trait::async_trait;
use lk_agents_abi::a2a::door::{DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::{TurnContext, TurnEvent, TurnHost, Verdict};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, StreamSink, StreamSource, Ts};
use lk_agents_storage::StorageError;
use tokio::sync::{Mutex, mpsc};

use crate::storage::{ResponseStreams, now_ms, shortuuid};
use lk_agents_core::a2a::wire::{self, Cursor};

pub struct Publisher {
    streams: Arc<dyn ResponseStreams>,
    stream_id: String,
    request_id: JsonRpcId,
    state: Mutex<State>,
}

struct State {
    sender: Option<mpsc::Sender<Result<DoorFrame, DoorError>>>,
    acknowledged: bool,
    task_id: String,
    artifact_id: String,
    text: String,
}

impl Publisher {
    pub fn new(
        streams: Arc<dyn ResponseStreams>,
        stream_id: String,
        request_id: JsonRpcId,
        sender: Option<mpsc::Sender<Result<DoorFrame, DoorError>>>,
    ) -> Self {
        Self {
            streams,
            stream_id,
            request_id,
            state: Mutex::new(State {
                sender,
                acknowledged: false,
                task_id: String::new(),
                artifact_id: shortuuid("art_"),
                text: String::new(),
            }),
        }
    }

    async fn emit(&self, event: StreamResponse, task_id: &str) -> Result<(), StorageError> {
        let bytes = serde_json::to_vec(&event).expect("A2A event");
        let offset = self.streams.append(&self.stream_id, &bytes).await?;
        let cursor =
            Cursor { task_id: task_id.into(), stream_id: self.stream_id.clone(), offset }.encode();
        let frame = wire::success(self.request_id.clone(), event, Some(&cursor));
        let mut state = self.state.lock().await;
        if let Some(sender) = state.sender.as_ref() {
            if sender.capacity() <= 1 {
                let error = lk_agents_core::a2a::rpc::rpc_error(
                    a2a::error_code::INTERNAL_ERROR,
                    "slow subscriber detached; resume using the last event cursor",
                );
                let _ = sender.try_send(Ok(wire::frame(
                    lk_agents_core::a2a::rpc::failure(self.request_id.clone(), error),
                    None,
                )));
                state.sender = None;
                log::warn!(target: "livekit.agents", taskId = task_id; "detached a slow A2A observer; the response remains durable");
            } else if sender.try_send(Ok(frame)).is_err() {
                state.sender = None;
            }
        }
        Ok(())
    }

    async fn acknowledge(&self, task_id: &str, context: &str) -> Result<(), StorageError> {
        {
            let mut state = self.state.lock().await;
            if state.acknowledged {
                return Ok(());
            }
            state.acknowledged = true;
            state.task_id = task_id.into();
        }
        self.emit(wire::submitted(task_id, context, now_ms()), task_id).await
    }

    async fn event(&self, event: &TurnEvent, context: &str) -> Result<(), StorageError> {
        match event {
            TurnEvent::Progress { task_id, text } => {
                if self.state.lock().await.task_id == *task_id {
                    let progress =
                        wire::progress(task_id, context, text.clone(), shortuuid("msg_"), now_ms());
                    self.emit(progress, task_id).await?;
                }
            }
            TurnEvent::Token { task_id, delta } => {
                let (artifact_id, append) = {
                    let mut state = self.state.lock().await;
                    if state.task_id != *task_id || delta.is_empty() {
                        return Ok(());
                    }
                    let append = !state.text.is_empty();
                    state.text.push_str(delta);
                    (state.artifact_id.clone(), append)
                };
                self.emit(
                    wire::artifact(
                        task_id,
                        context,
                        &artifact_id,
                        Some(delta.clone()),
                        append,
                        false,
                    ),
                    task_id,
                )
                .await?;
            }
            _ => {}
        }
        Ok(())
    }

    pub async fn finish(
        &self,
        result: &lk_agents_core::a2a::TurnResult,
        context: &str,
    ) -> Result<(), StorageError> {
        let row = &result.task;
        self.acknowledge(&row.task_id, context).await?;
        if let Some(artifact) = row.data.artifacts.as_ref().and_then(|artifacts| artifacts.first())
        {
            let (artifact_id, streamed) = {
                let state = self.state.lock().await;
                (state.artifact_id.clone(), state.text.clone())
            };
            let event = if result.output.as_ref().and_then(serde_json::Value::as_str)
                == Some(streamed.as_str())
                && !streamed.is_empty()
            {
                wire::artifact(&row.task_id, context, &artifact_id, None, true, true)
            } else {
                StreamResponse::ArtifactUpdate(a2a::TaskArtifactUpdateEvent {
                    task_id: row.task_id.clone(),
                    context_id: context.into(),
                    artifact: artifact.clone(),
                    append: Some(false),
                    last_chunk: Some(true),
                    metadata: None,
                })
            };
            self.emit(event, &row.task_id).await?;
        }
        self.emit(wire::final_status(row, context), &row.task_id).await?;
        self.streams.close(&self.stream_id).await
    }
}

pub struct Accepted {
    pub task_id: Option<String>,
    pub publisher: Arc<Publisher>,
    pub context: String,
}

#[async_trait]
impl crate::a2a::Acceptance for Accepted {
    async fn accepted(
        &self,
        task: &lk_agents_storage::schema::TaskRow,
    ) -> Result<(), StorageError> {
        self.publisher
            .acknowledge(self.task_id.as_deref().unwrap_or(&task.task_id), &self.context)
            .await
    }
}

pub struct WireHost {
    host: Arc<dyn TurnHost>,
    publisher: Arc<Publisher>,
    context: String,
}

impl WireHost {
    pub fn new(host: Arc<dyn TurnHost>, publisher: Arc<Publisher>, context: String) -> Self {
        Self { host, publisher, context }
    }
}

#[async_trait]
impl TurnHost for WireHost {
    async fn handle(
        &self,
        ctx: &TurnContext,
    ) -> Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>
    {
        self.publisher.acknowledge(&ctx.task_id, &self.context).await.map_err(provider_error)?;
        let (sink, source) = self.host.handle(ctx).await?;
        Ok((
            sink,
            Box::new(Events {
                source,
                publisher: self.publisher.clone(),
                context: self.context.clone(),
            }),
        ))
    }
}

struct Events {
    source: Box<dyn StreamSource<TurnEvent>>,
    publisher: Arc<Publisher>,
    context: String,
}

impl StreamSource<TurnEvent> for Events {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
        Box::pin(async move {
            let event = self.source.next().await?;
            if let Some(event) = &event {
                self.publisher.event(event, &self.context).await.map_err(provider_error)?;
            }
            Ok(event)
        })
    }
    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }
    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { self.source.close(deadline).await })
    }
}

fn provider_error(error: StorageError) -> ProviderError {
    ProviderError::Host {
        error: lk_agents_abi::errors::HostError {
            kind: "StorageUnavailable".into(),
            message: error.to_string(),
            host_ref: None,
        },
        api_kind: None,
        retryable: true,
        status_code: None,
    }
}

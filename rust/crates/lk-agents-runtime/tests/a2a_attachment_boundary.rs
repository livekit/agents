//! Attachments cross the managed turn as application references, never inline upload bytes.
use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_a2a::{Policy, PolicySpec, TurnOptions};
use lk_agents_abi::a2a::client::MessagePart;
use lk_agents_abi::a2a::turn::{HandlerOutput, TurnContext, TurnEvent, TurnHost, Verdict};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, RealClock, StreamSink, StreamSource, Ts};
use lk_agents_runtime::a2a::Turn;
use lk_agents_runtime::storage::sqlite::SqliteStorage;
use lk_agents_runtime::storage::{Storage, shortuuid};
use lk_agents_storage::StorageError;
use serde_json::json;

struct Host;
struct Sink;
struct Events(Option<TurnEvent>);

impl StreamSink<Verdict> for Sink {
    fn push(&self, _: Verdict) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

#[async_trait]
impl StreamSource<TurnEvent> for Events {
    async fn next(&mut self) -> Result<Option<TurnEvent>, ProviderError> {
        Ok(self.0.take())
    }
    async fn close(self: Box<Self>, _: Ts) -> Closed {
        Closed::Clean
    }
}

#[async_trait]
impl TurnHost for Host {
    async fn handle(
        &self,
        ctx: TurnContext,
    ) -> Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>
    {
        let parts = &ctx.batch[0].parts;
        assert!(matches!(&parts[0], MessagePart::TextPart { text, .. } if text == "review"));
        for (part, expected) in
            parts[1..].iter().zip(["private photo reference", "private document reference"])
        {
            assert!(
                matches!(part, MessagePart::FilePart { url: Some(url), data: None, .. } if url == expected)
            );
        }
        let event = TurnEvent::Output {
            output: HandlerOutput {
                output: Some(json!("received references")),
                ..Default::default()
            },
        };
        Ok((Box::new(Sink), Box::new(Events(Some(event)))))
    }
}

#[tokio::test]
async fn sqlite_restart_keeps_references_but_no_inline_upload_in_task_or_inbox() {
    let dir = std::env::temp_dir().join(shortuuid("lk-attachment-boundary-"));
    let storage = SqliteStorage::open(&dir, Default::default()).await.unwrap();
    let created = storage.create_thread(Default::default()).await.unwrap();
    let thread = storage.open(&created.thread_id, "first-worker").await.unwrap();
    let turn = Turn::new(
        thread.clone(),
        "desk",
        Arc::new(Host),
        Arc::new(RealClock::new()),
        TurnOptions { policy: PolicySpec::All(Policy::Queue), ..Default::default() },
    );
    for mime in ["image/png", "audio/wav", "application/pdf"] {
        let result = turn.run(json!({"messageId": "inline", "role": "ROLE_USER", "parts": [{"raw": "dXBsb2FkZWQgYnl0ZXM=", "mediaType": mime}]}), None).await;
        assert!(matches!(result, Err(StorageError::Invalid(_))));
    }
    assert!(thread.lane("desk").db().tasks(None, 10).await.unwrap().is_empty());
    assert!(thread.lane("desk").db().inbox(false).await.unwrap().is_empty());
    let parts = json!([
        {"text": "review"},
        {"url": "private photo reference", "mediaType": "image/png", "filename": "photo.png"},
        {"url": "private document reference", "mediaType": "application/pdf", "filename": "receipt.pdf"},
    ]);
    let result = turn
        .run(json!({"messageId": "references", "role": "ROLE_USER", "parts": parts}), None)
        .await
        .unwrap();
    let task_id = result.task.task_id;
    drop(turn);
    thread.close().await;
    storage.close().await;
    drop(storage);

    let reopened = SqliteStorage::open(&dir, Default::default()).await.unwrap();
    let thread = reopened.open(&created.thread_id, "second-worker").await.unwrap();
    let task = thread.task(&task_id).await.unwrap().unwrap();
    assert_eq!(serde_json::to_value(&task.data.history[0]).unwrap()["parts"], parts);
    let inbox = thread.lane("desk").db().inbox(false).await.unwrap();
    assert!(inbox.is_empty(), "the successful turn drained its inbox rows");
    let durable = serde_json::to_string(&task).unwrap() + &serde_json::to_string(&inbox).unwrap();
    assert!(!durable.contains("dXBsb2FkZWQgYnl0ZXM="));
    thread.close().await;
    reopened.delete_thread(&created.thread_id).await.unwrap();
    reopened.close().await;
    std::fs::remove_dir_all(dir).unwrap();
}

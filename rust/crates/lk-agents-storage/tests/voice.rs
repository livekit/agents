//! Observed voice lease loss stops admission and abandons staged fields/history.

use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_abi::storage::voice::{
    VoiceStorage as _, VoiceStorageCommand, VoiceStorageEvent, VoiceStore as _,
};
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use lk_agents_storage::SqliteConnection;
use lk_agents_storage::context::store::ContextStore;
use lk_agents_storage::sqlite::FileConnection;
use lk_agents_storage::testing::Backend;
use lk_agents_storage::voice::VoiceStore;
use lk_agents_storage::{StorageError, sql::Statement};
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn observed_voice_lease_loss_refuses_new_work_and_discards_staged_saves() {
    let fixture =
        Backend::Sqlite.fixture(Some((Duration::from_secs(5), Duration::from_millis(10)))).await;
    let store = std::sync::Arc::new(
        VoiceStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into(), None)
            .await
            .unwrap(),
    );
    store
        .acquire(
            1,
            vec![ChatItem::Message(ChatMessage::text(
                "u",
                Role::User,
                "save",
                std::time::UNIX_EPOCH,
            ))],
        )
        .await
        .unwrap();
    let activation = store.activation();
    let field = |value| {
        vec![ContextField {
            name: "count".into(),
            schema: json!({"type":"integer"}),
            value: json!(value),
        }]
    };
    let loaded = activation.load("root:desk".into(), "Draft".into(), field(0)).await.unwrap();
    activation.save("root:desk".into(), "Draft".into(), loaded.revision, field(1)).await.unwrap();
    let file =
        fixture.dir.as_ref().unwrap().join("threads").join(format!("{}.sqlite", fixture.thread_id));
    let intruder = FileConnection::open(file, false).await.unwrap();
    intruder
        .batch(vec![Statement::raw(
            "UPDATE lanes SET lease_holder='foreign', lease_token='foreign' WHERE lane='desk'",
        )])
        .await
        .unwrap();
    let (sink, mut events) = Events::new();
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::ObserveLoss { speech: 1 }, sink.clone()).unwrap();
    let lost = tokio::time::timeout(Duration::from_secs(1), events.recv()).await.unwrap();
    assert!(matches!(lost, Some(VoiceStorageEvent::LeaseLost { speech: 1 })));
    assert!(store.lease_lost());
    assert!(matches!(store.acquire(2, Vec::new()).await, Err(StorageError::LeaseLost(_))));
    std::sync::Arc::clone(&store).dispatch(VoiceStorageCommand::Abort { speech: 1 }, sink).unwrap();
    let abandoned = events.recv().await;
    assert!(matches!(abandoned, Some(VoiceStorageEvent::Abandoned { speech: 1, result: Ok(()) })));
    let thread = fixture.thread("reader").await;
    assert!(thread.lane("desk").history().await.unwrap().is_empty());
    let restored =
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    assert!(
        restored
            .load("root:desk".into(), "Draft".into(), field(0))
            .await
            .unwrap()
            .fields
            .is_empty()
    );
    store.aclose().await.unwrap();
    restored.close();
    intruder.close().await;
    fixture.cleanup(&[&thread]).await;
}

/// The voice store's events, as the voice executor receives them.
struct Events(tokio::sync::mpsc::UnboundedSender<VoiceStorageEvent>);

impl Events {
    fn new() -> (std::sync::Arc<Self>, tokio::sync::mpsc::UnboundedReceiver<VoiceStorageEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (std::sync::Arc::new(Events(tx)), rx)
    }
}

impl lk_agents_abi::StreamSink<VoiceStorageEvent> for Events {
    fn push(&self, item: VoiceStorageEvent) -> Result<(), lk_agents_abi::errors::ProviderError> {
        let _ = self.0.send(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

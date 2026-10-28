//! Observed voice lease loss stops admission and abandons staged fields/history.

use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_abi::storage::voice::{VoiceStorage as _, VoiceStore as _};
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
    let store =
        VoiceStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into(), None)
            .await
            .unwrap();
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
    assert!(
        tokio::time::timeout(Duration::from_secs(1), store.wait_lost()).await.unwrap().unwrap()
    );
    assert!(store.lease_lost());
    assert!(matches!(store.acquire(2, Vec::new()).await, Err(StorageError::LeaseLost(_))));
    store.abort(1).await.unwrap();
    let thread = fixture.thread("reader").await;
    assert!(thread.lane("desk").db().history().await.unwrap().is_empty());
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

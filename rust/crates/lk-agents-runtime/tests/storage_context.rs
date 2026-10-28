//! Persisted context scopes through the real shared storage runtime on both backends.

mod storage_common;

use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_runtime::storage::context::ContextActivation;
use lk_agents_storage::StorageError;
use serde_json::{Value, json};
use storage_common::Backend;

fn field(name: &str, value: Value) -> ContextField {
    ContextField { name: name.into(), schema: json!({"type": "string"}), value }
}

async fn direct_context_saves_survive_reload_and_refuse_stale_activations(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let first =
        ContextActivation::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let stale =
        ContextActivation::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let fields = vec![field("draft", json!("first"))];
    let a = first.load("root:desk".into(), "Booking".into(), fields.clone()).await.unwrap();
    let b = stale.load("root:desk".into(), "Booking".into(), fields.clone()).await.unwrap();
    let saved =
        first.save("root:desk".into(), "Booking".into(), a.revision, fields.clone()).await.unwrap();
    assert_eq!(saved.revision, 1);
    assert!(first.writes().is_empty(), "returning success means the direct save committed");
    first.close();

    let restored =
        ContextActivation::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let loaded = restored
        .load("root:desk".into(), "Booking".into(), vec![field("draft", json!("default"))])
        .await
        .unwrap();
    assert_eq!(loaded.fields["draft"], json!("first"));
    assert_eq!(loaded.revision, 1);
    assert!(matches!(
        stale
            .save(
                "root:desk".into(),
                "Booking".into(),
                b.revision,
                vec![field("draft", json!("stale"))]
            )
            .await,
        Err(StorageError::VersionConflict { .. })
    ));
    let unchanged = restored.load("root:desk".into(), "Booking".into(), fields).await.unwrap();
    assert_eq!(unchanged.fields["draft"], json!("first"));
    let refreshed = stale
        .load("root:desk".into(), "Booking".into(), vec![field("draft", json!("default"))])
        .await
        .unwrap();
    assert_eq!(refreshed.fields["draft"], json!("first"));
    restored.close();
    stale.close();
    fixture.cleanup(&[]).await;
}

async fn several_context_declarations_share_one_direct_scope(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let store =
        ContextActivation::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let draft = vec![field("draft", json!("booking"))];
    let preferences = vec![field("language", json!("fr"))];
    let a = store.load("root:desk".into(), "Booking".into(), draft.clone()).await.unwrap();
    store.load("root:desk".into(), "Preferences".into(), preferences).await.unwrap();
    store.save("root:desk".into(), "Booking".into(), a.revision, draft).await.unwrap();
    assert!(store.load("root:desk".into(), "Booking".into(), Vec::new()).await.is_err());
    store.close();
    fixture.cleanup(&[]).await;
}

contract!(
    direct_context_saves_survive_reload_and_refuse_stale_activations,
    several_context_declarations_share_one_direct_scope,
);

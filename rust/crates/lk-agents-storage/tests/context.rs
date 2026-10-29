//! Persisted context scopes through the real storage on both backends.

use lk_agents_abi::storage::context::{AppTransaction as _, ContextActivation as _, ContextField};
use lk_agents_storage::context::store::ContextStore;
use lk_agents_storage::testing::Backend;
use lk_agents_storage::{CommitOptions, RunOptions, StorageError};
use serde_json::{Value, json};

fn field(name: &str, value: Value) -> ContextField {
    ContextField { name: name.into(), schema: json!({"type": "string"}), value }
}

async fn direct_context_saves_survive_reload_and_refuse_stale_activations(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let first =
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let stale =
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
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
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
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
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
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

async fn notes(store: &ContextStore) -> Vec<Vec<Value>> {
    let read = store.transaction(true).await.unwrap();
    let rows = read.fetch("SELECT body FROM notes ORDER BY id".into(), Vec::new()).await.unwrap();
    read.commit().await.unwrap();
    rows
}

/// docs/context.md#application-records-and-transactions: an independent transaction (and a
/// joining one outside a managed turn) commits when it ends; a managed turn's joins its commit.
async fn application_transactions_commit_on_exit_or_with_the_turn(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let (storage, thread_id) = (fixture.storage.clone(), fixture.thread_id.clone());
    let direct =
        ContextStore::bound(storage.clone(), thread_id.clone(), "desk".into()).await.unwrap();
    let tx = direct.transaction(false).await.unwrap();
    tx.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)".into(), vec![])
        .await
        .unwrap();
    tx.execute("INSERT INTO notes (body) VALUES (?)".into(), vec![json!("first")]).await.unwrap();
    let unseen = tx.fetch("SELECT body FROM notes".into(), Vec::new()).await;
    assert!(matches!(unseen, Err(StorageError::Invalid(_))), "a read after a staged write");
    tx.commit().await.unwrap();
    assert_eq!(notes(&direct).await, [[json!("first")]]);

    let discarded = direct.transaction(true).await.unwrap();
    discarded.execute("INSERT INTO notes (body) VALUES ('gone')".into(), vec![]).await.unwrap();
    discarded.rollback();
    assert_eq!(notes(&direct).await.len(), 1);

    let turn = ContextStore::turn(storage, thread_id, "desk".into(), Vec::new(), None, false)
        .await
        .unwrap();
    let staged = turn.transaction(false).await.unwrap();
    staged
        .execute("INSERT INTO notes (body) VALUES (?)".into(), vec![json!("turn")])
        .await
        .unwrap();
    staged.commit().await.unwrap();
    assert_eq!(notes(&direct).await.len(), 1, "staged until the turn commits");
    let thread = fixture.thread("w1").await;
    let mut run = thread.lane("desk").open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.app_writes = turn.statements();
    run.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(notes(&direct).await, [[json!("first")], [json!("turn")]]);
    fixture.cleanup(&[&thread]).await;
}

/// A turn's application writes are fenced with it: after a takeover they never apply.
async fn a_turns_application_writes_never_apply_after_a_takeover(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let direct =
        ContextStore::bound(fixture.storage.clone(), fixture.thread_id.clone(), "desk".into())
            .await
            .unwrap();
    let tx = direct.transaction(true).await.unwrap();
    tx.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)".into(), vec![])
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let (s1, s2) = (fixture.thread("w1").await, fixture.thread("w2").await);
    let mut stale = s1.lane("desk").open(RunOptions::new("r1", "AST_1")).await.unwrap();
    stale.lease().unwrap().release(None).await.unwrap();
    let mut fresh = s2.lane("desk").open(RunOptions::new("r2", "AST_2")).await.unwrap();
    fresh.commit(CommitOptions::default()).await.unwrap();

    stale.app_writes = vec![lk_agents_abi::storage::context::AppStatement {
        sql: "INSERT INTO notes (body) VALUES ('late')".into(),
        params: Vec::new(),
    }];
    let err = stale.commit(CommitOptions::default()).await.err().expect("fenced out");
    assert!(matches!(err, StorageError::VersionConflict { .. }), "{err}");
    assert!(notes(&direct).await.is_empty(), "the loser's application SQL wrote nothing");
    fixture.cleanup(&[&s1, &s2]).await;
}

lk_agents_storage::contract!(
    direct_context_saves_survive_reload_and_refuse_stale_activations,
    several_context_declarations_share_one_direct_scope,
    application_transactions_commit_on_exit_or_with_the_turn,
    a_turns_application_writes_never_apply_after_a_takeover,
);

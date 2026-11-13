use crate::context::store::ContextStore;
use crate::testing::Backend;
use crate::{CommitOptions, RunOptions, StorageError};
use lk_agents_abi::storage::context::{AppTransaction as _, ContextActivation as _, ContextField};
use lk_agents_abi::telemetry::Cx;
use serde_json::{Value, json};

fn field(name: &str, value: Value) -> ContextField {
    ContextField { name: name.into(), schema: json!({"type": "string"}), value }
}

#[tokio::test]
async fn standalone_session_owns_a_run_and_settles_only_explicit_app_saves() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let values = store
        .load("root:desk".into(), "App".into(), vec![field("draft", json!("initial"))])
        .await
        .unwrap();
    store.begin_session().await.unwrap();
    store
        .save(
            "root:desk".into(),
            "App".into(),
            values.revision,
            vec![field("draft", json!("saved"))],
        )
        .await
        .unwrap();
    store
        .stage_session(json!({"version":1,"kind":"voice","agent":"desk","history_ids":[]}))
        .unwrap();
    store.flush_session(vec![]).await.unwrap();
    let thread = fixture.thread().await;
    assert!(thread.context_scopes("desk").await.unwrap().is_empty());
    store.finish_session(true).await.unwrap();
    assert_eq!(thread.context_scopes("desk").await.unwrap()[0].fields["draft"], json!("saved"));
    store.begin_session().await.unwrap();
    let values = store
        .load("root:desk".into(), "App".into(), vec![field("draft", json!("default"))])
        .await
        .unwrap();
    store
        .save(
            "root:desk".into(),
            "App".into(),
            values.revision,
            vec![field("draft", json!("discard"))],
        )
        .await
        .unwrap();
    store.finish_session(false).await.unwrap();
    assert_eq!(thread.context_scopes("desk").await.unwrap()[0].fields["draft"], json!("saved"));
    assert_eq!(thread.partition("desk").info().await.unwrap().unwrap().lease_expires_at_ms, 0);
    store.close();
    fixture.cleanup(&[&thread]).await;
}

async fn direct_context_saves_survive_reload_and_refuse_stale_activations(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let first = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let stale = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
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

    let restored = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
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
    let store = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
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
    let rows = read.fetch("SELECT body FROM notes ORDER BY id".into(), None).await.unwrap();
    read.commit().await.unwrap();
    rows
}

/// An independent transaction (and a joining one outside a managed turn) commits when it ends; a
/// managed turn's joins its commit.
async fn application_transactions_commit_on_exit_or_with_the_turn(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let (storage, thread_id) = (fixture.storage.clone(), fixture.thread_id.clone());
    let direct = ContextStore::bound(storage.clone(), thread_id.clone(), "desk".into(), Cx::NONE)
        .await
        .unwrap();
    let tx = direct.transaction(false).await.unwrap();
    tx.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)".into(), None)
        .await
        .unwrap();
    tx.execute("INSERT INTO notes (body) VALUES (?)".into(), Some(vec![json!("first")]))
        .await
        .unwrap();
    let unseen = tx.fetch("SELECT body FROM notes".into(), None).await;
    assert!(matches!(unseen, Err(StorageError::Invalid(_))), "a read after a staged write");
    tx.commit().await.unwrap();
    assert_eq!(notes(&direct).await, [[json!("first")]]);

    let discarded = direct.transaction(true).await.unwrap();
    discarded.execute("INSERT INTO notes (body) VALUES ('gone')".into(), None).await.unwrap();
    discarded.rollback();
    assert_eq!(notes(&direct).await.len(), 1);

    let turn =
        ContextStore::turn(storage, thread_id, "desk".into(), Vec::new(), None, false, Cx::NONE)
            .await
            .unwrap();
    let staged = turn.transaction(false).await.unwrap();
    staged
        .execute("INSERT INTO notes (body) VALUES (?)".into(), Some(vec![json!("turn")]))
        .await
        .unwrap();
    staged.commit().await.unwrap();
    assert_eq!(notes(&direct).await.len(), 1, "staged until the turn commits");
    let thread = fixture.thread().await;
    let mut run = thread
        .partition("desk")
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
    run.stage_statements(turn.statements());
    run.commit(CommitOptions::default()).await.unwrap();
    assert_eq!(notes(&direct).await, [[json!("first")], [json!("turn")]]);
    fixture.cleanup(&[&thread]).await;
}

/// A turn's application writes are fenced with it: after a takeover they never apply.
async fn a_turns_application_writes_never_apply_after_a_takeover(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let direct = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let tx = direct.transaction(true).await.unwrap();
    tx.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)".into(), None)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let (s1, s2) = (fixture.thread().await, fixture.thread().await);
    let mut stale = s1
        .partition("desk")
        .open(RunOptions::new("r1", "AST_1"), &crate::Holder::new("test"))
        .await
        .unwrap();
    stale.lease().unwrap().release(None).await.unwrap();
    let mut fresh = s2
        .partition("desk")
        .open(RunOptions::new("r2", "AST_2"), &crate::Holder::new("test"))
        .await
        .unwrap();
    fresh.commit(CommitOptions::default()).await.unwrap();

    stale.stage_statements(vec![lk_agents_abi::storage::context::AppStatement {
        sql: "INSERT INTO notes (body) VALUES ('late')".into(),
        params: Vec::new(),
    }]);
    let err = stale.commit(CommitOptions::default()).await.expect_err("fenced out");
    assert!(matches!(err, StorageError::VersionConflict { .. }), "{err}");
    assert!(notes(&direct).await.is_empty(), "the loser's application SQL wrote nothing");
    fixture.cleanup(&[&s1, &s2]).await;
}

/// The SQLite backend on paused time, every thread connection under the returned faults: the
/// store, one thread's id, and the directory it lives in.
async fn faulty_store() -> (
    std::sync::Arc<dyn crate::Storage>,
    std::sync::Arc<crate::testing::Faults>,
    String,
    crate::testing::TempDir,
) {
    use crate::Storage as _;
    use crate::testing::{Faults, TempDir, Wrapped, tasks};
    use std::sync::Arc;
    let dir = TempDir::new("faulty_store_");
    let lease = crate::thread::LeaseTiming::default();
    let storage = crate::backends::sqlite::SqliteStorage::open(&dir.0, lease, tasks());
    let faults = Arc::<Faults>::default();
    let shared = faults.clone();
    let wrap = move |conn: crate::backends::sqlite::FileConnection| {
        Arc::new(shared.over(Box::new(conn))) as Arc<dyn crate::ThreadDbConnection>
    };
    let storage = Wrapped { storage: Arc::new(storage.await.unwrap()), wrap };
    let thread = storage.create_thread(Default::default()).await.unwrap().thread_id;
    (Arc::new(storage), faults, thread, dir)
}

/// A direct save whose acknowledgment stayed lost past the commit's wait resolves before the
/// next save: it committed, so the next save builds on it instead of failing a stale revision.
#[tokio::test(start_paused = true)]
async fn a_lost_direct_save_resolves_before_the_next_save() {
    use crate::testing::WriteFault;
    let (storage, faulty, thread, _dir) = faulty_store().await;
    let store = ContextStore::bound(storage, thread, "desk".into(), Cx::NONE).await.unwrap();
    let first = vec![field("draft", json!("first"))];
    let loaded = store.load("root:desk".into(), "Booking".into(), first.clone()).await.unwrap();
    faulty.fail_next_commit(WriteFault::LoseAckAfter);
    faulty.unreachable_after_commit_for(crate::thread::RESOLVE_WAIT * 2);
    let lost = store.save("root:desk".into(), "Booking".into(), loaded.revision, first).await;
    assert!(matches!(lost, Err(StorageError::Uncertain(_))), "{lost:?}");
    let second = vec![field("draft", json!("second"))];
    let saved = store.save("root:desk".into(), "Booking".into(), loaded.revision + 1, second).await;
    assert_eq!(saved.expect("the lost save committed; this one builds on it").revision, 2);
    let fields = vec![field("draft", json!("default"))];
    let reloaded = store.load("root:desk".into(), "Booking".into(), fields).await.unwrap();
    assert_eq!(reloaded.fields["draft"], json!("second"));
}

/// A refused direct save leaves nothing staged, even when the database cannot be read right
/// after: the next save never carries the refused values.
#[tokio::test(start_paused = true)]
async fn a_refused_direct_save_keeps_none_of_its_values() {
    use crate::testing::WriteFault;
    let (storage, faulty, thread, _dir) = faulty_store().await;
    let store = ContextStore::bound(storage, thread, "desk".into(), Cx::NONE).await.unwrap();
    let fields = vec![field("draft", json!("refused"))];
    let loaded = store.load("root:desk".into(), "Booking".into(), fields.clone()).await.unwrap();
    faulty.fail_next_commit(WriteFault::Refuse);
    faulty.unreachable_after_commit_for(std::time::Duration::from_secs(1));
    let refused = store.save("root:desk".into(), "Booking".into(), loaded.revision, fields).await;
    assert!(refused.is_err(), "{refused:?}");
    assert!(store.writes().is_empty(), "the refused values stayed staged");
}

/// An application transaction whose acknowledgment was lost after it applied is sent again
/// under its request id: it commits once and reports the commit.
#[tokio::test(start_paused = true)]
async fn a_lost_application_transaction_applies_once() {
    use crate::testing::WriteFault;
    let (storage, faulty, thread, _dir) = faulty_store().await;
    let store = ContextStore::bound(storage, thread, "desk".into(), Cx::NONE).await.unwrap();
    let tx = store.transaction(true).await.unwrap();
    tx.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)".into(), None)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let tx = store.transaction(true).await.unwrap();
    tx.execute("INSERT INTO notes (body) VALUES ('once')".into(), None).await.unwrap();
    faulty.fail_next_write("INSERT INTO notes", WriteFault::LoseAckAfter);
    tx.commit().await.expect("the resend finds its own commit");
    assert_eq!(notes(&store).await, [[json!("once")]]);
}

crate::contract!(
    direct_context_saves_survive_reload_and_refuse_stale_activations,
    several_context_declarations_share_one_direct_scope,
    application_transactions_commit_on_exit_or_with_the_turn,
    a_turns_application_writes_never_apply_after_a_takeover,
);

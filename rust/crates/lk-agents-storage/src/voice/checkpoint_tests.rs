use std::sync::Arc;

use lk_agents_abi::storage::context::{ContextActivation as _, ContextField};
use lk_agents_abi::storage::voice::{VoiceHistory, VoiceStorage as _, VoiceStore as _};
use lk_agents_abi::telemetry::Cx;
use serde_json::json;

use super::VoiceStore;
use crate::context::store::ContextStore;
use crate::testing::Backend;

fn name(value: &str) -> Vec<ContextField> {
    vec![ContextField {
        name: "name".into(),
        schema: json!({"type": "string"}),
        value: json!(value),
    }]
}

#[tokio::test]
async fn an_idle_checkpoint_releases_its_partition_and_leaves_application_saves_explicit() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let activation = store.activation();
    let loaded = activation.load("app".into(), "Person".into(), name("")).await.unwrap();
    let scope = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    activation
        .stage_session(crate::session_state::voice_value("desk".into(), vec![], None).unwrap())
        .unwrap();
    let mut save = std::pin::pin!(activation.save(
        "app".into(),
        "Person".into(),
        loaded.revision,
        name("Ada"),
        None,
    ));
    assert!(
        std::future::Future::poll(
            save.as_mut(),
            &mut std::task::Context::from_waker(std::task::Waker::noop()),
        )
        .is_pending(),
        "an app save cannot join an idle bookkeeping run"
    );
    let saved = scope.commit().await.unwrap();
    assert!(store.state.lock().await.active.is_none(), "the idle scope releases its lease");
    save.await.unwrap();
    let independent = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let reloaded = independent.load("app".into(), "Person".into(), name("")).await.unwrap();
    assert_eq!(reloaded.fields["name"], json!("Ada"), "the app save commits on its own");
    assert_eq!(independent.load_session().await.unwrap(), Some(saved));
    independent.close();
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

#[tokio::test]
async fn a_speech_can_take_an_idle_checkpoint_and_commit_its_explicit_application_saves() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let activation = store.activation();
    let loaded = activation.load("app".into(), "Person".into(), name("")).await.unwrap();
    let scope = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    let history = store.acquire(1, VoiceHistory::default()).await.unwrap();
    activation
        .save("app".into(), "Person".into(), loaded.revision, name("Ada"), None)
        .await
        .unwrap();
    let history = store.checkpoint(1, 1, "desk".into(), history).await.unwrap();
    store.finish(Some(1), 2, history).await.unwrap();
    assert!(store.state.lock().await.active.is_some(), "the checkpoint still owns the run");
    scope.commit().await.unwrap();
    let independent = ContextStore::bound(
        fixture.storage.clone(),
        fixture.thread_id.clone(),
        "desk".into(),
        Cx::NONE,
    )
    .await
    .unwrap();
    let reloaded = independent.load("app".into(), "Person".into(), name("")).await.unwrap();
    assert_eq!(reloaded.fields["name"], json!("Ada"));
    independent.close();
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

#[tokio::test]
async fn an_abandoned_speech_cannot_make_an_old_checkpoint_release_a_new_run() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let store = Arc::new(
        VoiceStore::bound(
            fixture.storage.clone(),
            fixture.thread_id.clone(),
            "desk".into(),
            None,
            Cx::NONE,
        )
        .await
        .unwrap(),
    );
    let initial = store.acquire(1, VoiceHistory::default()).await.unwrap();
    store.checkpoint(1, 1, "desk".into(), initial).await.unwrap();
    let old = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    store.abandon(1).await.unwrap();
    let current = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    assert!(matches!(old.commit().await, Err(crate::StorageError::Cancelled(_))));
    assert_eq!(store.state.lock().await.active.as_ref().unwrap().controls, 1);
    current.commit().await.unwrap();
    assert!(store.state.lock().await.active.is_none());
    store.aclose().await.unwrap();
    fixture.cleanup(&[]).await;
}

#[tokio::test]
async fn a_refused_final_commit_releases_the_partition_pinned_by_a_checkpoint() {
    use crate::Storage as _;
    use crate::testing::{Faults, TempDir, Wrapped, WriteFault, tasks};

    let dir = TempDir::new("refused_checkpoint_");
    let sqlite = Arc::new(
        crate::backends::sqlite::SqliteStorage::open(
            &dir.0,
            crate::thread::LeaseTiming::default(),
            tasks(),
        )
        .await
        .unwrap(),
    );
    let faults = Arc::<Faults>::default();
    let shared = faults.clone();
    let wrap = move |conn: crate::backends::sqlite::FileConnection| {
        Arc::new(shared.over(Box::new(conn))) as Arc<dyn crate::ThreadDbConnection>
    };
    let storage = Arc::new(Wrapped { storage: sqlite.clone(), wrap });
    let thread_id = storage.create_thread(Default::default()).await.unwrap().thread_id;
    let store = Arc::new(
        VoiceStore::bound(storage.clone(), thread_id.clone(), "desk".into(), None, Cx::NONE)
            .await
            .unwrap(),
    );
    let activation = store.activation();
    let loaded = activation.load("app".into(), "Person".into(), name("")).await.unwrap();
    let scope = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    let history = store.acquire(1, VoiceHistory::default()).await.unwrap();
    activation
        .save("app".into(), "Person".into(), loaded.revision, name("Ada"), None)
        .await
        .unwrap();
    let history = store.checkpoint(1, 1, "desk".into(), history).await.unwrap();
    store.finish(Some(1), 2, history).await.unwrap();
    faults.fail_next_commit(WriteFault::RefuseTooLarge);
    assert!(matches!(scope.commit().await, Err(crate::StorageError::TooLarge { .. })));
    let observer = sqlite.connect(&thread_id).await.unwrap();
    assert!(
        crate::testing::lessee(&observer, "desk").await.is_none(),
        "a refused commit cannot strand its lease"
    );
    crate::ThreadDbConnection::close(&observer).await;
    store.aclose().await.unwrap();
    storage.delete_thread(&thread_id).await.unwrap();
    storage.close().await;
}

#[tokio::test]
async fn canceling_a_checkpoint_waiter_after_the_write_applied_cannot_roll_it_back() {
    use crate::Storage as _;
    use crate::testing::{Faults, TempDir, Wrapped, tasks};

    let dir = TempDir::new("cancel_checkpoint_receipt_");
    let sqlite = Arc::new(
        crate::backends::sqlite::SqliteStorage::open(
            &dir.0,
            crate::thread::LeaseTiming::default(),
            tasks(),
        )
        .await
        .unwrap(),
    );
    let faults = Arc::<Faults>::default();
    let shared = faults.clone();
    let wrap = move |conn: crate::backends::sqlite::FileConnection| {
        Arc::new(shared.over(Box::new(conn))) as Arc<dyn crate::ThreadDbConnection>
    };
    let storage = Arc::new(Wrapped { storage: sqlite.clone(), wrap });
    let thread_id = storage.create_thread(Default::default()).await.unwrap().thread_id;
    let store = Arc::new(
        VoiceStore::bound(storage.clone(), thread_id.clone(), "desk".into(), None, Cx::NONE)
            .await
            .unwrap(),
    );
    let activation = store.activation();
    let initial = crate::session_state::voice_value("desk".into(), vec![], None).unwrap();
    activation.stage_session(initial.clone()).unwrap();
    let scope = store.acquire_checkpoint(VoiceHistory::default()).await.unwrap();
    let mut received = initial;
    let mut remote =
        lk_agents_abi::delegation::remote::RemoteAgentState::new("https://booking.test".into());
    remote.default_context_id = Some("trip".into());
    received["state"]["remote_agents"]["booking"] = serde_json::to_value(remote).unwrap();
    activation.stage_session(received.clone()).unwrap();
    let (applied, release) = faults.hold_next_receipt("UPDATE partitions SET state =");
    let background = storage.background();
    let (completed, ended) = lk_agents_net::signal::once();
    let waiter = background.tasks.scoped(&*background.spawner, "checkpoint caller", async move {
        completed.resolve(scope.commit().await);
    });
    applied.wait().await.unwrap();
    drop(waiter);
    assert!(ended.wait().await.is_err(), "the caller has ended before the receipt");
    release.resolve(());
    assert!(store.tasks.wait_empty(&*background.clock, None).await);
    assert_eq!(
        activation.committed_session(),
        Some(received.clone()),
        "the native acknowledgement must describe the applied checkpoint"
    );
    let observer = sqlite.connect(&thread_id).await.unwrap();
    let (_, saved) = crate::testing::conversation(&observer, "desk").await.unwrap();
    assert_eq!(
        saved,
        Some(received),
        "cleanup cannot replace the applied checkpoint with older state"
    );
    assert!(crate::testing::lessee(&observer, "desk").await.is_none());
    crate::ThreadDbConnection::close(&observer).await;
    store.aclose().await.unwrap();
    storage.delete_thread(&thread_id).await.unwrap();
    storage.close().await;
}

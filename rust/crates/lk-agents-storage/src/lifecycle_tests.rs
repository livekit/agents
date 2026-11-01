//! Thread lifecycle through the `Storage` trait, the same outcomes on both backends
//! (`storage_common`): create, read, list, update and delete, expiry, and what a turn commit
//! does not change.

use std::collections::HashMap;
use std::time::Duration;

use livekit_protocol::agent::agent_thread::{CreateRequest, ListRequest, UpdateRequest};
use lk_agents_storage::StorageError;
use lk_agents_storage::engine::LeaseTiming;
use lk_agents_storage::testing::Backend;
use lk_agents_storage::{CommitOptions, RunOptions};

fn ttl(seconds: i64) -> Option<pbjson_types::Duration> {
    Some(pbjson_types::Duration { seconds, nanos: 0 })
}

fn attributes(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn not_found<T: std::fmt::Debug>(r: Result<T, StorageError>) -> bool {
    matches!(r, Err(StorageError::NotFound(_)))
}

async fn a_thread_is_created_read_listed_and_updated(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let scope = lk_agents_storage::shortuuid("user-");
    let create = |a: &[(&str, &str)]| CreateRequest {
        scope: scope.clone(),
        attributes: attributes(a),
        idle_ttl: ttl(3600),
    };
    let first = s.create_thread(create(&[("topic", "billing")])).await.unwrap();
    let second = s.create_thread(create(&[])).await.unwrap();
    assert!(first.thread_id.starts_with("AT_") && first.thread_id != second.thread_id);
    assert_eq!(
        s.get_thread(&first.thread_id).await.unwrap().attributes,
        attributes(&[("topic", "billing")])
    );

    let page = s.list_threads(ListRequest { scope: scope.clone(), limit: 1, ..Default::default() });
    let page = page.await.unwrap();
    assert_eq!(page.threads.len(), 1);
    let rest = ListRequest { scope: scope.clone(), page_token: page.next_page_token, limit: 1 };
    let rest = s.list_threads(rest).await.unwrap();
    let mut listed: Vec<String> =
        page.threads.iter().chain(&rest.threads).map(|t| t.thread_id.clone()).collect();
    listed.sort();
    let mut created = vec![first.thread_id.clone(), second.thread_id.clone()];
    created.sort();
    assert_eq!(listed, created, "two pages, no gap or duplicate");

    // an empty value deletes its key only; an explicit zero TTL is kept, never read as "unset"
    let update = UpdateRequest {
        thread_id: first.thread_id.clone(),
        attributes: attributes(&[("topic", ""), ("tier", "gold")]),
        idle_ttl: ttl(0),
    };
    let updated = s.update_thread(update).await.unwrap();
    assert_eq!(updated.attributes, attributes(&[("tier", "gold")]));
    assert_eq!(updated.idle_ttl.map(|d| d.seconds), Some(0));
    let reread = s.get_thread(&first.thread_id).await.unwrap();
    assert_eq!(
        (reread.attributes, reread.idle_ttl.map(|d| d.seconds)),
        (attributes(&[("tier", "gold")]), Some(0))
    );

    for t in [&first, &second] {
        s.delete_thread(&t.thread_id).await.unwrap();
    }
    store.close().await;
}

async fn a_deleted_thread_is_gone_and_deleting_again_is_safe(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let t = s.create_thread(CreateRequest::default()).await.unwrap();
    s.delete_thread(&t.thread_id).await.unwrap();
    s.delete_thread(&t.thread_id).await.unwrap();
    assert!(not_found(s.get_thread(&t.thread_id).await));
    assert!(not_found(s.open(&t.thread_id, "w1").await.map(|_| ())));
    store.close().await;
}

async fn an_expired_thread_reads_as_not_found(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let scope = lk_agents_storage::shortuuid("user-");
    let req = CreateRequest { scope: scope.clone(), idle_ttl: ttl(1), ..Default::default() };
    let t = s.create_thread(req).await.unwrap();
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert!(not_found(s.get_thread(&t.thread_id).await));
    let listed = s.list_threads(ListRequest { scope, ..Default::default() }).await.unwrap();
    assert!(listed.threads.is_empty());
    store.close().await;
}

/// Only `update_thread` moves expiry: a managed turn's commit does not touch the thread.
async fn a_commit_does_not_extend_expiry(backend: Backend) {
    let store = backend.store(LeaseTiming::default()).await;
    let s = &store.storage;
    let t = s.create_thread(CreateRequest { idle_ttl: ttl(3600), ..Default::default() }).await;
    let t = t.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let thread = s.open(&t.thread_id, "w1").await.unwrap();
    let mut run = thread.lane("front-desk").open(RunOptions::new("r1", "AST_1")).await.unwrap();
    run.state = Some(serde_json::json!({"n": 1}));
    run.commit(CommitOptions::default()).await.unwrap();
    thread.close().await;
    assert_eq!(s.get_thread(&t.thread_id).await.unwrap().expires_at, t.expires_at);
    s.delete_thread(&t.thread_id).await.unwrap();
    store.close().await;
}

lk_agents_storage::contract!(
    a_thread_is_created_read_listed_and_updated,
    a_deleted_thread_is_gone_and_deleting_again_is_safe,
    an_expired_thread_reads_as_not_found,
    a_commit_does_not_extend_expiry,
);

/// Local ids name files: anything but a plain `AT_` id is refused before a path is built, and a
/// database file the registry does not list (a create that crashed before its registry row) is
/// not a thread.
#[tokio::test]
async fn sqlite_refuses_unknown_and_unlisted_ids() {
    let store = Backend::Sqlite.store(LeaseTiming::default()).await;
    let s = &store.storage;
    for id in ["AT_../../registry", "AT_a/b", "DB_x", "", "AT_unknown"] {
        assert!(not_found(s.get_thread(id).await), "{id}");
        assert!(not_found(s.open(id, "w1").await.map(|_| ())), "{id}");
    }
    let dir = store.dir.clone().unwrap();
    std::fs::write(dir.join("threads").join("AT_orphan.sqlite"), b"").unwrap();
    assert!(not_found(s.open("AT_orphan", "w1").await.map(|_| ())));
    store.close().await;
}

//! In-memory threads use the ordinary SQLite thread engine without file persistence.

use livekit_protocol::agent::agent_thread::CreateRequest;
use lk_agents_runtime::storage::engine::LeaseTiming;
use lk_agents_runtime::storage::sqlite::SqliteStorage;
use lk_agents_runtime::storage::{CommitOptions, RunOptions, Storage};
use lk_agents_storage::StorageError;
use serde_json::json;

#[tokio::test]
async fn memory_threads_keep_committed_state_between_handles_and_isolate_stores() {
    let storage = SqliteStorage::in_memory(LeaseTiming::default()).await.unwrap();
    let info = storage.create_thread(CreateRequest::default()).await.unwrap();
    let first = storage.open(&info.thread_id, "first").await.unwrap();
    let mut run = first.lane("desk").open(RunOptions::new("run", "")).await.unwrap();
    run.state = Some(json!({"reservation": "R1"}));
    run.commit(CommitOptions::default()).await.unwrap();
    first.close().await;

    let reopened = storage.open(&info.thread_id, "second").await.unwrap();
    assert_eq!(
        reopened.lane("desk").info().await.unwrap().unwrap().state,
        Some(json!({"reservation": "R1"}))
    );
    assert!(reopened.events(0, 100).await.unwrap().iter().any(|e| e.r#type == "run_ended"));
    reopened.close().await;

    let other = SqliteStorage::in_memory(LeaseTiming::default()).await.unwrap();
    assert!(matches!(
        other.get_thread(&info.thread_id).await,
        Err(StorageError::ThreadNotFound(_))
    ));
    let streams = storage.response_streams();
    let stream = streams.create(None).await.unwrap();
    streams.append(&stream.stream_id, b"hello").await.unwrap();
    streams.close(&stream.stream_id).await.unwrap();
    let mut reader = streams.read(&stream.stream_id, None, false);
    assert_eq!(reader.next().await.unwrap(), Some(b"hello".to_vec()));
    assert_eq!(reader.next().await.unwrap(), None);

    storage.delete_thread(&info.thread_id).await.unwrap();
    assert!(matches!(
        storage.open(&info.thread_id, "late").await,
        Err(StorageError::ThreadNotFound(_))
    ));
    storage.close().await;
    other.close().await;
}

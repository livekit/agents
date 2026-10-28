//! In-memory threads use the ordinary SQLite thread engine without file persistence.

use livekit_protocol::agent::agent_thread::CreateRequest;
use lk_agents_storage::engine::LeaseTiming;
use lk_agents_storage::sqlite::SqliteStorage;
use lk_agents_storage::{CommitOptions, RunOptions, Storage};
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_handles_commit_concurrently_while_their_leases_renew() {
    use lk_agents_storage::Thread;
    use std::time::Duration;

    async fn write(thread: &Thread, lane: &str) {
        for index in 0..32 {
            let mut run = thread
                .lane(lane)
                .open(RunOptions::new(format!("{lane}-{index}"), ""))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(3)).await;
            assert!(!run.lease().unwrap().is_lost());
            run.state = Some(json!({"saved": index}));
            run.commit(CommitOptions::default()).await.unwrap();
        }
    }

    let lease = LeaseTiming::new(Duration::from_secs(2), Duration::from_millis(1)).unwrap();
    let storage = SqliteStorage::in_memory(lease).await.unwrap();
    let info = storage.create_thread(CreateRequest::default()).await.unwrap();
    let first = storage.open(&info.thread_id, "first").await.unwrap();
    let second = storage.open(&info.thread_id, "second").await.unwrap();
    tokio::join!(write(&first, "first"), write(&second, "second"));
    for (thread, lane) in [(&first, "first"), (&second, "second")] {
        assert_eq!(
            thread.lane(lane).info().await.unwrap().unwrap().state,
            Some(json!({"saved": 31}))
        );
    }
    first.close().await;
    assert_eq!(
        second.lane("first").info().await.unwrap().unwrap().state,
        Some(json!({"saved": 31}))
    );
    storage.close().await;
    assert!(matches!(second.lane("second").info().await, Err(StorageError::Unavailable(_))));
    second.close().await;
}

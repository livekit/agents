use serde_json::json;

use super::*;
use crate::testing::Backend;
use crate::{CommitOptions, Holder, RunOptions};

#[tokio::test]
async fn compact_checkpoints_reuse_values_and_preserve_private_revisions() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let partition = thread.partition("default");
    let holder = Holder::new("snapshot-test");
    let mut run = partition.open(RunOptions::new("first", ""), &holder).await.unwrap();
    let original = json!({"id": "shared", "text": "x".repeat(80_000)});
    let mut state = json!({"version": 1, "kind": "text", "state": {"frames": {
        "root": {"history": [original.clone()], "visible_dialogue": [original.clone()]},
        "child": {"history": [original.clone()], "visible_dialogue": []}
    }}});
    run.checkpoint_session(state.clone(), vec![]).await.unwrap();
    run.checkpoint_session(state.clone(), vec![]).await.unwrap();
    let journal = thread.journal(0, 100).await.unwrap();
    assert_eq!(journal.iter().filter(|row| row.r#type == "session_item").count(), 3);
    let raw = partition.info().await.unwrap().unwrap().state;
    assert!(serde_json::to_string(&raw).unwrap().len() < 1000);
    let (restored, _) = hydrate(&**thread.conn(), "default", raw).await.unwrap();
    assert_eq!(restored, Some(state.clone()));

    // One fork edits the shared item; the other must keep the exact old revision.
    state["state"]["frames"]["root"]["history"][0]["text"] = json!("edited");
    run.checkpoint_session(state.clone(), vec![]).await.unwrap();
    run.commit(CommitOptions::default()).await.unwrap();
    let mut reopened = partition.open(RunOptions::new("second", ""), &holder).await.unwrap();
    assert_eq!(reopened.state(), Some(&state));
    reopened.checkpoint_session(state.clone(), vec![]).await.unwrap();
    let journal = thread.journal(0, 100).await.unwrap();
    assert_eq!(journal.iter().filter(|row| row.r#type == "session_item").count(), 5);
    reopened.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn a_missing_immutable_revision_is_an_error() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    thread.ensure_schema().await.unwrap();
    let packed = json!({"version": 1, "kind": "text", "storage_version": 1,
        "state": {"frames": {"root": {"history": "missing", "visible_dialogue": "missing"}}}});
    assert!(matches!(
        hydrate(&**thread.conn(), "default", Some(packed)).await,
        Err(StorageError::Invalid(_))
    ));
    fixture.cleanup(&[&thread]).await;
}

#[tokio::test]
async fn long_histories_do_not_grow_the_routing_snapshot() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let partition = thread.partition("default");
    let holder = Holder::new("long-history");
    let mut run = partition.open(RunOptions::new("long", ""), &holder).await.unwrap();
    let items: Vec<_> = (0..2500).map(|index| json!({"id":index,"text":"saved"})).collect();
    let state = json!({"version":1,"kind":"text","state":{"frames":{"root":{
        "history":items,"visible_dialogue":[]
    }}}});
    run.checkpoint_session(state.clone(), vec![]).await.unwrap();
    let raw = partition.info().await.unwrap().unwrap().state;
    assert!(serde_json::to_string(&raw).unwrap().len() < 1000);
    let (hydrated, _) = hydrate(&**thread.conn(), "default", raw).await.unwrap();
    assert_eq!(hydrated, Some(state));
    run.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

/// Encoding microbenchmark only: excludes transport, fencing and whole-task JSON rewrites.
#[test]
#[ignore = "manual JSON/JSONB CPU and encoded-size measurement"]
#[allow(clippy::print_stdout, reason = "an explicitly invoked benchmark reports its measurements")]
fn jsonb_history_payload_benchmark() {
    let payloads: Vec<_> = (0..2000)
        .map(|index| {
            serde_json::to_string(&json!({
                "id": format!("message-{index}"), "type":"message", "role":"assistant",
                "content":[{"type":"text","text":"word ".repeat(100)}],
                "created_at":1_700_000_000 + index,"extra":{"source":"test"}
            }))
            .unwrap()
        })
        .collect();
    for (label, encode) in [("JSON", "?2"), ("JSONB", "jsonb(?2)")] {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE items (id INTEGER PRIMARY KEY, payload)").unwrap();
        let started = std::time::Instant::now();
        let transaction = conn.transaction().unwrap();
        for (index, payload) in payloads.iter().enumerate() {
            transaction
                .execute(
                    &format!("INSERT INTO items VALUES (?1, {encode})"),
                    rusqlite::params![index as i64, payload],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
        let write = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..100 {
            let length: i64 = conn
                .query_row(
                    "SELECT SUM(length(json_extract(payload, '$.content[0].text'))) FROM items",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(length, 1_000_000);
        }
        let read = started.elapsed();
        let bytes: i64 =
            conn.query_row("SELECT SUM(length(payload)) FROM items", [], |row| row.get(0)).unwrap();
        println!(
            "{label}: 2000 items, encoded_bytes={bytes}, write={write:?}, 100 history field scans={read:?}"
        );
    }
}

/// Local debug-build measurement through the real fenced SQLite checkpoint path.
#[tokio::test]
#[ignore = "manual long-history checkpoint and reader latency measurement"]
#[allow(clippy::print_stdout, reason = "an explicitly invoked benchmark reports its measurements")]
async fn long_history_checkpoint_latency_benchmark() {
    let fixture = Backend::Sqlite.fixture(None).await;
    let thread = fixture.thread().await;
    let partition = thread.partition("default");
    let holder = Holder::new("latency-benchmark");
    let mut run = partition.open(RunOptions::new("stream", ""), &holder).await.unwrap();
    let history: Vec<_> = (0..2000)
        .map(|index| {
            json!({"id":format!("message-{index}"),"type":"message",
            "role":"assistant","content":["word ".repeat(100)],"created_at":1700000000})
        })
        .collect();
    let mut state = json!({"version":1,"kind":"text","state":{"frames":{"root":{
        "history":history,"visible_dialogue":[]
    }}}});
    let started = std::time::Instant::now();
    run.checkpoint_session(state.clone(), vec![]).await.unwrap();
    let initial = started.elapsed();
    let mut writes = Vec::new();
    let mut reads = Vec::new();
    let started = std::time::Instant::now();
    for index in 0..100 {
        state["state"]["frames"]["root"]["history"][1999]["content"] =
            json!(["token ".repeat(index + 1)]);
        let checkpoint_started = std::time::Instant::now();
        run.checkpoint_session(state.clone(), vec![]).await.unwrap();
        writes.push(checkpoint_started.elapsed());
        let read_started = std::time::Instant::now();
        let raw = partition.info().await.unwrap().unwrap().state;
        let (restored, _) = hydrate(&**thread.conn(), "default", raw).await.unwrap();
        reads.push(read_started.elapsed());
        assert_eq!(restored.as_ref(), Some(&state));
    }
    let elapsed = started.elapsed();
    writes.sort();
    reads.sort();
    let packed = partition.info().await.unwrap().unwrap().state.unwrap();
    println!(
        "SQLite fenced checkpoints: 2000 messages, 100 partial revisions; initial={initial:?}, \
         write_p50={:?}, write_p95={:?}, write_max={:?}, read_p50={:?}, read_p95={:?}, \
         combined_iterations_per_second={:.1}, routing_bytes={}",
        writes[50],
        writes[95],
        writes[99],
        reads[50],
        reads[95],
        100.0 / elapsed.as_secs_f64(),
        serde_json::to_vec(&packed).unwrap().len()
    );
    run.commit(CommitOptions::default()).await.unwrap();
    fixture.cleanup(&[&thread]).await;
}

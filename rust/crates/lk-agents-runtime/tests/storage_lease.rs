//! A cloned thread handle cannot replace an active run's lease.

use lk_agents_runtime::storage::{CommitOptions, RunOptions};
use lk_agents_storage::{StorageError, run::RunOutcome};

mod storage_common;
use storage_common::Backend;

async fn cloned_handles_do_not_replace_an_active_runs_lease(backend: Backend) {
    let fixture = backend.fixture(None).await;
    let thread = fixture.thread("worker-a").await;
    let cloned = thread.clone();
    let lane = thread.lane("front-desk");
    let mut first = lane.open(RunOptions::new("first", "AST_first")).await.unwrap();
    let result = cloned
        .lane("front-desk")
        .open(RunOptions { wait: false, ..RunOptions::new("second", "AST_second") })
        .await;
    let refused = match result {
        Err(StorageError::LeaseHeld(_)) => true,
        Ok(mut stolen) => {
            stolen.abandon(RunOutcome::Failed, "test cleanup").await.unwrap();
            false
        }
        Err(error) => panic!("unexpected second-run result: {error}"),
    };
    let original = first.commit(CommitOptions::default()).await;
    fixture.cleanup(&[&thread]).await;
    assert!(refused, "a clone replaced the active run's lease");
    assert!(original.is_ok(), "the original run lost its lease: {original:?}");
}

contract!(cloned_handles_do_not_replace_an_active_runs_lease);

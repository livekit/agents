use super::*;
use futures_util::FutureExt as _;

#[tokio::test(start_paused = true)]
async fn a_hundred_unawaited_updates_are_owned_ordered_and_fenced_before_the_tool_returns() {
    let (send, mut receive) = mpsc::channel(1);
    let owner = Arc::new(TaskSet::default());
    let spawner: Arc<dyn Spawner> = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let port = Progress::new("original".into(), send, &owner, spawner);
    let operation = port.open(7, CallId(3));
    for n in 0..100 {
        drop(port.enqueue(7, format!("step {n}"), n % 2 == 0));
    }
    let mut final_fence = Box::pin(operation.settle());
    assert!((&mut final_fence).now_or_never().is_none());
    for n in 0..100 {
        assert!(
            matches!(receive.recv().await, Some((source, Report::Progress { call: CallId(3), message, silent })) if source == "original" && message == format!("step {n}") && silent == (n % 2 == 0))
        );
    }
    final_fence.await;
    assert_eq!(port.state.lock().pending, 0);
    assert!(owner.wait_empty(&lk_agents_net::clock::RealClock::new(), None).await);
}

#[tokio::test(start_paused = true)]
async fn local_report_pressure_is_bounded_before_tasks_are_admitted_and_remains_retryable() {
    let (send, receive) = mpsc::channel(1);
    let owner = Arc::new(TaskSet::default());
    let spawner: Arc<dyn Spawner> = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let port = Progress::new("original".into(), send, &owner, spawner);
    let operation = port.open(7, CallId(3));
    for _ in 0..MAX_PENDING_REPORTS {
        drop(port.enqueue(7, "queued".into(), false));
    }
    assert_eq!(port.state.lock().pending, MAX_PENDING_REPORTS);
    let error = port.enqueue(7, "over capacity".into(), false).await.unwrap_err();
    assert!(matches!(
        error,
        ProviderError::Local(ApiError { error_kind: ErrorKind::RateLimit, .. })
    ));
    drop(receive);
    operation.settle().await;
    assert!(owner.wait_empty(&lk_agents_net::clock::RealClock::new(), None).await);
    assert_eq!(port.state.lock().pending, 0);
}

use super::*;
use lk_agents_inference::testkit::fallback::{PolicyProvider, policy};

#[tokio::test]
async fn fallback_tts_resolves_each_operation_for_streaming_and_sentence_backends() {
    for streaming in [true, false] {
        let (calls, mut seen) = mpsc::channel(16);
        let (clock, spawner, observer) = parts();
        let fallback: Arc<dyn Tts> = Arc::new(
            FallbackTts::new(
                vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming })],
                None,
                ChainPolicy::tts_defaults(),
                clock,
                spawner,
                observer,
            )
            .unwrap(),
        );
        // Binding a second call before opening the first must not replace its options.
        let first = fallback.with_conn_options(&policy(0.125)).unwrap_or_else(|| fallback.clone());
        let second = fallback.with_conn_options(&policy(0.25)).unwrap_or_else(|| fallback.clone());
        let mut source = first.synthesize("first").await.unwrap();
        drain_tts(&mut source).await;
        source.close(NO_DEADLINE).await;
        let (sink, mut source) = second.stream().await.unwrap();
        sink.push("second.".into()).unwrap();
        sink.end_input();
        drain_tts(&mut source).await;
        source.close(NO_DEADLINE).await;
        for read in [0.125, 0.25] {
            let actual = seen.try_recv().expect("the provider executed");
            assert_eq!(actual, ApiConnectOptions { max_retry: 0, ..policy(read) });
        }
    }
}

#[tokio::test]
async fn a_fallback_deadline_includes_the_wait_for_its_runtime_task() {
    let (calls, mut seen) = mpsc::channel(4);
    let clock = Arc::new(VirtualClock::new());
    let (_, spawner, observer) = parts();
    let fallback = FallbackTts::new(
        vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming: true })],
        None,
        ChainPolicy::tts_defaults(),
        clock.clone(),
        spawner,
        observer,
    )
    .unwrap();
    let call = fallback
        .with_conn_options(&ApiConnectOptions { total_timeout: Some(1.0), ..policy(0.125) })
        .unwrap();
    let mut source = call.synthesize("hello").await.unwrap();
    // No task has been polled yet: this is admitted work awaiting runtime execution.
    clock.advance(Duration::from_secs(2));
    let outcome = source.next().await;
    assert!(outcome.is_err(), "the deadline started at acceptance, before the task ran");
    assert!(seen.try_recv().is_err(), "expired work cannot open a provider");
    source.close(NO_DEADLINE).await;
    fallback.aclose();
}

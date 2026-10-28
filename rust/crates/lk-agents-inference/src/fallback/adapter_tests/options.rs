use super::*;
use crate::testkit::fallback::{PolicyProvider, policy};

#[tokio::test]
async fn fallback_stt_resolves_each_operation_for_batch_and_stream() {
    let (calls, mut seen) = mpsc::channel(16);
    let (clock, spawner, observer) = parts();
    let fallback: Arc<dyn Stt> = Arc::new(
        FallbackStt::<Standalone>::new(
            vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming: true })],
            ChainPolicy {
                attempt_timeout: None,
                retry_interval: None,
                ..ChainPolicy::stt_defaults()
            },
            clock,
            spawner,
            observer,
        )
        .unwrap(),
    );
    let first = fallback.with_conn_options(&policy(0.125)).unwrap_or_else(|| fallback.clone());
    let second = fallback.with_conn_options(&policy(0.25)).unwrap_or_else(|| fallback.clone());
    first.recognize(&[], None).await.unwrap();
    let (sink, mut source) = second.stream(None, None).await.unwrap().unwrap();
    sink.end_input();
    assert!(source.next().await.unwrap().is_none());
    source.close(NO_DEADLINE).await;
    for read in [0.125, 0.25] {
        let actual = seen.try_recv().expect("the provider executed");
        assert_eq!(actual, ApiConnectOptions { max_retry: 0, ..policy(read) });
    }
}

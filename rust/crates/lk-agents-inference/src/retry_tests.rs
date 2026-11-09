use super::*;
use lk_agents_abi::VirtualClock;
use std::sync::atomic::{AtomicBool, Ordering};

struct Panicking(Arc<AtomicBool>);

impl StreamOutput<u32> for Panicking {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<u32>, ProviderError>> {
        Box::pin(async { panic!("the provider's pull panicked") })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        self.0.store(true, Ordering::SeqCst);
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<u32> for Panicking {}

#[tokio::test]
async fn a_panicked_pull_still_closes_its_source() {
    let closed = Arc::new(AtomicBool::new(false));
    let source = closed.clone();
    let open: Open<u32> = Arc::new(move || {
        let source = source.clone();
        Box::pin(async move { Ok(Box::new(Panicking(source)) as Box<dyn ProviderStream<u32>>) })
    });
    let mut stream = stream(
        &ApiConnectOptions::default(),
        Arc::new(VirtualClock::new()),
        open,
        |_| true,
        Arc::new(|_, _, _| {}),
        CancellationToken::new(),
    )
    .await;
    let pulled = AssertUnwindSafe(stream.next()).catch_unwind().await;
    assert!(pulled.is_err());
    assert_eq!(stream.close(Ts::from_nanos(u64::MAX)).await, Closed::Clean);
    assert!(closed.load(Ordering::SeqCst));
}

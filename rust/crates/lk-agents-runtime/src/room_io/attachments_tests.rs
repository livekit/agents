use super::*;
use lk_agents_abi::{Closed, VirtualClock};
use lk_agents_net::signal;
use lk_agents_net::task::TokioSpawner;
use std::future::Future as _;
use std::task::{Context, Waker};

#[tokio::test]
async fn a_second_close_joins_cleanup_after_the_first_waiter_is_cancelled() {
    let registry = AttachmentRegistry::default();
    let owner =
        registry.owner(Arc::new(VirtualClock::new()), Arc::new(TokioSpawner::current())).unwrap();
    let (release, gate) = signal::once();
    let binding = owner
        .automatic(BindingCapabilities::default())
        .unwrap()
        .activate(
            || (),
            Box::new(move |_, _| {
                Box::pin(async move {
                    gate.wait().await.unwrap();
                    Closed::Clean
                })
            }),
        )
        .unwrap();
    let mut first = Box::pin(registry.aclose());
    assert!(first.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
    drop(first);
    let mut second = Box::pin(registry.aclose());
    assert!(second.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
    release.finish(Some(()));
    second.await.unwrap();
    assert_eq!(binding.aclose().await, Closed::Clean);
    registry.aclose().await.unwrap();
}

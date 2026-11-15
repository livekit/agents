use super::*;
use lk_agents_abi::{Closed, VirtualClock};
use lk_agents_net::owned::{Background, Closer};
use lk_agents_net::signal;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use lk_agents_voice::room_io::bindings::{BindingKey, BindingRefused};
use lk_agents_voice::room_io::shared::BindingCapabilities;
use std::future::Future as _;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

fn owner() -> Arc<BindingOwner> {
    BindingOwner::new(
        Background::new(
            Arc::new(TaskSet::default()),
            Arc::new(TokioSpawner::current()),
            Arc::new(VirtualClock::new()),
        ),
        true,
    )
}

fn voice() -> BindingCapabilities {
    BindingCapabilities { legacy_voice: true }
}

fn close_count(counter: Arc<AtomicUsize>) -> Closer<Arc<()>> {
    Box::new(move |_, _| {
        Box::pin(async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Closed::Clean
        })
    })
}

#[tokio::test]
async fn dropping_a_binding_runs_its_owned_cleanup_before_the_key_can_be_reused() {
    let owner = owner();
    let (release, gate) = signal::once();
    let (began, closing) = signal::once();
    let resource = owner
        .bind("voice".into(), voice())
        .unwrap()
        .activate(
            || (),
            Box::new(move |_, _| {
                Box::pin(async move {
                    began.finish(Some(()));
                    gate.wait().await.unwrap();
                    Closed::Clean
                })
            }),
        )
        .unwrap();
    resource.started().unwrap();
    drop(resource);
    closing.wait().await.unwrap();
    assert!(matches!(
        owner.bind("voice".into(), voice()),
        Err(BindingRefused::KeyInUse(BindingKey::Named(key))) if key == "voice"
    ));
    assert_eq!(owner.legacy_projection(), None);
    release.finish(Some(()));
    assert!(owner.background.tasks.wait_empty(owner.background.clock.as_ref(), None).await);
    let replacement = owner.bind("voice".into(), voice()).unwrap();
    drop(replacement);
    assert_eq!(owner.aclose().await, Closed::Clean);
}

#[tokio::test]
async fn cancelling_a_binding_close_waiter_does_not_cancel_or_repeat_its_cleanup() {
    let owner = owner();
    let count = Arc::new(AtomicUsize::new(0));
    let (release, gate) = signal::once();
    let count_close = count.clone();
    let resource = owner
        .automatic(voice())
        .unwrap()
        .activate(
            || (),
            Box::new(move |_, _| {
                Box::pin(async move {
                    gate.wait().await.unwrap();
                    count_close.fetch_add(1, Ordering::SeqCst);
                    Closed::Clean
                })
            }),
        )
        .unwrap();
    let mut closing = Box::pin(resource.aclose());
    assert!(Pin::new(&mut closing).poll(&mut Context::from_waker(Waker::noop())).is_pending());
    drop(closing);
    release.finish(Some(()));
    assert_eq!(owner.aclose().await, Closed::Clean);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn owner_close_retires_unused_declarations_and_closes_every_live_resource_once() {
    let owner = owner();
    let count = Arc::new(AtomicUsize::new(0));
    let unused = owner.bind("unused".into(), voice()).unwrap();
    let first =
        owner.automatic(voice()).unwrap().activate(|| (), close_count(count.clone())).unwrap();
    let second =
        owner.automatic(voice()).unwrap().activate(|| (), close_count(count.clone())).unwrap();
    first.started().unwrap();
    second.started().unwrap();
    assert_eq!(owner.aclose().await, Closed::Clean);
    assert_eq!(owner.legacy_projection(), None);
    assert!(matches!(owner.automatic(voice()), Err(BindingRefused::OwnerClosed)));
    assert!(matches!(
        unused.activate(|| (), close_count(count.clone())),
        Err(BindingRefused::OwnerClosed)
    ));
    assert_eq!(first.aclose().await, Closed::Clean);
    assert_eq!(second.aclose().await, Closed::Clean);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn owner_close_racing_resource_construction_cleans_the_late_resource() {
    let owner = owner();
    let count = Arc::new(AtomicUsize::new(0));
    let declaration = owner.bind("voice".into(), voice()).unwrap();
    let during_construction = owner.clone();
    assert!(matches!(
        declaration.activate(|| during_construction.request_close(), close_count(count.clone())),
        Err(BindingRefused::Unavailable)
    ));
    assert_eq!(owner.aclose().await, Closed::Clean);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failed_retirement_reserves_the_key_instead_of_publishing_over_uncertain_resources() {
    let owner = owner();
    let resource = owner
        .bind("voice".into(), voice())
        .unwrap()
        .activate(|| (), Box::new(|_, _| Box::pin(async { Closed::Failed })))
        .unwrap();
    resource.started().unwrap();
    assert_eq!(resource.aclose().await, Closed::Failed);
    assert!(matches!(owner.bind("voice".into(), voice()), Err(BindingRefused::KeyInUse(_))));
    assert_eq!(owner.aclose().await, Closed::Failed);
}

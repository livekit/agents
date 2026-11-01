use super::*;
use lk_agents_abi::agent::resources::RunResources as _;
use lk_agents_abi::llm::toolsets::{self, ToolsetLifetime as _};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

struct Hooks {
    lifetime: Arc<dyn toolsets::Lifetime>,
    setups: AtomicUsize,
    closes: AtomicUsize,
    entered: Notify,
    closing: Notify,
    release: Notify,
    hold_setup: bool,
    hold_close: bool,
    fail_setup: bool,
    fail_close: bool,
}
impl Hooks {
    fn new() -> Self {
        Self {
            lifetime: Arc::new(crate::toolsets::lifetime::Lifetime::new(
                &ProcessEnv::current_runtime(),
            )),
            setups: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            entered: Notify::new(),
            closing: Notify::new(),
            release: Notify::new(),
            hold_setup: false,
            hold_close: false,
            fail_setup: false,
            fail_close: false,
        }
    }
}
impl ToolsetHooks for Hooks {
    fn lifetime(&self) -> &Arc<dyn toolsets::Lifetime> {
        &self.lifetime
    }
    fn setup(
        &self,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            self.setups.fetch_add(1, Ordering::SeqCst);
            self.entered.notify_one();
            if self.hold_setup {
                cancel.cancelled().await;
                // Model an authored finally block that must settle before resources close.
                self.closing.notify_one();
                self.release.notified().await;
                return Err(HostError::new("cancelled setup"));
            }
            if self.fail_setup { Err(HostError::new("original setup failure")) } else { Ok(()) }
        })
    }
    fn close(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.closing.notify_one();
            if self.hold_close {
                self.release.notified().await;
            }
            if self.fail_close { Err(HostError::new("close failure")) } else { Ok(()) }
        })
    }
    fn children(&self) -> Result<Option<Arc<dyn toolsets::LeaseGroup>>, HostError> {
        Ok(None)
    }
}

fn resources(hooks: Vec<Arc<dyn ToolsetHooks>>) -> RunResources {
    RunResources::new(&ProcessEnv::current_runtime(), None, None, None, "agent".into(), hooks)
        .unwrap()
}

struct Hydrated {
    activation: ContextStore,
    releases: Arc<AtomicUsize>,
}
impl ContextHydration for Hydrated {
    fn release(&self) -> Result<(), HostError> {
        assert!(self.activation.closed(), "release language objects after native activation");
        self.releases.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn cancelled_close_waiter_rejoins_cleanup_and_hydration_releases_once() {
    let hooks = Arc::new(Hooks { hold_close: true, ..Hooks::new() });
    let owner = resources(vec![hooks.clone()]);
    owner.open().unwrap().await.unwrap();
    assert!(!*owner.0.closing.borrow(), "successful startup must keep resources open");
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 0);
    let activation = owner.activation().unwrap();
    let releases = Arc::new(AtomicUsize::new(0));
    owner.retain_hydration(Arc::new(Hydrated {
        activation: activation.clone(),
        releases: releases.clone(),
    }));
    let closing = owner.aclose();
    hooks.closing.notified().await;
    drop(closing);
    assert!(!activation.closed());
    let mut rejoined = owner.aclose();
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(rejoined.as_mut().poll(cx).is_pending()))
            .await
    );
    hooks.release.notify_one();
    rejoined.await.unwrap();
    assert!(activation.closed());
    assert_eq!(releases.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 1);
    owner.aclose().await.unwrap();
    assert!(matches!(owner.open(), Err(ResourceError::Closed)));
}

#[tokio::test]
async fn cancelling_open_waits_for_setup_cleanup_before_ending_owned_context() {
    let hooks = Arc::new(Hooks { hold_setup: true, ..Hooks::new() });
    let owner = resources(vec![hooks.clone()]);
    let opening = owner.open().unwrap();
    hooks.entered.notified().await;
    let activation = owner.0.opened.get().unwrap().activation.clone().unwrap();
    drop(opening);
    hooks.closing.notified().await;
    let mut closing = owner.aclose();
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(closing.as_mut().poll(cx).is_pending()))
            .await
    );
    assert!(!activation.closed());
    hooks.release.notify_one();
    closing.await.unwrap();
    assert!(activation.closed());
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn borrowed_thread_and_context_remain_usable_after_the_request() {
    let env = ProcessEnv::current_runtime();
    let namespace = Arc::new(Namespace::new(&env, None, None, None, None, None, None).unwrap());
    let info = namespace.create_thread(None, None, 0).await.unwrap();
    let activation =
        namespace._core_context_store(info.thread_id.clone(), "agent".into()).await.unwrap();
    let owner = RunResources::new(
        &env,
        Some(namespace.clone()),
        Some(info.thread_id.clone()),
        Some(activation.clone()),
        "agent".into(),
        vec![],
    )
    .unwrap();
    owner.open().unwrap().await.unwrap();
    owner.aclose().await.unwrap();
    assert!(!activation.closed());
    assert_eq!(
        namespace.get_thread(info.thread_id.clone()).await.unwrap().thread_id,
        info.thread_id
    );
    activation.close();
    namespace.aclose().await;
}

#[tokio::test]
async fn a_new_context_over_borrowed_storage_ends_only_that_context() {
    let env = ProcessEnv::current_runtime();
    let namespace = Arc::new(Namespace::new(&env, None, None, None, None, None, None).unwrap());
    let info = namespace.create_thread(None, None, 0).await.unwrap();
    let owner = RunResources::new(
        &env,
        Some(namespace.clone()),
        Some(info.thread_id.clone()),
        None,
        "agent".into(),
        vec![],
    )
    .unwrap();
    owner.open().unwrap().await.unwrap();
    let activation = owner.activation().unwrap();
    owner.aclose().await.unwrap();
    assert!(activation.closed());
    namespace.get_thread(info.thread_id).await.unwrap();
    namespace.aclose().await;
}

#[tokio::test]
async fn failed_start_preserves_setup_error_when_rollback_also_fails() {
    let ready = Arc::new(Hooks { fail_close: true, ..Hooks::new() });
    let failed = Arc::new(Hooks { fail_setup: true, ..Hooks::new() });
    let owner = resources(vec![ready.clone(), failed.clone()]);
    let error = owner.open().unwrap().await.unwrap_err();
    assert_eq!(error.to_string(), "original setup failure");
    assert_eq!(ready.closes.load(Ordering::SeqCst), 1);
    assert_eq!(failed.closes.load(Ordering::SeqCst), 0);
    owner.rollback().await;
    assert!(owner.aclose().await.is_err());
    assert!(owner.0.opened.get().unwrap().activation.as_ref().unwrap().closed());
    let opened = owner.0.opened.get().unwrap();
    assert!(matches!(
        opened.namespace.as_ref().unwrap().get_thread(opened.thread_id.clone().unwrap()).await,
        Err(lk_agents_abi::storage::StorageError::Unavailable(_))
    ));
}

#[tokio::test]
async fn close_before_open_never_initializes_toolsets_or_storage() {
    let hooks = Arc::new(Hooks::new());
    let owner = resources(vec![hooks.clone()]);
    owner.aclose().await.unwrap();
    assert!(matches!(owner.open(), Err(ResourceError::Closed)));
    assert!(owner.0.opened.get().is_none());
    assert_eq!(hooks.setups.load(Ordering::SeqCst), 0);
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn dropping_the_owner_still_settles_its_ready_resources() {
    let hooks = Arc::new(Hooks::new());
    let owner = resources(vec![hooks.clone()]);
    owner.open().unwrap().await.unwrap();
    let activation = owner.activation().unwrap();
    let closed = owner.0.closed.clone();
    drop(owner);
    wait(&closed).await.unwrap();
    assert!(activation.closed());
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 1);
}

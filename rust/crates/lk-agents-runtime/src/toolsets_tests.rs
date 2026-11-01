use super::{group::Group, lifetime::Lifetime, scopes::Scopes};
use lk_agents_abi::llm::toolsets::{
    self, ToolsetGroup as _, ToolsetLifetime as _, ToolsetScopes as _,
};
use lk_agents_abi::{BoxFuture, SpawnerExt, errors::HostError};
use lk_agents_net::{env::ProcessEnv, task::TokioSpawner};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

type Outcome = Result<(), HostError>;
struct Hooks {
    lifetime: Arc<dyn toolsets::Lifetime>,
    setups: AtomicUsize,
    closes: AtomicUsize,
    setup_entered: Notify,
    setup_release: Notify,
    close_entered: Notify,
    close_release: Notify,
    hold_setup: bool,
    hold_close: bool,
    fail_setup: bool,
    fail_close: bool,
    children: Option<Arc<Group>>,
}
impl Hooks {
    fn new() -> Self {
        Self {
            lifetime: Arc::new(Lifetime::new(&ProcessEnv::current_runtime())),
            setups: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            setup_entered: Notify::new(),
            setup_release: Notify::new(),
            close_entered: Notify::new(),
            close_release: Notify::new(),
            hold_setup: false,
            hold_close: false,
            fail_setup: false,
            fail_close: false,
            children: None,
        }
    }
}
impl toolsets::ToolsetHooks for Hooks {
    fn lifetime(&self) -> &Arc<dyn toolsets::Lifetime> {
        &self.lifetime
    }
    fn setup(
        &self,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'_, Outcome> {
        Box::pin(async move {
            let setup = async {
                self.setups.fetch_add(1, Ordering::SeqCst);
                if let Some(children) = &self.children {
                    children.start().await?;
                }
                self.setup_entered.notify_one();
                if self.hold_setup {
                    self.setup_release.notified().await;
                }
                if self.fail_setup { Err(HostError::new("setup failed")) } else { Ok(()) }
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Err(HostError::new("setup cancelled")),
                result = setup => result,
            }
        })
    }
    fn close(&self) -> BoxFuture<'_, Outcome> {
        Box::pin(async move {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.close_entered.notify_one();
            if self.hold_close {
                self.close_release.notified().await;
            }
            if let Some(children) = &self.children {
                children.aclose().await?;
            }
            if self.fail_close { Err(HostError::new("close failed")) } else { Ok(()) }
        })
    }
    fn children(&self) -> Result<Option<Arc<dyn toolsets::LeaseGroup>>, HostError> {
        Ok(self.children.clone().map(|group| group as Arc<dyn toolsets::LeaseGroup>))
    }
}
fn group(hooks: Vec<Arc<Hooks>>) -> Arc<Group> {
    Arc::new(Group::new(
        &ProcessEnv::current_runtime(),
        hooks.into_iter().map(|hooks| hooks as _).collect(),
    ))
}

#[tokio::test]
async fn concurrent_scopes_share_setup_and_last_close() {
    let hooks = Arc::new(Hooks { hold_setup: true, ..Hooks::new() });
    let (a, b) = (group(vec![hooks.clone()]), group(vec![hooks.clone()]));
    let work = a.clone();
    let opening = TokioSpawner::current().spawn(async move { work.start().await });
    hooks.setup_entered.notified().await;
    let work = b.clone();
    let second = TokioSpawner::current().spawn(async move { work.start().await });
    hooks.setup_release.notify_one();
    opening.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(hooks.setups.load(Ordering::SeqCst), 1);
    a.aclose().await.unwrap();
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 0);
    b.aclose().await.unwrap();
    b.aclose().await.unwrap();
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failed_setup_rolls_back_children_and_earlier_roots_without_closing_failed_parent() {
    let (earlier, child) = (Arc::new(Hooks::new()), Arc::new(Hooks::new()));
    let parent = Arc::new(Hooks {
        fail_setup: true,
        children: Some(group(vec![child.clone()])),
        ..Hooks::new()
    });
    let scope = group(vec![earlier.clone(), parent.clone()]);
    assert!(scope.start().await.is_err());
    assert_eq!(earlier.closes.load(Ordering::SeqCst), 1);
    assert_eq!(child.closes.load(Ordering::SeqCst), 1);
    assert_eq!(parent.closes.load(Ordering::SeqCst), 0);
    scope.aclose().await.unwrap();
}

#[tokio::test]
async fn cancelling_last_setup_waiter_settles_child_rollback_before_close_returns() {
    let child = Arc::new(Hooks { hold_close: true, ..Hooks::new() });
    let parent = Arc::new(Hooks {
        hold_setup: true,
        children: Some(group(vec![child.clone()])),
        ..Hooks::new()
    });
    let scope = group(vec![parent.clone()]);
    let work = scope.clone();
    let opening = TokioSpawner::current().spawn(async move { work.start().await });
    parent.setup_entered.notified().await;
    opening.cancel();
    let _ = opening.await;
    child.close_entered.notified().await;
    let mut closing = Box::pin(scope.aclose());
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(closing.as_mut().poll(cx).is_pending()))
            .await
    );
    child.close_release.notify_one();
    closing.await.unwrap();
    assert_eq!(child.closes.load(Ordering::SeqCst), 1);
    assert_eq!(parent.closes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn one_close_failure_does_not_abandon_a_sibling_and_cancelled_close_waiter_can_rejoin() {
    let bad = Arc::new(Hooks { fail_close: true, ..Hooks::new() });
    let slow = Arc::new(Hooks { hold_close: true, ..Hooks::new() });
    let scope = group(vec![bad.clone(), slow.clone()]);
    scope.start().await.unwrap();
    let mut closing = Box::pin(scope.aclose());
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(closing.as_mut().poll(cx).is_pending()))
            .await
    );
    slow.close_entered.notified().await;
    drop(closing);
    slow.close_release.notify_one();
    assert!(scope.aclose().await.is_err());
    assert_eq!(bad.closes.load(Ordering::SeqCst), 1);
    assert_eq!(slow.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn session_roots_outlive_activation_and_duplicate_handles_acquire_once() {
    let hooks = Arc::new(Hooks::new());
    let scopes = Scopes::new(&ProcessEnv::current_runtime());
    scopes.acquire(0, vec![hooks.clone(), hooks.clone()]).await.unwrap();
    scopes.acquire(1, vec![hooks.clone()]).await.unwrap();
    scopes.release(Some(1)).await.unwrap();
    assert_eq!(hooks.setups.load(Ordering::SeqCst), 1);
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 0);
    scopes.release(None).await.unwrap();
    assert_eq!(hooks.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn refused_activation_does_not_release_an_older_use_of_the_same_agent() {
    let first = Arc::new(Hooks::new());
    let rejected = Arc::new(Hooks::new());
    let scopes = Scopes::new(&ProcessEnv::current_runtime());
    scopes.acquire(9, vec![first.clone()]).await.unwrap();
    let activation = scopes.acquire(9, vec![rejected.clone()]).await.unwrap();
    toolsets::ScopesPort::discard(&scopes, activation).await.unwrap();
    assert_eq!(rejected.closes.load(Ordering::SeqCst), 1);
    assert_eq!(first.closes.load(Ordering::SeqCst), 0);
    scopes.release(Some(9)).await.unwrap();
    assert_eq!(first.closes.load(Ordering::SeqCst), 1);
    scopes.release(None).await.unwrap();
    assert_eq!(rejected.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failed_admission_does_not_leave_a_slot_before_the_next_activation() {
    let failed = Arc::new(Hooks { fail_setup: true, ..Hooks::new() });
    let next = Arc::new(Hooks::new());
    let scopes = Scopes::new(&ProcessEnv::current_runtime());
    assert!(scopes.acquire(9, vec![failed]).await.is_err());
    scopes.acquire(9, vec![next.clone()]).await.unwrap();
    scopes.release(Some(9)).await.unwrap();
    assert_eq!(next.closes.load(Ordering::SeqCst), 1);
    scopes.release(None).await.unwrap();
}

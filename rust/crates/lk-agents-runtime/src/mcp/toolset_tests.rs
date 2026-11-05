use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::HostError;

use super::*;

/// A host toolset whose server answers once `reachable`; a `blocking` listing waits for its
/// cancellation.
#[derive(Default)]
struct Host {
    reachable: AtomicBool,
    blocking: AtomicBool,
    lists: AtomicUsize,
    cancelled: AtomicUsize,
    closes: AtomicUsize,
}

impl McpToolsetHost for Host {
    fn list<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        _context: &'a CallContext,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move {
            self.lists.fetch_add(1, Ordering::SeqCst);
            if self.blocking.load(Ordering::SeqCst) {
                cancel.cancelled().await;
                self.cancelled.fetch_add(1, Ordering::SeqCst);
                return Err(HostError::new("listing cancelled"));
            }
            match self.reachable.load(Ordering::SeqCst) {
                true => Ok(()),
                false => Err(HostError::new("server unreachable")),
            }
        })
    }

    fn retry_in(&self) -> Result<f64, HostError> {
        Ok(2.0)
    }

    fn close_idle<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        _context: &'a CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

struct Lifecycle(McpToolsetLifecycle, Arc<Host>);

fn lifecycle(host: &Arc<Host>) -> Lifecycle {
    Lifecycle(McpToolsetLifecycle::new("test".into(), 60.0), host.clone())
}

async fn setup(lifecycle: &Lifecycle) {
    lifecycle.0.setup(lifecycle.1.clone(), CallContext::default(), telemetry::Cx::NONE).await;
}

async fn release(lifecycle: &Lifecycle) {
    lifecycle.0.release(lifecycle.1.clone(), CallContext::default(), telemetry::Cx::NONE).await;
}

// Catches: an unreachable server failing setup, or never listed again once it answers.
#[tokio::test(start_paused = true)]
async fn an_unreachable_server_is_listed_again_at_the_sessions_backoff() {
    let host = Arc::new(Host::default());
    let lifecycle = lifecycle(&host);
    setup(&lifecycle).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 1);
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2, "one retry per backoff");
    host.reachable.store(true, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 3);
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 3, "recovered: no further listing");
    setup(&lifecycle).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 3, "a listed toolset is not listed again");
}

// Catches: closing after every run (a handshake per request), or never closing an idle one.
#[tokio::test(start_paused = true)]
async fn the_connection_closes_only_once_idle_after_the_last_release() {
    let host = Arc::new(Host { reachable: AtomicBool::new(true), ..Host::default() });
    let lifecycle = lifecycle(&host);
    setup(&lifecycle).await;
    release(&lifecycle).await;
    tokio::time::sleep(Duration::from_secs(30)).await;
    setup(&lifecycle).await;
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert_eq!(host.closes.load(Ordering::SeqCst), 0, "a setup inside the window disarms it");
    assert_eq!(host.lists.load(Ordering::SeqCst), 1, "and reuses the listing");
    release(&lifecycle).await;
    tokio::time::sleep(Duration::from_millis(59_900)).await;
    assert_eq!(host.closes.load(Ordering::SeqCst), 0);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(host.closes.load(Ordering::SeqCst), 1);
    setup(&lifecycle).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2, "a closed toolset lists again");
}

// Catches: a recovery outliving the toolset's last run, or its listing left running.
#[tokio::test(start_paused = true)]
async fn release_stops_a_recovery_and_settles_its_listing() {
    let host = Arc::new(Host::default());
    let lifecycle = lifecycle(&host);
    setup(&lifecycle).await;
    host.blocking.store(true, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(2100)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2, "the recovery's listing is pending");
    release(&lifecycle).await;
    assert_eq!(host.cancelled.load(Ordering::SeqCst), 1, "cancelled and settled by release");
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2);
}

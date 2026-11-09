use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::mcp::McpSession as _;

use super::*;

/// A host toolset whose server answers once `reachable`; a `blocking` listing waits for its
/// cancellation.
struct Host {
    reachable: AtomicBool,
    blocking: AtomicBool,
    lists: AtomicUsize,
    cancelled: AtomicUsize,
    closes: AtomicUsize,
    session: Arc<super::super::session::McpSession>,
    server: Arc<super::super::session::tests::Server>,
}

impl Default for Host {
    fn default() -> Self {
        let server = Arc::new(super::super::session::tests::Server::default());
        server.refuse.store(true, Ordering::SeqCst);
        Self {
            reachable: AtomicBool::new(false),
            blocking: AtomicBool::new(false),
            lists: AtomicUsize::new(0),
            cancelled: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            session: Arc::new(super::super::session::McpSession::new(10.0, None, None).unwrap()),
            server,
        }
    }
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
            if self.reachable.load(Ordering::SeqCst) {
                return Ok(());
            }
            let server = self.server.clone();
            self.session
                .connect_fake(move || server.connect())
                .await
                .map_err(|error| HostError::new(error.to_string()))
        })
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
    Lifecycle(McpToolsetLifecycle::new("test".into(), host.session.clone()), host.clone())
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
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 1, "the native session is backing off");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2, "the first native backoff expired");
    tokio::time::sleep(Duration::from_millis(1800)).await;
    assert_eq!(host.lists.load(Ordering::SeqCst), 2, "the next native backoff is two seconds");
    host.reachable.store(true, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(200)).await;
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

use super::*;
use crate::ipc::ipc_channel::{MAGIC, VERSION, memory};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::AsyncWriteExt;

type WatchedChannel = (
    tokio::sync::mpsc::Receiver<ToChild>,
    lk_agents_abi::OwnedTask<Result<(), ChannelError>>,
    Channel<ToChild, ToParent>,
    Arc<AtomicUsize>,
);

/// The child's watch over a channel whose parent end the test drives.
fn watched(health_timeout: Option<Duration>) -> WatchedChannel {
    let (parent, child) = memory();
    let orphaned = Arc::new(AtomicUsize::new(0));
    let count = orphaned.clone();
    let (inbox, watching) = watch_parent(Arc::new(child), health_timeout, move || {
        count.fetch_add(1, Ordering::SeqCst);
    });
    (inbox, watching, parent, orphaned)
}

// Catches: an orphaned child (the parent stopped talking) never noticing.
#[tokio::test(start_paused = true)]
async fn the_child_calls_itself_orphaned_when_the_parent_goes_quiet() {
    let (mut inbox, watching, parent, orphaned) = watched(Some(Duration::from_secs(100)));
    let start = Instant::now();
    tokio::time::sleep(Duration::from_secs(60)).await;
    parent.send(ToChild::Ping { sent_ms: 1 }).await.unwrap();
    assert_eq!(inbox.recv().await, Some(ToChild::Ping { sent_ms: 1 }));
    assert_eq!(inbox.recv().await, None);
    let quiet = start.elapsed();
    assert!(quiet >= Duration::from_secs(160) && quiet < Duration::from_secs(161), "{quiet:?}");
    assert_eq!(orphaned.load(Ordering::SeqCst), 1);
    assert!(matches!(watching.await.unwrap(), Err(ChannelError::Io { .. })));

    let (mut inbox, _watching, _parent, _) = watched(None);
    let pending = tokio::time::timeout(Duration::from_secs(10_000), inbox.recv()).await;
    assert!(pending.is_err(), "no health check without a ping deadline");
}

// Catches: a malformed frame from the parent read past, instead of ending the connection.
#[tokio::test]
async fn the_child_stops_at_a_malformed_frame() {
    let (parent, child) = tokio::io::duplex(1 << 16);
    let child = Arc::new(Channel::<ToParent, ToChild>::new(child));
    let (mut inbox, watching) = watch_parent(child, None, || {});
    let mut parent = parent;
    parent.write_all(&MAGIC).await.unwrap();
    parent.write_all(&VERSION.to_be_bytes()).await.unwrap();
    parent.write_all(&[0, 0, 0, 1, 99]).await.unwrap();
    assert_eq!(inbox.recv().await, None);
    assert!(matches!(watching.await.unwrap(), Err(ChannelError::Malformed(_))));
}

// Catches: a busy child loop (an undrained inbox) suspending the health check.
#[tokio::test(start_paused = true)]
async fn a_full_child_inbox_does_not_suspend_the_health_check() {
    let (_inbox, watching, parent, orphaned) = watched(Some(Duration::from_millis(100)));
    for _ in 0..65 {
        parent.send(ToChild::Ping { sent_ms: 0 }).await.unwrap();
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(watching.is_finished());
    assert_eq!(orphaned.load(Ordering::SeqCst), 1);
}

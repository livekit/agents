use super::*;
use lk_agents_abi::{MediaRef, RefRegistry};
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

/// Everything queued now, without waiting.
fn drain<E>(rx: &mut VideoReceiver<E>) -> Vec<Item<VideoFrameRef, E>> {
    std::iter::from_fn(|| match pin!(rx.recv()).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(item) => item,
        Poll::Pending => None,
    })
    .collect()
}

/// A video frame numbered `n`.
fn frame(n: u64) -> VideoFrameRef {
    struct Noop;
    impl RefRegistry for Noop {
        fn release(&self, _: u64) {}
    }
    let registry: Arc<dyn RefRegistry> = Arc::new(Noop);
    VideoFrameRef { media: MediaRef::new(n, &registry), width: 2, height: 2, timestamp_us: 0 }
}

fn ids(items: Vec<Item<VideoFrameRef, &'static str>>) -> Vec<Option<u64>> {
    items
        .into_iter()
        .map(|item| match item {
            Item::Media(frame) => Some(frame.media.id()),
            Item::Event(_) => None,
            Item::Gap(_) => panic!("video never reports a gap"),
        })
        .collect()
}

// Catches `mpsc::try_send`'s behaviour on video: rejecting the newest frame when full, merging,
// or a gap.
#[test]
fn a_full_video_queue_drops_the_oldest_frame() {
    let (tx, mut rx) = video::<&str>(3);
    for n in 1..=5 {
        tx.push(frame(n));
    }
    assert_eq!(rx.dropped(), 2);
    assert_eq!(ids(drain(&mut rx)), [Some(3), Some(4), Some(5)]);
}

// Catches: a video flush keeping frames from before it, or the bound reaching past an event.
#[test]
fn a_video_flush_drops_the_queued_frames() {
    let (tx, mut rx) = video::<&str>(2);
    tx.push(frame(1));
    tx.event("mark").unwrap();
    tx.push(frame(2));
    tx.push(frame(3));
    tx.push(frame(4));
    assert_eq!(ids(drain(&mut rx)), [Some(1), None, Some(3), Some(4)]);
    tx.push(frame(5));
    tx.flush("changed").unwrap();
    tx.push(frame(6));
    assert_eq!(ids(drain(&mut rx)), [None, Some(6)]);
}

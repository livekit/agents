use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::task::Poll;

pub(super) async fn pending<F: Future>(mut future: Pin<&mut F>) {
    poll_fn(|context| {
        assert!(future.as_mut().poll(context).is_pending(), "operation should remain pending");
        Poll::Ready(())
    })
    .await;
}

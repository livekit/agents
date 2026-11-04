//! The SDK's frame queue (`rtc.AudioStream`/`rtc.VideoStream`'s `RingQueue`): drop-oldest by
//! count, imitated here rather than borrowed from the runtime, so the fake cannot share a bug with
//! what it tests.

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, StreamOutput, Ts};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

struct Ring<T> {
    items: VecDeque<T>,
    capacity: usize,
    sender: bool,
    receiver: bool,
    waker: Option<Waker>,
}

pub fn sdk_queue<T>(capacity: usize) -> (SdkTx<T>, SdkRx<T>) {
    let ring = Arc::new(Mutex::new(Ring {
        items: VecDeque::new(),
        capacity: capacity.max(1),
        sender: true,
        receiver: true,
        waker: None,
    }));
    (SdkTx(ring.clone()), SdkRx(ring))
}

/// Dropping it ends the stream.
pub struct SdkTx<T>(Arc<Mutex<Ring<T>>>);

impl<T> SdkTx<T> {
    /// `false` once the reader closed its stream.
    pub fn push(&self, item: T) -> bool {
        let waker = {
            let mut ring = self.0.lock().unwrap();
            if !ring.receiver {
                return false;
            }
            if ring.items.len() >= ring.capacity {
                ring.items.pop_front();
            }
            ring.items.push_back(item);
            ring.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        true
    }
}

impl<T> Drop for SdkTx<T> {
    fn drop(&mut self) {
        let waker = {
            let mut ring = self.0.lock().unwrap();
            ring.sender = false;
            ring.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

pub struct SdkRx<T>(Arc<Mutex<Ring<T>>>);

impl<T> SdkRx<T> {
    pub async fn recv(&mut self) -> Option<T> {
        std::future::poll_fn(|cx| {
            let mut ring = self.0.lock().unwrap();
            if let Some(item) = ring.items.pop_front() {
                Poll::Ready(Some(item))
            } else if !ring.sender {
                Poll::Ready(None)
            } else {
                ring.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await
    }
}

impl<T> Drop for SdkRx<T> {
    fn drop(&mut self) {
        let mut ring = self.0.lock().unwrap();
        ring.receiver = false;
        ring.items.clear();
    }
}

impl<T: Send + 'static> StreamOutput<T> for SdkRx<T> {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.recv().await) })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}
